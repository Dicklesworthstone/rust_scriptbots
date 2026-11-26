import { captureLocalStudy, createLocalStudyFile, readLocalStudyFile, validateLocalStudyResult } from "./local-study.js";

export function attachLocalStudy(panel, { getOwner, getAction, getLimit, createWorker }) {
  const field = name => panel.querySelector(`#local-study-${name}`);
  const run = field("run"), cancel = field("cancel"), status = field("status"), table = field("metrics");
  const horizon = field("ticks"), delay = field("delay");
  horizon.addEventListener("input", () => {
    delay.max = horizon.validity.valid && Number.isInteger(horizon.valueAsNumber)
      ? String(horizon.valueAsNumber - 1) : "0";
  });
  const exportLink = field("export");
  const fileExport = field("file-export"), fileInput = field("import"), replay = field("replay"), fileStatus = field("file-status");
  const samplePicker = field("sample"), sampleControls = field("samples"), sampleLabel = field("sample-label");
  let task = null, url = null, fileUrl = null, available = false, replayAvailable = false, report = null, imported = null;

  function controls() {
    run.disabled = !available || task !== null;
    cancel.disabled = task === null;
    fileInput.disabled = !replayAvailable || task !== null;
    replay.disabled = !replayAvailable || !imported || task !== null;
  }

  function clearReport() {
    report = null;
    sampleControls.hidden = true;
    table.replaceChildren();
    exportLink.hidden = true;
    exportLink.removeAttribute("href");
    fileExport.hidden = true;
    fileExport.removeAttribute("href");
    if (url) URL.revokeObjectURL(url);
    if (fileUrl) URL.revokeObjectURL(fileUrl);
    url = fileUrl = null;
  }
  function stop() {
    const previous = task;
    task = null;
    previous?.abort.abort();
    previous?.worker?.terminate();
    controls();
  }
  function reset() {
    stop();
    imported = null;
    fileInput.value = "";
    fileStatus.textContent = "Choose a saved study to inspect its source and repeat it on independent copies.";
    clearReport();
    status.textContent = "";
    controls();
  }
  function showSample() {
    if (!report) return;
    const index = Number(samplePicker.value);
    if (!Number.isSafeInteger(index) || index < 0 || index >= report.control.samples.length) return;
    const control = report.control.samples[index], treatment = report.treatment.samples[index];
    const tick = BigInt(control.tick), queueTick = BigInt(report.treatment.receipt.applicationBoundary.tick);
    const phase = tick === queueTick ? "treatment queued here" : tick < queueTick ? "before treatment" : "after treatment";
    sampleLabel.textContent = `Scientific tick ${control.tick} · observation ${index + 1} of ${report.control.samples.length} · ${phase}`;
    table.replaceChildren();
    const row = (tag, values) => {
      const tr = table.insertRow();
      for (const value of values) {
        const cell = panel.ownerDocument.createElement(tag);
        cell.textContent = value;
        tr.append(cell);
      }
    };
    row("th", ["Measure", "Control", "Treatment", "Treatment − control"]);
    for (const [key, label] of [["population", "Population"], ["averageEnergy", "Mean energy"], ["averageHealth", "Mean health"]]) {
      const format = n => n === null ? "—" : key === "population" ? String(n) : String(Number(n.toPrecision(6)));
      const diff = control[key] === null || treatment[key] === null ? null : treatment[key] - control[key];
      row("td", [label, format(control[key]), format(treatment[key]), format(diff)]);
    }
  }
  samplePicker.addEventListener("input", showSample);
  function showResult(result, file) {
    report = result;
    const last = result.control.samples.length - 1;
    samplePicker.max = String(last);
    samplePicker.value = String(last);
    sampleControls.hidden = false;
    showSample();
    status.textContent = `One saved world · ${result.action.kind} queued at tick ${result.treatment.receipt.applicationBoundary.tick} · tick ${result.source.tick} to ${result.control.finalTick}. Differences describe this pair of runs.`;
    url = URL.createObjectURL(new Blob([JSON.stringify(result, null, 2)], { type: "application/json" }));
    exportLink.href = url;
    exportLink.download = `scriptbots-local-study-${result.source.tick}-${result.control.finalTick}.json`;
    exportLink.hidden = false;
    if (file) {
      fileUrl = URL.createObjectURL(file);
      fileExport.href = fileUrl;
      fileExport.download = `scriptbots-local-study-${result.source.tick}-${result.control.finalTick}.sbstudy`;
      fileExport.hidden = false;
    } else {
      status.textContent += " Portable-file export requires Web Crypto on a secure connection.";
    }
  }

  async function launch(prepare) {
    if (task) return;
    clearReport();
    const current = { kind: "study", abort: new AbortController(), worker: null };
    task = current;
    controls();
    status.textContent = "Preparing the saved source and exact study inputs…";
    try {
      const { job, checkpoint, file } = await prepare(current.abort.signal);
      if (task !== current) return;
      const { bytes: _, ...expected } = job;
      const worker = createWorker();
      current.worker = worker;
      status.textContent = `Restoring the control at tick ${job.source.tick}…`;
      worker.onmessage = ({ data }) => {
        if (task !== current) return;
        if (data?.progress) {
          const { arm, done, total } = data.progress;
          if (["control", "treatment"].includes(arm) && total === expected.ticks
              && Number.isSafeInteger(done) && done >= 0 && done <= total) {
            status.textContent = `${arm} · ${done} / ${total} ticks`;
          }
        } else {
          try {
            if (data.error) throw new Error(String(data.error).slice(0, 1024));
            const result = validateLocalStudyResult(data.result, expected);
            stop();
            showResult({ ...result, checkpoint }, file);
          } catch (error) {
            stop();
            clearReport();
            status.textContent = `Study failed: ${String(error.message ?? error).slice(0, 1024)}`;
          }
        }
      };
      worker.onerror = worker.onmessageerror = () => {
        if (task !== current) return;
        stop();
        status.textContent = "The study worker failed. Check that the browser package is available.";
      };
      worker.postMessage(job, [job.bytes.buffer]);
    } catch (error) {
      if (task !== current) return;
      stop();
      status.textContent = `Study unavailable: ${String(error.message ?? error).slice(0, 1024)}`;
    }
  }

  run.addEventListener("click", () => {
    if (!available || task) return;
    return launch(signal => {
      const owner = getOwner();
      const job = captureLocalStudy(owner, getAction(owner.frame()), horizon.valueAsNumber, getLimit(), delay.valueAsNumber);
      return globalThis.crypto?.subtle ? createLocalStudyFile(job, getLimit(), { signal })
        : { job, checkpoint: { bytes: job.bytes.byteLength, sha256: null }, file: null };
    });
  });

  fileInput.addEventListener("change", async () => {
    if (!replayAvailable || task) return;
    const file = fileInput.files?.[0];
    // Permit choosing the same file after a refusal or cancelled read.
    fileInput.value = "";
    imported = null;
    controls();
    if (!file) { fileStatus.textContent = "No saved study selected."; return; }
    const current = { kind: "import", abort: new AbortController(), worker: null };
    task = current;
    controls();
    fileStatus.textContent = "Checking the saved study and checkpoint…";
    try {
      const loaded = await readLocalStudyFile(file, getLimit(), { signal: current.abort.signal });
      if (task !== current) return;
      // Retain the immutable file, not a second long-lived copy of its checkpoint buffer.
      imported = file;
      const queueTick = BigInt(loaded.job.source.tick) + BigInt(loaded.job.delayTicks);
      fileStatus.textContent = `Saved ${loaded.job.action.kind} comparison · source tick ${loaded.job.source.tick} · ${loaded.job.ticks} ticks per arm · treatment queued at tick ${queueTick} after ${loaded.job.delayTicks} ticks · checkpoint SHA-256 ${loaded.checkpoint.sha256}. Ready to repeat; no world has been restored.`;
      stop();
    } catch (error) {
      if (task !== current) return;
      stop();
      fileStatus.textContent = `Study file refused: ${String(error.message ?? error).slice(0, 1024)}`;
    }
  });

  replay.addEventListener("click", () => {
    if (!replayAvailable || !imported || task) return;
    const selected = imported;
    return launch(signal => readLocalStudyFile(selected, getLimit(), { signal }));
  });
  cancel.addEventListener("click", () => {
    if (task?.kind === "import") fileStatus.textContent = "Study file check cancelled. Choose the file again to retry.";
    stop();
    status.textContent = "Study cancelled.";
  });
  return { reset, setAvailable(value, canReplay = value) {
    available = value;
    replayAvailable = canReplay;
    controls();
  } };
}
