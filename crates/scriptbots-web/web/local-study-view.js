import { captureLocalStudy } from "./local-study.js";

export function attachLocalStudy(panel, { getOwner, getAction, getLimit, createWorker }) {
  const field = name => panel.querySelector(`#local-study-${name}`);
  const run = field("run"), cancel = field("cancel"), status = field("status"), table = field("metrics");
  const horizon = field("ticks"), delay = field("delay");
  horizon.addEventListener("input", () => {
    delay.max = horizon.validity.valid && Number.isInteger(horizon.valueAsNumber)
      ? String(horizon.valueAsNumber - 1) : "0";
  });
  const exportLink = field("export");
  const samplePicker = field("sample"), sampleControls = field("samples"), sampleLabel = field("sample-label");
  let worker = null, url = null, available = false, report = null;

  function clearReport() {
    report = null;
    sampleControls.hidden = true;
    table.replaceChildren();
    exportLink.hidden = true;
    exportLink.removeAttribute("href");
    if (url) URL.revokeObjectURL(url);
    url = null;
  }
  function stop() {
    worker?.terminate();
    worker = null;
    run.disabled = !available;
    cancel.disabled = true;
  }
  function reset() {
    stop();
    clearReport();
    status.textContent = "";
  }
  function showSample() {
    if (!report) return;
    const index = Number(samplePicker.value);
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
  function showResult(result) {
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
  }
  run.addEventListener("click", () => {
    if (!available || worker) return;
    clearReport();
    try {
      const owner = getOwner();
      const job = captureLocalStudy(owner, getAction(owner.frame()), horizon.valueAsNumber, getLimit(), delay.valueAsNumber);
      const current = createWorker();
      worker = current;
      cancel.disabled = false;
      run.disabled = true;
      status.textContent = `Restoring the control at tick ${job.source.tick}…`;
      current.onmessage = ({ data }) => {
        if (worker !== current) return;
        if (data.progress) {
          status.textContent = `${data.progress.arm} · ${data.progress.done} / ${data.progress.total} ticks`;
        } else {
          stop();
          if (data.error) status.textContent = `Study failed: ${data.error}`;
          else showResult(data.result);
        }
      };
      current.onerror = current.onmessageerror = () => {
        if (worker !== current) return;
        stop();
        status.textContent = "The study worker failed. Check that the browser package is available.";
      };
      current.postMessage(job, [job.bytes.buffer]);
    } catch (error) {
      stop();
      status.textContent = `Study unavailable: ${String(error.message ?? error).slice(0, 1024)}`;
    }
  });
  cancel.addEventListener("click", () => {
    stop();
    status.textContent = "Study cancelled.";
  });
  return { reset, setAvailable(value) {
    available = value;
    run.disabled = !available || !!worker;
  } };
}
