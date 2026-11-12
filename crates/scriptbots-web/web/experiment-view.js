import { EXPERIMENT_ENDPOINT_METRICS } from "./experiment-protocol.js";

const MAX_SEED = 18446744073709551615n;

function field(form, name) {
  const input = form.querySelector("#experiment-" + name);
  if (!input) throw new Error("Experiment form is missing " + name + ".");
  return input.value;
}

function integer(value, label, maximum) {
  if (!/^[0-9]+$/.test(value.trim())) throw new Error(label + " must be a positive whole number.");
  const parsed = Number(value.trim());
  if (!Number.isSafeInteger(parsed) || parsed < 1 || parsed > maximum) {
    throw new Error(label + " must be between 1 and " + maximum + ".");
  }
  return parsed;
}

function jsonInput(value, label, array = false) {
  if (value.length > 16384) throw new Error(label + " exceeds the 16 KiB editor limit.");
  const source = value.trim();
  let parsed;
  try { parsed = source ? JSON.parse(source) : (array ? [] : {}); }
  catch { throw new Error(label + " must be valid JSON."); }
  if (array ? !Array.isArray(parsed) : (!parsed || typeof parsed !== "object" || Array.isArray(parsed))) {
    throw new Error(label + (array ? " must be a JSON array." : " must be a JSON object."));
  }
  return parsed;
}

function arm(form, prefix) {
  const variantId = field(form, prefix + "-id").trim();
  if (!/^[A-Za-z0-9_-]{1,64}$/.test(variantId)) {
    throw new Error("Each arm ID must contain 1–64 letters, digits, underscores, or hyphens.");
  }
  const family = field(form, prefix + "-family");
  if (!["mlp", "dwraon", "assembly", "ft"].includes(family)) throw new Error("Choose an available brain family.");
  return {
    variant_id: variantId,
    brain_family: family,
    config_overrides: jsonInput(field(form, prefix + "-config"), prefix + " overrides"),
    interventions: jsonInput(field(form, prefix + "-schedule"), prefix + " schedule", true),
  };
}

/** Keep seeds as exact decimal strings until the protocol serializer emits u64 tokens. */
export function readExperimentForm(form) {
  const seedText = field(form, "seeds").trim();
  if (seedText.length > 2048) throw new Error("Seed input is too long.");
  const seeds = seedText.split(/[\s,]+/).filter(Boolean);
  if (seeds.length < 1 || seeds.length > 32) throw new Error("Enter 1–32 distinct decimal seeds.");
  for (const seed of seeds) {
    if (!/^(0|[1-9][0-9]*)$/.test(seed) || seed.length > 20 || BigInt(seed) > MAX_SEED) {
      throw new Error("Seeds must be exact unsigned 64-bit decimal integers, without exponents or leading zeroes.");
    }
  }
  if (new Set(seeds).size !== seeds.length) throw new Error("Each seed must be distinct.");
  const experimentId = field(form, "id").trim();
  if (experimentId && !/^[A-Za-z0-9_-]{1,128}$/.test(experimentId)) {
    throw new Error("Experiment ID must contain 1–128 letters, digits, underscores, or hyphens.");
  }
  const description = field(form, "description").trim();
  if (description.length > 1024) throw new Error("Description must not exceed 1024 characters.");
  const variants = [arm(form, "baseline"), arm(form, "challenger")];
  if (variants[0].variant_id === variants[1].variant_id) throw new Error("The two arm IDs must differ.");
  return {
    experiment_id: experimentId || null,
    description: description || null,
    seeds,
    ticks_per_run: integer(field(form, "ticks"), "Horizon", 10000000),
    max_concurrency: integer(field(form, "concurrency"), "Concurrency", 8),
    variants,
  };
}

function node(document, tag, text) {
  const result = document.createElement(tag);
  if (text !== undefined) result.textContent = String(text);
  return result;
}

function setText(panel, id, value) {
  const target = panel.querySelector("#experiment-" + id);
  if (target) target.textContent = value;
}

function disable(panel, id, disabled) {
  const target = panel.querySelector("#experiment-" + id);
  if (target) target.disabled = disabled;
}

function choices(select, entries, preferred) {
  if (!select) return;
  const previous = preferred === undefined ? select.value : preferred;
  const signature = JSON.stringify(entries);
  if (select.dataset.options !== signature) {
    select.replaceChildren(...entries.map(([value, label]) => {
      const item = node(select.ownerDocument, "option", label);
      item.value = value;
      return item;
    }));
    select.dataset.options = signature;
  }
  select.value = entries.some(([value]) => value === previous) ? previous : (entries[0]?.[0] || "");
}

function number(value) {
  if (value === null || value === undefined || !Number.isFinite(value)) return "unavailable";
  return String(Number(value.toPrecision(6)));
}

function renderRuns(panel, controller) {
  const target = panel.querySelector("#experiment-runs");
  if (!target) return;
  const rows = controller.visibleRuns || [];
  const signature = JSON.stringify(rows);
  if (target.dataset.rows === signature) return;
  target.dataset.rows = signature;
  target.replaceChildren(...rows.map((run) => {
    const item = node(target.ownerDocument, "li");
    item.append(node(target.ownerDocument, "strong", run.variant_id + " · seed " + run.seed));
    item.append(node(target.ownerDocument, "p", run.brain_family + " · " + run.state + " · target " + run.target_ticks + " ticks"));
    if (run.state === "completed") {
      item.append(node(target.ownerDocument, "p", "Recorded terminal endpoint: " + run.total_ticks + " ticks."));
    }
    const progress = run.checkpoint_progress;
    if (progress?.state === "observed") {
      item.append(node(target.ownerDocument, "p", "Saved checkpoint observed at tick " + progress.tick + " (" + progress.source + ")."));
      item.append(node(target.ownerDocument, "p", "Checkpoint digest: " + progress.checkpoint_digest));
    } else {
      item.append(node(target.ownerDocument, "p", "Saved progress " + (progress?.state || "unavailable") + ": " + (progress?.reason || "No checkpoint observation is available.")));
    }
    if (run.error_reason) item.append(node(target.ownerDocument, "p", "Run failure: " + run.error_reason));
    if (run.final_digest) item.append(node(target.ownerDocument, "p", "Final digest: " + run.final_digest));
    if (run.bundle_path) item.append(node(target.ownerDocument, "p", "Server bundle path: " + run.bundle_path));
    return item;
  }));
}

function renderComparison(panel, comparison) {
  const target = panel.querySelector("#experiment-comparison");
  if (!target) return;
  target.hidden = !comparison;
  if (!comparison) {
    target.replaceChildren();
    delete target.dataset.comparison;
    return;
  }
  const signature = JSON.stringify(comparison);
  if (target.dataset.comparison === signature) return;
  target.dataset.comparison = signature;
  const document = target.ownerDocument;
  const count = comparison.matched_seeds.length;
  const results = [
    node(document, "h4", comparison.challenger_variant + " minus " + comparison.baseline_variant),
    node(document, "p", comparison.metric + " at terminal tick " + comparison.endpoint_tick + " · " + count + " matched seed pair" + (count === 1 ? "" : "s") + "."),
    node(document, "p", "Mean paired difference: " + number(comparison.mean_difference)),
    node(document, "p", "A larger endpoint value is not a fitness verdict. Each distinct seed contributes one terminal pair; ticks are not independent replicates."),
  ];
  if (comparison.descriptive_only) {
    results.push(node(document, "p", "Descriptive only: inferential statistics are unavailable for this comparison."));
  } else {
    results.push(node(document, "p", "95% interval: " + (comparison.ci_95 ? "[" + number(comparison.ci_95[0]) + ", " + number(comparison.ci_95[1]) + "]" : "unavailable")));
    results.push(node(document, "p", "Standardized paired effect: " + number(comparison.standardized_effect)));
    results.push(node(document, "p", "p-value: " + number(comparison.p_value) + " · adjusted p-value (server-reported): " + number(comparison.adjusted_p_value)));
    results.push(node(document, "p", "Server significance flag: " + (comparison.significant === null ? "unavailable" : comparison.significant ? "true" : "false") + ". Separate metric requests do not constitute one jointly corrected analysis."));
  }
  if (comparison.underpowered_reason) results.push(node(document, "p", "Uncertainty: " + comparison.underpowered_reason));
  if (comparison.undefined_effect_reason) results.push(node(document, "p", "Effect limitation: " + comparison.undefined_effect_reason));
  const seeds = node(document, "details");
  seeds.append(node(document, "summary", "Matched decimal seeds"), node(document, "p", comparison.matched_seeds.join(", ")));
  results.push(seeds);
  results.push(node(document, "p", "Observed plan digest: " + comparison.plan_digest + " · status generation " + comparison.status_generation + "."));
  target.replaceChildren(...results);
}

export function renderExperimentLab(panel, controller) {
  if (!panel) return;
  const selected = controller.selected;
  const page = controller.page;
  setText(panel, "source", controller.enabled && controller.source
    ? "Connected host session " + controller.source.sessionId + "."
    : "Experiment controls are inactive. Connect a remote host, wait for its frame, and open this panel. Retained observations remain visible.");
  setText(panel, "notice", controller.notice || "Create independent worlds with the same seed set, or open a retained experiment.");
  const entries = (page?.items || []).map((item) => [item.experiment_id, item.experiment_id + " · " + item.status + " · " + item.completed_runs + "/" + item.total_runs + " complete"]);
  if (selected && !entries.some(([id]) => id === selected.experiment_id)) {
    entries.unshift([selected.experiment_id, selected.experiment_id + " · selected (outside this catalog page)"]);
  }
  if (!entries.length) entries.push(["", "No retained experiments observed"]);
  else if (!selected) entries.unshift(["", "Choose a retained experiment"]);
  choices(panel.querySelector("#experiment-select"), entries, selected?.experiment_id || "");
  disable(panel, "select", !controller.canSelect);
  setText(panel, "catalog-detail", page
    ? "Catalog page: " + page.items.length + " observed of " + page.total + " retained experiments; offset " + page.offset + "."
    : "Refresh to observe the retained experiment catalog.");
  const capacity = page?.execution_capacity;
  setText(panel, "capacity", capacity
    ? "Host reservations: " + capacity.reserved_run_slots + "/" + capacity.max_run_slots + " run slots; " + capacity.reserved_grid_cells + "/" + capacity.max_grid_cells + " grid cells; " + capacity.reserved_batches + " batches."
    : "Host capacity has not been observed.");
  const creation = controller.creation;
  const admissionText = {
    pending: "Admission is pending. The exact submitted request is retained.",
    unconfirmed: "Admission is unconfirmed. Use Retry exact request to resolve this same request; it does not invent a new request.",
    refused: "The host refused admission. Review the reported reason before creating another request.",
    observed: "The host admission was observed" + (creation?.experimentId ? " for " + creation.experimentId : "") + ".",
  };
  setText(panel, "admission", creation ? (admissionText[creation.state] || "Admission state unavailable.") : "No request has been submitted from this browser session.");
  setText(panel, "status", selected
    ? selected.experiment_id + " · " + selected.status + " · " + selected.completed_runs + "/" + selected.total_runs + " complete · " + selected.failed_runs + " failed · generation " + selected.generation + "."
    : "Choose or create an experiment to observe its runs.");
  setText(panel, "batch-error", selected?.error_reason ? "Batch failure: " + selected.error_reason : "");
  setText(panel, "run-page", selected
    ? "Run page " + (controller.runPageIndex + 1) + " of " + controller.totalRunPages + " · at most 16 runs shown."
    : "No run page selected.");
  for (const [id, allowed] of [
    ["catalog-refresh", controller.canOperate], ["catalog-first", controller.canOperate && !!page && page.offset > 0],
    ["catalog-next", controller.canOperate && !!page?.has_more], ["launch", controller.canLaunch],
    ["retry", controller.canRetry], ["refresh", controller.canOperate && !!selected],
    ["cancel", controller.canCancel], ["resume", controller.canResume], ["compare", controller.canCompare],
    ["run-previous", !!selected && controller.runPageIndex > 0],
    ["run-next", !!selected && controller.runPageIndex + 1 < controller.totalRunPages],
    ["export", !!selected || !!controller.comparison],
  ]) disable(panel, id, !allowed);
  renderRuns(panel, controller);
  const cohorts = new Map();
  for (const run of selected?.runs || []) {
    const prior = cohorts.get(run.variant_id);
    cohorts.set(run.variant_id, (prior === undefined || prior) && run.state === "completed");
  }
  const completed = [...cohorts].filter(([, ready]) => ready).map(([id]) => [id, id]);
  const baseline = panel.querySelector("#experiment-compare-baseline");
  const challenger = panel.querySelector("#experiment-compare-challenger");
  const oldChallenger = challenger?.value;
  choices(baseline, completed);
  choices(challenger, completed);
  if (challenger && !completed.some(([id]) => id === oldChallenger)) {
    challenger.value = completed.find(([id]) => id !== baseline?.value)?.[0] || completed[0]?.[0] || "";
  }
  choices(panel.querySelector("#experiment-metric"), EXPERIMENT_ENDPOINT_METRICS.map((metric) => [metric, metric]));
  for (const id of ["compare-baseline", "compare-challenger", "metric"]) disable(panel, id, !controller.canCompare);
  renderComparison(panel, controller.comparison);
}

/** Browser observations are useful provenance, but are not the host's audited run bundle. */
export function experimentObservationExport(controller) {
  if (!controller.selected && !controller.comparison) throw new Error("Observe an experiment before exporting.");
  const admitted = controller.creation?.state === "observed"
    && controller.creation.experimentId === controller.selected?.experiment_id;
  const payload = {
    schema: "scriptbots.browser-experiment-observation.v1",
    scope: "Browser-observed request, status, and comparison; this is not a server run bundle.",
    source_session_id: controller.source?.sessionId || null,
    request_json: admitted ? controller.creation.prepared.body : null,
    request_context: admitted
      ? "Exact request admitted for the selected experiment."
      : "No matching admitted request is retained; the selected plan cannot be reconstructed from these observations.",
    admission_state: admitted ? controller.creation.state : null,
    status: controller.selected || null,
    comparison: controller.comparison || null,
    seed_encoding: "Status and comparison seeds are decimal strings; request_json preserves exact numeric JSON seed tokens.",
  };
  const content = JSON.stringify(payload, null, 2) + "\n";
  const blob = new Blob([content], { type: "application/json;charset=utf-8" });
  if (blob.size > 2 * 1024 * 1024) throw new Error("Observed export exceeds the 2 MiB browser export limit.");
  return { filename: "scriptbots-experiment-observation.json", blob };
}
