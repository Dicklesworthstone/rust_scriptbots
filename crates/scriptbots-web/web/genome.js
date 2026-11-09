import { InspectionController, exactIdentity, inspectionStatus, validateInspectionSource } from "./inspector.js";

// These are independent ceilings, not page sizes supplied by a remote server.
export const GENOME_LIMITS = Object.freeze({ maxSourceLoci: 65536, locusPageSize: 64,
  deltaPageSize: 32, maxJsonBytes: 65536 });
const DIGEST = /^[0-9a-f]{64}$/;
const UNAVAILABLE = new Set(["missing_agent", "unbound_brain", "genome_unavailable", "unsupported",
  "source_budget_exceeded", "invalid_genome", "page_out_of_range"]);
const PARENT_UNAVAILABLE = new Set(["parent_unavailable", "material_mismatch", "family_mismatch",
  "schema_mismatch", "unsupported", "source_budget_exceeded", "invalid_genome", "incompatible_loci", "page_out_of_range"]);
const DERIVATIONS = new Map([["Founder", 0], ["Clone", 1], ["MutationOnly", 1],
  ["Crossover", 2], ["CrossoverThenMutation", 2]]);
const count = (value, limit = 0xffff_ffff) => Number.isSafeInteger(value) && value >= 0 && value <= limit;
const tagged = (value) => value && typeof value === "object" && !Array.isArray(value)
  && Object.keys(value).length === 1 ? Object.entries(value)[0] : [];

function validateLocus(locus) {
  const [kind, index] = tagged(locus);
  if (["NodeBias", "NodeDamping", "NodeGain", "Cell"].includes(kind) && count(index)) return;
  if (kind === "Hyper" && count(index, 255)) return;
  if (["NodeWeight", "NodeKind", "NodeTarget"].includes(kind) && count(index?.node)
      && count(index?.conn, 255) && Object.keys(index).length === 2) return;
  throw new Error("Invalid canonical genome locus");
}

function validateValue(value) {
  const [kind, data] = tagged(value);
  if ((kind === "Scalar" && Number.isFinite(data)) || (kind === "Target" && count(data))
      || (kind === "Kind" && count(data, 255))) return;
  throw new Error("Invalid canonical genome value");
}

function validatePaging(page, requested, size, rows) {
  if (!page || page.offset !== requested || page.limit !== size || !count(page.total, GENOME_LIMITS.maxSourceLoci)
      || !count(page.offset, page.total) || page.offset % size !== 0
      || rows.length !== Math.min(size, page.total - page.offset)
      || page.hasMore !== (page.offset + rows.length < page.total)) {
    throw new Error("Genome page does not match its requested range");
  }
}

/** Refuse oversized, malformed, stale or differently sourced pages before rendering. */
export function validateGenomeInspection(detail, request) {
  if (!detail || detail.schema !== "scriptbots.genome-inspection.v1") throw new Error("Invalid genome inspection schema");
  validateInspectionSource(detail, request);
  for (const [key, limit] of Object.entries(GENOME_LIMITS)) {
    if (detail.limits?.[key] !== limit) throw new Error("Genome page limits differ from the bounded protocol");
  }
  if (detail.status === "unavailable") {
    if (!UNAVAILABLE.has(detail.reason) || detail.loci !== undefined || detail.parentDiff !== undefined) {
      throw new Error("Invalid genome unavailability result");
    }
    return detail;
  }
  if (detail.status !== "ready" || typeof detail.familyId !== "string"
      || detail.familyId.length > 64 || !/^[a-z0-9]+(?:[-_][a-z0-9]+)*$/.test(detail.familyId) || !count(detail.schemaVersion)
      || !count(detail.generation) || !DIGEST.test(detail.genomeDigest)
      || !Array.isArray(detail.loci) || detail.loci.length > GENOME_LIMITS.locusPageSize) {
    throw new Error("Invalid genome material or locus page");
  }
  if (request.genomeDigest && detail.genomeDigest !== request.genomeDigest) {
    throw new Error("Genome material changed between pages; refresh from the first page");
  }
  validatePaging(detail.paging, request.locusOffset, GENOME_LIMITS.locusPageSize, detail.loci);
  if (detail.paging.total === 0) throw new Error("Ready genome has no canonical loci");
  detail.loci.forEach((row, offset) => {
    if (row?.index !== detail.paging.offset + offset || typeof row.label !== "string"
        || new TextEncoder().encode(row.label).length > 256) throw new Error("Invalid genome locus row");
    validateLocus(row.locus);
    validateValue(row.value);
  });
  const provenance = detail.provenance;
  if (!provenance || !DERIVATIONS.has(provenance.derivation)
      || !Array.isArray(provenance.parentUids) || provenance.parentUids.length !== 2
      || !Array.isArray(provenance.parentGenomeDigests) || provenance.parentGenomeDigests.length !== 2) {
    throw new Error("Invalid genome provenance");
  }
  exactIdentity(provenance.createdAt);
  if (BigInt(provenance.createdAt) > BigInt(detail.source.tick)) throw new Error("Genome creation follows its observation");
  let parents = 0;
  for (let index = 0; index < 2; index += 1) {
    const uid = provenance.parentUids[index];
    const hash = provenance.parentGenomeDigests[index];
    if (uid === null) { if (hash !== null) throw new Error("Parent genome has no parent identity"); }
    else { exactIdentity(uid); if (!DIGEST.test(hash)) throw new Error("Invalid parent genome digest"); parents += 1; }
  }
  if (parents !== DERIVATIONS.get(provenance.derivation)
      || (parents > 0 && provenance.parentUids[0] === null)
      || (parents === 2 && provenance.parentUids[0] === provenance.parentUids[1])) {
    throw new Error("Genome derivation disagrees with its parents");
  }
  const diff = detail.parentDiff;
  if (diff?.status === "founder") {
    if (parents !== 0 || diff.deltas !== undefined) throw new Error("Founder comparison contradicts provenance");
  } else if (diff?.status === "unavailable") {
    if (parents === 0 || !PARENT_UNAVAILABLE.has(diff.reason) || diff.deltas !== undefined) {
      throw new Error("Invalid parent comparison refusal");
    }
  } else if (diff?.status === "ready") {
    if (!parents || diff.parentUid !== provenance.parentUids[0]
        || diff.parentGenomeDigest !== provenance.parentGenomeDigests[0]
        || diff.primaryOnly !== (parents === 2) || !Array.isArray(diff.deltas)
        || diff.deltas.length > GENOME_LIMITS.deltaPageSize) throw new Error("Parent comparison changed its subject");
    validatePaging(diff.paging, request.deltaOffset, GENOME_LIMITS.deltaPageSize, diff.deltas);
    const summary = diff.summary;
    if (!summary || summary.total_loci !== detail.paging.total || summary.changed_loci !== diff.paging.total
        || !count(summary.changed_loci, summary.total_loci) || !Number.isFinite(summary.l1) || summary.l1 < 0
        || !Number.isFinite(summary.linf) || summary.linf < 0 || summary.linf > summary.l1
        || !summary.by_kind || typeof summary.by_kind !== "object" || Array.isArray(summary.by_kind)
        || Object.entries(summary.by_kind).some(([kind, n]) => !["Scalar", "Retarget", "KindFlip"].includes(kind)
          || !count(n, summary.changed_loci))
        || Object.values(summary.by_kind).reduce((sum, n) => sum + n, 0) !== summary.changed_loci) {
      throw new Error("Invalid complete parent difference summary");
    }
    for (const delta of diff.deltas) {
      const [kind, change] = tagged(delta);
      if (!change || !["Scalar", "Retarget", "KindFlip"].includes(kind)) throw new Error("Invalid parent difference row");
      validateLocus(change.locus);
      for (const value of [change.before, change.after]) {
        if (kind === "Scalar" ? !Number.isFinite(value) : !count(value, kind === "KindFlip" ? 255 : 0xffff_ffff)) {
          throw new Error("Invalid parent difference value");
        }
      }
    }
  } else throw new Error("Unknown parent comparison status");
  return detail;
}

/** Genome material is immutable: reads occur on selection, paging and explicit refresh. */
export class GenomeInspectionController extends InspectionController {
  constructor({ read, onChange, onFrame, now }) {
    const page = { locusOffset: 0, deltaOffset: 0, genomeDigest: null };
    super({ read, onChange, onFrame, now, enabled: false, autoRefresh: false,
      requestContext: () => ({ ...page }), validate: (detail, request) => {
        const valid = validateGenomeInspection(detail, request);
        if (valid.status === "ready") page.genomeDigest = valid.genomeDigest;
        return valid;
      } });
    this.page = page;
  }

  clear() {
    Object.assign(this.page, { locusOffset: 0, deltaOffset: 0, genomeDigest: null });
    super.clear();
  }

  select(uid) {
    Object.assign(this.page, { locusOffset: 0, deltaOffset: 0, genomeDigest: null });
    super.select(uid);
  }

  restart() {
    Object.assign(this.page, { locusOffset: 0, deltaOffset: 0, genomeDigest: null });
    this.cancel();
    if (this.selectedUid) this.emit({ status: "loading", uid: this.selectedUid, detail: null });
    return this.refresh(true);
  }

  navigate(kind, direction) {
    if (!this.enabled || this.pending || this.state.status !== "ready" || this.state.detail.status !== "ready"
        || !["locus", "delta"].includes(kind) || ![-1, 1].includes(direction)) return;
    const detail = this.state.detail;
    const paging = kind === "locus" ? detail.paging : detail.parentDiff.status === "ready" ? detail.parentDiff.paging : null;
    if (!paging) return;
    const offset = paging.offset + direction * paging.limit;
    if (offset < 0 || offset >= paging.total) return;
    this.page[kind === "locus" ? "locusOffset" : "deltaOffset"] = offset;
    this.emit({ status: "loading", uid: this.selectedUid, detail: null });
    void this.refresh(true);
  }
}

function number(value) { return Object.is(value, -0) ? "−0" : String(value); }
function locusName(locus) {
  const [kind, index] = tagged(locus);
  if (typeof index === "number") return `${kind} ${index}`;
  return `${kind} · node ${index.node} · slot ${index.conn}`;
}
function pageText(page, rows) {
  return page.total === 0 ? "No rows" : `${page.offset + 1}–${page.offset + rows.length} of ${page.total}`;
}
function row(document, values) {
  const element = document.createElement("tr");
  for (const value of values) { const cell = document.createElement("td"); cell.textContent = value; element.append(cell); }
  return element;
}

export function renderGenomeSource(panel, state, frame) {
  if (panel && state.status === "ready") panel.querySelector("#genome-source").textContent = inspectionStatus(state.detail, frame);
}

/** Render exact scientific values with text nodes, including operator/target loci. */
export function renderGenomeInspector(panel, state, frame) {
  if (!panel) return;
  panel.dataset.state = state.status;
  const query = (id) => panel.querySelector(`#genome-${id}`);
  query("refresh").disabled = !state.uid || state.status === "loading" || state.status === "inactive";
  query("content").hidden = true;
  query("source").textContent = "";
  query("loci").replaceChildren();
  query("deltas").replaceChildren();
  for (const id of ["locus-prev", "locus-next", "delta-prev", "delta-next"]) query(id).disabled = true;
  const messages = { inactive: "Open this panel to browse the selected creature’s inherited genome.",
    empty: "Select a creature to inspect its inherited genome.", loading: `Reading genome page for creature ${state.uid}…`,
    missing: "The selected creature is no longer present in the displayed world.", error: state.message };
  if (state.status !== "ready") { query("status").textContent = messages[state.status] || "Genome unavailable."; return; }
  const detail = state.detail;
  renderGenomeSource(panel, state, frame);
  if (detail.status !== "ready") {
    query("status").textContent = `Genome unavailable: ${detail.reason.replaceAll("_", " ")}.`;
    return;
  }
  query("status").textContent = `${detail.familyId} · schema ${detail.schemaVersion} · generation ${detail.generation}`;
  query("content").hidden = false;
  query("digest").textContent = `Genome ${detail.genomeDigest}`;
  const provenance = detail.provenance;
  query("provenance").textContent = `${provenance.derivation} at tick ${provenance.createdAt}. Brain parents: ${provenance.parentUids.filter((uid) => uid !== null).join(", ") || "none (founder)"}.`;
  query("locus-page").textContent = pageText(detail.paging, detail.loci);
  for (const locus of detail.loci) {
    const [kind, value] = tagged(locus.value);
    query("loci").append(row(panel.ownerDocument, [String(locus.index), locus.label, kind, number(value)]));
  }
  query("locus-prev").disabled = detail.paging.offset === 0;
  query("locus-next").disabled = !detail.paging.hasMore;
  const diff = detail.parentDiff;
  query("delta-page").textContent = "";
  if (diff.status === "founder") query("parent").textContent = "Founder genome: no inherited parent material to compare.";
  else if (diff.status === "unavailable") query("parent").textContent = `Parent comparison unavailable: ${diff.reason.replaceAll("_", " ")}.`;
  else {
    query("parent").textContent = `${diff.primaryOnly ? "Primary brain parent" : "Brain parent"} ${diff.parentUid}: ${diff.summary.changed_loci} of ${diff.summary.total_loci} loci changed; scalar L1 ${number(diff.summary.l1)}, maximum ${number(diff.summary.linf)}.${diff.primaryOnly ? " This compares one parent; it does not attribute changes solely to mutation." : ""}`;
    query("delta-page").textContent = pageText(diff.paging, diff.deltas);
    for (const delta of diff.deltas) {
      const [kind, value] = tagged(delta);
      query("deltas").append(row(panel.ownerDocument, [locusName(value.locus), kind, number(value.before), number(value.after)]));
    }
    query("delta-prev").disabled = diff.paging.offset === 0;
    query("delta-next").disabled = !diff.paging.hasMore;
  }
}
