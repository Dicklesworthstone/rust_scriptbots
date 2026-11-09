import assert from "node:assert/strict";
import test from "node:test";
import { GENOME_LIMITS, GenomeInspectionController, validateGenomeInspection } from "../genome.js";
import { RemoteStreamClient } from "../remote.js";

const UID = "18446744073709551615";
const HASH = "1".repeat(64);
const PARENT_HASH = "2".repeat(64);
const flush = () => new Promise((resolve) => setImmediate(resolve));
function frame(revision = 1, epoch = "0", uid = UID) {
  return { identity: { sessionId: "9007199254740993", publicationRevision: String(revision),
    scientificRevision: String(revision), controlRevision: "0", configRevision: "0", tick: String(revision) },
  snapshot: { epoch, agents: [{ uid }] } };
}
function request(locusOffset = 0, deltaOffset = 0) {
  const selected = frame();
  return { uid: UID, sessionId: selected.identity.sessionId, epoch: "0", identity: selected.identity,
    locusOffset, deltaOffset, genomeDigest: null };
}
function detail(req, { total = 130, changed = 35 } = {}) {
  const revisions = { scientific: req.identity.scientificRevision, control: req.identity.controlRevision,
    config: req.identity.configRevision };
  const loci = Array.from({ length: Math.min(64, total - req.locusOffset) }, (_, i) => {
    const index = req.locusOffset + i;
    return { index, label: `cell ${index}`, locus: { Cell: index }, value: { Scalar: index / 2 } };
  });
  const deltas = Array.from({ length: Math.min(32, changed - req.deltaOffset) }, (_, i) =>
    ({ Scalar: { locus: { Cell: req.deltaOffset + i }, before: 0, after: 0.5 } }));
  return { schema: "scriptbots.genome-inspection.v1", uid: UID, limits: { ...GENOME_LIMITS }, status: "ready",
    source: { sessionId: req.sessionId, epoch: req.epoch, tick: req.identity.tick,
      publicationRevision: req.identity.publicationRevision, published: revisions, inspected: { ...revisions } },
    familyId: "assembly", schemaVersion: 1, genomeDigest: HASH, generation: 4,
    provenance: { createdAt: "0", parentUids: ["9007199254740994", null],
      parentGenomeDigests: [PARENT_HASH, null], derivation: "MutationOnly" },
    paging: { offset: req.locusOffset, limit: 64, total, hasMore: req.locusOffset + loci.length < total }, loci,
    parentDiff: { status: "ready", parentUid: "9007199254740994", parentGenomeDigest: PARENT_HASH,
      primaryOnly: false, summary: { changed_loci: changed, total_loci: total, l1: changed / 2,
        linf: changed ? 0.5 : 0, by_kind: { Scalar: changed } },
      paging: { offset: req.deltaOffset, limit: 32, total: changed, hasMore: req.deltaOffset + deltas.length < changed }, deltas } };
}

test("complete canonical loci and differences retain exact source, parents and independent page ranges", () => {
  const req = request(64, 32);
  const page = detail(req);
  assert.equal(validateGenomeInspection(page, req), page);
  assert.equal(page.uid, UID);
  assert.equal(page.loci[0].index, 64);
  assert.equal(page.parentDiff.deltas.length, 3);
  assert.equal(page.parentDiff.summary.changed_loci, 35);
  page.loci[0].locus = { NodeKind: { node: 50, conn: 2 } };
  page.loci[0].value = { Kind: 2 };
  assert.equal(validateGenomeInspection(page, req).loci[0].value.Kind, 2, "operator locus is not an invented edge weight");
});

test("genome validation rejects malformed bounds, pages, lineage, values, summaries and material changes", () => {
  const req = request();
  for (const mutate of [
    (d) => { d.uid = "9007199254740994"; },
    (d) => { d.source.sessionId = "2"; },
    (d) => { d.source.epoch = "1"; },
    (d) => { d.source.tick = "0"; },
    (d) => { d.limits.maxSourceLoci += 1; },
    (d) => { d.familyId = "assembly.experimental"; },
    (d) => { d.familyId = "Uppercase"; },
    (d) => { d.paging.offset = 64; },
    (d) => { d.paging.total = 65537; },
    (d) => { d.paging.hasMore = false; },
    (d) => { d.loci.push(d.loci[0]); },
    (d) => { d.loci[0].index = 1; },
    (d) => { d.loci[0].value = { Scalar: Infinity }; },
    (d) => { d.loci[0].locus = { NodeWeight: { node: 1, conn: 256 } }; },
    (d) => { d.loci[0].label = "x".repeat(257); },
    (d) => { d.provenance.createdAt = "2"; },
    (d) => { d.provenance.parentGenomeDigests[0] = null; },
    (d) => { d.provenance.derivation = "Founder"; },
    (d) => { d.parentDiff.parentGenomeDigest = HASH; },
    (d) => { d.parentDiff.primaryOnly = true; },
    (d) => { d.parentDiff.summary.total_loci = 129; },
    (d) => { d.parentDiff.summary.by_kind.Scalar = 34; },
    (d) => { d.parentDiff.summary.l1 = NaN; },
    (d) => { d.parentDiff.deltas[0].Scalar.after = Infinity; },
  ]) {
    const value = detail(req); mutate(value);
    assert.throws(() => validateGenomeInspection(value, req));
  }
  assert.throws(() => validateGenomeInspection(detail(req), { ...req, genomeDigest: PARENT_HASH }), /material changed/);
});

test("founders, absent parent material and typed producer refusals are distinct scientific outcomes", () => {
  const req = request();
  const founder = detail(req);
  founder.provenance = { createdAt: "0", parentUids: [null, null], parentGenomeDigests: [null, null], derivation: "Founder" };
  founder.parentDiff = { status: "founder" };
  assert.equal(validateGenomeInspection(founder, req).parentDiff.status, "founder");
  const unavailable = detail(req);
  unavailable.parentDiff = { status: "unavailable", reason: "parent_unavailable" };
  assert.equal(validateGenomeInspection(unavailable, req).loci.length, 64);
  const refused = { schema: founder.schema, uid: UID, source: founder.source, limits: founder.limits,
    status: "unavailable", reason: "source_budget_exceeded" };
  assert.equal(validateGenomeInspection(refused, req).reason, "source_budget_exceeded");
  refused.loci = [];
  assert.throws(() => validateGenomeInspection(refused, req), /unavailability/);
});

function deferred() {
  const reads = [];
  const controller = new GenomeInspectionController({ read: (uid, req, signal) =>
    new Promise((resolve) => reads.push({ uid, req, signal, resolve })) });
  controller.observe(frame(), 1);
  controller.select(UID);
  return { controller, reads };
}

test("genome reads are opt-in and publication churn cannot repeatedly decode inherited material", async () => {
  const { controller, reads } = deferred();
  assert.equal(reads.length, 0);
  controller.setEnabled(true);
  assert.equal(reads.length, 1);
  reads[0].resolve(detail(reads[0].req));
  await flush();
  for (let revision = 2; revision < 100; revision += 1) controller.observe(frame(revision), 1);
  assert.equal(reads.length, 1, "ninety-eight presented updates create no genome work");
  assert.equal(controller.state.detail.genomeDigest, HASH);
  controller.navigate("locus", 1);
  controller.navigate("locus", 1);
  assert.equal(reads.length, 2, "navigation has at most one current request");
  assert.equal(reads[1].req.locusOffset, 64);
  assert.equal(reads[1].req.genomeDigest, HASH);
  assert.equal(reads[1].req.identity.tick, "99");
  reads[1].resolve(detail(reads[1].req));
  await flush();
  controller.navigate("delta", 1);
  assert.equal(reads[2].req.locusOffset, 64);
  assert.equal(reads[2].req.deltaOffset, 32);
  reads[2].resolve(detail(reads[2].req));
  await flush();
  controller.navigate("delta", 1);
  assert.equal(reads.length, 3, "last difference page is a hard navigation boundary");
});

test("selection, closed panels and new owner incarnations cancel late genome pages", async () => {
  const { controller, reads } = deferred();
  controller.setEnabled(true);
  controller.setEnabled(false);
  assert.equal(reads[0].signal.aborted, true);
  reads[0].resolve(detail(reads[0].req));
  await flush();
  assert.equal(controller.state.status, "inactive");
  controller.setEnabled(true);
  controller.observe(frame(1, "0"), 2);
  assert.equal(reads[1].signal.aborted, true);
  reads[1].resolve(detail(reads[1].req));
  await flush();
  assert.equal(controller.selectedUid, null);
  assert.equal(controller.page.genomeDigest, null);
  controller.select(UID);
  reads[2].resolve(detail(reads[2].req));
  await flush();
  controller.navigate("locus", 1);
  controller.observe(frame(1, "1"), 2);
  reads[3].resolve(detail(reads[3].req));
  await flush();
  assert.equal(controller.state.status, "empty");
  assert.equal(controller.page.locusOffset, 0);
});

test("a differently materialized later page is refused until an explicit first-page refresh", async () => {
  const { controller, reads } = deferred();
  controller.setEnabled(true);
  reads[0].resolve(detail(reads[0].req)); await flush();
  controller.navigate("locus", 1);
  const changed = detail(reads[1].req); changed.genomeDigest = PARENT_HASH;
  reads[1].resolve(changed); await flush();
  assert.equal(controller.state.status, "error");
  assert.match(controller.state.message, /material changed/);
  void controller.restart();
  assert.equal(reads[2].req.locusOffset, 0);
  assert.equal(reads[2].req.genomeDigest, null);
  const refreshed = detail(reads[2].req); refreshed.genomeDigest = PARENT_HASH;
  reads[2].resolve(refreshed); await flush();
  assert.equal(controller.state.status, "ready");
});

test("remote genome pages use the negotiated host, exact subject and bounded integer offsets", async () => {
  const client = new RemoteStreamClient({ url: "wss://lab.example/prefix/api/ws/stream", decodeFrame() {} });
  client.state = "connected"; client.tracker.sessionId = "9007199254740993";
  const original = globalThis.fetch;
  const calls = [];
  globalThis.fetch = async (url, options) => {
    calls.push({ url: new URL(url), options });
    return new Response(JSON.stringify(detail(request(64, 32))), { headers: { "content-type": "application/json" } });
  };
  try {
    await client.inspectGenome(UID, { sessionId: client.tracker.sessionId, epoch: "0", locusOffset: 64, deltaOffset: 32 });
    assert.equal(calls[0].url.pathname, "/prefix/api/agents/genome");
    assert.equal(calls[0].url.searchParams.get("uid"), UID);
    assert.equal(calls[0].url.searchParams.get("locus_offset"), "64");
    assert.equal(calls[0].url.searchParams.get("delta_offset"), "32");
    assert.equal(calls[0].options.credentials, "omit");
    await assert.rejects(client.inspectGenome(UID, { sessionId: client.tracker.sessionId, epoch: "0", locusOffset: 1 }));
    await assert.rejects(client.inspectGenome(UID, { sessionId: "2", epoch: "0" }));
    assert.equal(calls.length, 1, "invalid pages and hosts never reach the network");
  } finally { globalThis.fetch = original; }
});

test("remote genome byte ceiling applies to actual streaming bytes, including dishonest lengths", async () => {
  const client = new RemoteStreamClient({ url: "ws://lab.example/api/ws/stream", decodeFrame() {} });
  client.state = "connected"; client.tracker.sessionId = "1";
  const original = globalThis.fetch;
  globalThis.fetch = async () => new Response(" ".repeat(65537), { headers: { "content-length": "1" } });
  try {
    await assert.rejects(client.inspectGenome(UID, { sessionId: "1", epoch: "0" }), /64 KiB bound/);
  } finally { globalThis.fetch = original; }
});
