import assert from "node:assert/strict";
import test from "node:test";
import { FIT_WORLD_VIEW, MAX_WORLD_ZOOM, WorldViewport, WorldDrag,
  worldProjection, visitWorldCopies, visitWorldTiles } from "../world-view.js";
import { agentRadius, pickAgent } from "../inspector.js";
import { CanvasRenderer } from "../renderers/canvas.js";
import { WebGpuRenderer } from "../renderers/webgpu.js";

const UID = "18446744073709551615";
const flush = () => new Promise((resolve) => setImmediate(resolve));
const close = (actual, expected, tolerance = 1e-8) => assert.ok(Math.abs(actual - expected) < tolerance,
  `${actual} differs from ${expected}`);
const agent = (uid = UID, position = [50, 50]) => ({ uid, position, health: 1,
  color: [0.9, 0.2, 0.1], boost: false });
const frame = (agents = [agent()], revision = 1) => ({
  identity: { sessionId: "9007199254740993", publicationRevision: String(revision), tick: String(revision) },
  snapshot: { tick: revision, epoch: "0", world: { width: 100, height: 100 }, agents },
});

function copies(position, projection, padding = 0) {
  const result = [];
  visitWorldCopies(position, projection, padding, (x, y) => result.push([x, y]));
  return result;
}

test("zoom keeps its painted pointer anchor fixed without changing source state or a retained paint", () => {
  const source = frame();
  const before = JSON.stringify(source);
  const camera = new WorldViewport();
  camera.observe(source, 1);
  const painted = camera.view;
  camera.zoomBy(2, 0.75, 0.25, painted);
  assert.deepEqual(camera.view, { centerX: 0.625, centerY: 0.375, zoom: 2 });
  assert.deepEqual(copies([75, 25], worldProjection(source.snapshot.world, 400, 200, camera.view)), [[300, 50]]);
  // Further queued input targets the same still-visible point, not the desired view.
  camera.zoomBy(2, 0.75, 0.25, painted);
  assert.deepEqual(copies([75, 25], worldProjection(source.snapshot.world, 400, 200, camera.view)), [[300, 50]]);
  assert.equal(painted, FIT_WORLD_VIEW);
  assert.ok(Object.isFrozen(camera.view));
  const observed = camera.view;
  camera.observe(frame(source.snapshot.agents, 2), 1);
  assert.equal(camera.view, observed, "ordinary publications preserve the camera");
  assert.equal(JSON.stringify(source), before);
});

test("toroidal projection agrees with independent world-image enumeration at seams and zoom bounds", () => {
  const world = { width: 100, height: 80 };
  for (const view of [FIT_WORLD_VIEW, { centerX: 0, centerY: 0, zoom: 1 },
    { centerX: 0.99, centerY: 0.02, zoom: 4 }, { centerX: 0.25, centerY: 0.75, zoom: MAX_WORLD_ZOOM }]) {
    const projection = worldProjection(world, 300, 170, view);
    for (const position of [[0, 0], [99, 79], [1, 1], [25, 60], [50, 40]]) {
      const actual = copies(position, projection, 8);
      const expected = [];
      for (let row = -2; row <= 2; row += 1) for (let column = -2; column <= 2; column += 1) {
        const x = ((position[0] + column * world.width - view.centerX * world.width)
          / (world.width / view.zoom) + 0.5) * 300;
        const y = ((position[1] + row * world.height - view.centerY * world.height)
          / (world.height / view.zoom) + 0.5) * 170;
        if (x >= -8 && x <= 308 && y >= -8 && y <= 178) expected.push([x, y]);
      }
      const order = (a, b) => a[1] - b[1] || a[0] - b[0];
      actual.sort(order); expected.sort(order);
      assert.equal(actual.length, expected.length);
      actual.forEach((point, index) => point.forEach((value, axis) => close(value, expected[index][axis])));
      assert.ok(actual.length <= 4, "ordinary point/radius input has at most four visible seam images");
    }
  }
});

test("bounded whole-world tiles cover every visible sample exactly once across both seams", () => {
  for (const view of [FIT_WORLD_VIEW, { centerX: 0, centerY: 0, zoom: 1 },
    { centerX: 0.99, centerY: 0.02, zoom: 3 }, { centerX: 0.1, centerY: 0.8, zoom: MAX_WORLD_ZOOM }]) {
    const tiles = [];
    visitWorldTiles(worldProjection({ width: 100, height: 100 }, 200, 120, view),
      (x, y, width, height) => tiles.push({ x, y, width, height }));
    assert.ok(tiles.length >= 1 && tiles.length <= 4);
    for (const x of [0.5, 17.5, 99.5, 199.5]) for (const y of [0.5, 59.5, 119.5]) {
      const visible = tiles.filter((tile) => x >= tile.x && x < tile.x + tile.width
        && y >= tile.y && y < tile.y + tile.height);
      assert.equal(visible.length, 1);
      const tile = visible[0];
      const worldX = (x / 200 - 0.5) / view.zoom + view.centerX;
      const worldY = (y / 120 - 0.5) / view.zoom + view.centerY;
      close((x - tile.x) / tile.width, worldX - Math.floor(worldX));
      close((y - tile.y) / tile.height, worldY - Math.floor(worldY));
    }
  }
});

test("follow retains the exact UID across seams, ends on death and cannot transfer to another owner", () => {
  const camera = new WorldViewport();
  const source = frame([agent(UID, [99, 20])]);
  const before = JSON.stringify(source);
  camera.observe(source, 3);
  camera.zoomBy(4);
  camera.follow(UID, source.snapshot);
  camera.observe(frame([agent(UID, [1, 21])], 2), 3);
  close(camera.view.centerX, 0.01);
  close(camera.view.centerY, 0.21);
  assert.equal(camera.view.zoom, 4);
  camera.zoomBy(2, 0, 1);
  close(camera.view.centerX, 0.01);
  close(camera.view.centerY, 0.21);
  const lastFollow = camera.view;
  camera.observe(frame([agent("9", [1, 21])], 3), 3);
  assert.equal(camera.followedUid, null, "another stable UID cannot inherit follow");
  assert.equal(camera.view, lastFollow, "death leaves the last observed place inspectable");
  assert.match(camera.notice, /no longer present/);
  camera.follow("9", frame([agent("9", [1, 21])]).snapshot);
  camera.observe(frame([agent("9", [1, 21])], 4), 4);
  assert.equal(camera.followedUid, null);
  assert.deepEqual(camera.view, FIT_WORLD_VIEW);
  assert.equal(JSON.stringify(source), before);
});

test("owner epoch and dimension changes reset view while manual pan releases follow", () => {
  for (const replacement of [
    (next) => { next.identity.sessionId = "9007199254740994"; },
    (next) => { next.snapshot.epoch = "1"; },
    (next) => { next.snapshot.world.width = 200; },
  ]) {
    const camera = new WorldViewport(), source = frame();
    camera.observe(source, 1); camera.zoomBy(8); camera.follow(UID, source.snapshot);
    const next = structuredClone(source); replacement(next);
    camera.observe(next, 1);
    assert.deepEqual(camera.view, FIT_WORLD_VIEW);
    assert.equal(camera.followedUid, null);
  }
  const camera = new WorldViewport(), source = frame();
  camera.observe(source, 1); camera.zoomBy(4); camera.follow(UID, source.snapshot);
  const painted = camera.view;
  camera.panFrom(painted, 0.6, -0.4);
  close(camera.view.centerX, 0.35); close(camera.view.centerY, 0.6);
  assert.equal(camera.followedUid, null);
  camera.fit();
  assert.deepEqual(camera.view, FIT_WORLD_VIEW);
});

test("ordinary repeated pans cross a floating-point seam without leaving the toroidal range", () => {
  const camera = new WorldViewport();
  const source = frame([agent(UID, [300, 50])]);
  source.snapshot.world.width = 1000;
  camera.observe(source, 1);
  camera.follow(UID, source.snapshot);
  for (let step = 0; step < 3; step += 1) camera.panFrom(camera.view, 0.1, 0);
  assert.equal(camera.view.centerX, 0);
  assert.equal(Object.is(camera.view.centerX, -0), false);
  camera.panFrom(camera.view, 0.1, 0);
  close(camera.view.centerX, 0.9);
  assert.equal(camera.followedUid, null);
});

test("invalid camera inputs refuse without corrupting the finite bounded view", () => {
  const camera = new WorldViewport();
  camera.observe(frame(), 1);
  for (const factor of [0, -1, NaN, Infinity]) assert.equal(camera.zoomBy(factor), false);
  assert.equal(camera.panFrom(camera.view, NaN, 0), false);
  for (const zoom of [0, MAX_WORLD_ZOOM + 1, Infinity]) {
    assert.throws(() => worldProjection({ width: 100, height: 100 }, 100, 100,
      { centerX: 0.5, centerY: 0.5, zoom }), /bounds/);
  }
  assert.throws(() => worldProjection({ width: 0, height: 100 }, 100, 100), /dimensions/);
  assert.throws(() => camera.follow("18446744073709551616", frame().snapshot), /exact/);
  assert.throws(() => camera.follow("9", frame().snapshot), /present/);
  assert.deepEqual(camera.view, FIT_WORLD_VIEW);
  camera.zoomBy(1e100);
  assert.equal(camera.view.zoom, MAX_WORLD_ZOOM);
  camera.zoomBy(1e-100);
  assert.equal(camera.view.zoom, 1);
});

test("a primary drag distinguishes click jitter, suppresses its click and ignores competing pointers", () => {
  const drag = new WorldDrag();
  assert.equal(drag.begin(1, 10, 20, FIT_WORLD_VIEW), true);
  assert.equal(drag.begin(2, 10, 20, FIT_WORLD_VIEW), false);
  assert.equal(drag.move(2, 40, 50), null);
  assert.equal(drag.move(1, 12, 22), null);
  assert.equal(drag.end(2), false);
  assert.equal(drag.end(1), true);
  assert.equal(drag.consumeClick(), false);
  drag.begin(3, 10, 20, FIT_WORLD_VIEW);
  assert.deepEqual(drag.move(3, 30, 50), { view: FIT_WORLD_VIEW, deltaX: 20, deltaY: 30 });
  drag.end(3);
  assert.equal(drag.consumeClick(), true);
  assert.equal(drag.consumeClick(), false);
  drag.begin(4, 10, 20, FIT_WORLD_VIEW);
  drag.cancel(true);
  assert.equal(drag.move(4, 50, 50), null);
  assert.equal(drag.end(4), false);
  assert.equal(drag.consumeClick(), true, "owner replacement cannot turn an old gesture into new-owner selection");
  drag.begin(5, 10, 20, FIT_WORLD_VIEW);
  drag.end(5);
  assert.equal(drag.consumeClick(), false, "the next independent click is preserved");
  drag.begin(6, 10, 20, FIT_WORLD_VIEW, 200, 100);
  assert.equal(drag.move(6, 60, 30, 400, 200), null);
  assert.equal(drag.active, null, "resize cancels the gesture instead of changing its coordinate system");
  assert.equal(drag.consumeClick(), true);
});

test("picking uses retained paint coordinates even when a desired zoom has not been accepted", () => {
  const source = frame([agent(UID, [99, 50]), agent("9", [1, 50])]);
  const camera = new WorldViewport();
  camera.observe(source, 1);
  const painted = camera.view;
  camera.zoomBy(4);
  const options = { width: 400, height: 200, canvasWidth: 200, canvasHeight: 100, view: painted };
  assert.equal(pickAgent(source.snapshot, 396, 100, options), UID);
  assert.equal(pickAgent(source.snapshot, 396, 100, { ...options, view: camera.view }), null);
  const seam = { centerX: 0, centerY: 0.5, zoom: 4 };
  assert.equal(pickAgent(source.snapshot, 184, 100, { ...options, view: seam }), UID);
  assert.equal(pickAgent(source.snapshot, 216, 100, { ...options, view: seam }), "9");
});

function rendererPair(t, width = 200, height = 100) {
  for (const [key, value] of Object.entries({
    GPUBufferUsage: { UNIFORM: 1, COPY_DST: 2, STORAGE: 4 },
    GPUTextureUsage: { TEXTURE_BINDING: 1, COPY_DST: 2 },
    ImageData: class { constructor(data, width, height) { Object.assign(this, { data, width, height }); } },
    OffscreenCanvas: class { getContext() { return { putImageData() {} }; } },
  })) {
    const previous = globalThis[key]; globalThis[key] = value;
    t.after(() => { if (previous === undefined) delete globalThis[key]; else globalThis[key] = previous; });
  }
  const painted = [], tiles = [], draws = [], writes = [], uploads = [];
  let arc;
  const ctx = { fillRect() {}, beginPath() {}, arc(x, y, radius) { arc = { x, y, radius }; },
    fill() { painted.push({ ...arc }); }, stroke() {}, drawImage(...args) { tiles.push(args.slice(1)); } };
  const canvas = new CanvasRenderer({ width, height, getContext: () => ctx });
  const buffer = (options) => ({ ...options, destroy() {} });
  const pass = { setPipeline() {}, setBindGroup() {}, draw(...args) { draws.push(args); }, end() {} };
  const device = { lost: new Promise(() => {}), addEventListener() {},
    limits: { maxBufferSize: 1_048_576, maxStorageBufferBindingSize: 1_048_576 },
    createBuffer: buffer, createBindGroup: () => ({}),
    createTexture: () => ({ createView: () => ({}), destroy() {} }),
    createCommandEncoder: () => ({ beginRenderPass: () => pass, finish: () => ({}) }),
    queue: { writeBuffer(target, offset, data, start = 0, size) {
      const bytes = ArrayBuffer.isView(data)
        ? new Uint8Array(data.buffer, data.byteOffset + start, size ?? data.byteLength - start)
        : new Uint8Array(data, start, size ?? data.byteLength - start);
      writes.push({ target, values: Array.from(new Float32Array(bytes.slice().buffer)) });
    }, writeTexture(...args) { uploads.push(args); }, submit() {}, onSubmittedWorkDone: () => Promise.resolve() },
    destroy() {} };
  const context = { getCurrentTexture: () => ({ createView: () => ({}) }), unconfigure() {} };
  const pipeline = { getBindGroupLayout: () => ({}) };
  const gpu = new WebGpuRenderer({ width, height }, device, context, pipeline, buffer({}),
    "rgba8unorm", assert.fail, pipeline);
  t.after(() => gpu.destroy());
  return { canvas, gpu, painted, tiles, draws, writes, uploads };
}

test("Canvas and WebGPU consume the same seam-aware projection and cull offscreen creatures", async (t) => {
  const renderers = rendererPair(t);
  const snapshot = frame([agent(UID, [99, 50]), agent("9", [1, 50]), agent("8", [50, 50])]).snapshot;
  const before = JSON.stringify(snapshot);
  const presentation = { view: { centerX: 0, centerY: 0.5, zoom: 4 }, selectedUid: UID };
  assert.equal(renderers.canvas.render(snapshot, presentation), true);
  assert.equal(renderers.gpu.render(snapshot, presentation), true);
  assert.equal(renderers.painted.length, 2);
  assert.deepEqual(renderers.draws, [[6, 2, 0, 0]]);
  renderers.painted.forEach((circle, index) => {
    close(renderers.gpu.agentArray[index * 8], circle.x, 1e-5);
    close(renderers.gpu.agentArray[index * 8 + 1], circle.y, 1e-5);
    close(renderers.gpu.agentArray[index * 8 + 2], circle.radius, 1e-5);
  });
  close(renderers.painted[0].x, 92); close(renderers.painted[1].x, 108);
  close(renderers.painted[0].radius, agentRadius(snapshot.agents[0]) * 4);
  assert.equal(renderers.gpu.agentArray[3], 1);
  assert.equal(renderers.gpu.agentArray[11], 0);
  assert.equal(renderers.gpu.render(snapshot, { view: FIT_WORLD_VIEW }), false,
    "a busy GPU does not silently accept another camera or build a frame queue");
  assert.equal(renderers.draws.length, 1);
  await flush();
  assert.equal(JSON.stringify(snapshot), before);
});

test("seam copies can grow the retained GPU buffer without dropping earlier creature data", (t) => {
  const { gpu, draws } = rendererPair(t, 100, 100);
  const agents = Array.from({ length: 260 }, (_, index) => ({ ...agent(String(index + 1), [0, 0]),
    color: [index / 300, 0.2, 0.5] }));
  gpu.render(frame(agents).snapshot);
  assert.deepEqual(draws, [[6, 1040, 0, 0]]);
  assert.ok(gpu.agentCapacity >= 1040);
  for (const index of [0, 127, 128, 255, 259]) {
    for (let copy = 0; copy < 4; copy += 1) {
      const offset = (index * 4 + copy) * 8;
      close(gpu.agentArray[offset + 4], index / 300, 1e-6);
      close(gpu.agentArray[offset + 6], 0.5);
    }
  }
});

test("camera-only movement resamples the same environment texture and wraps Canvas tiles", async (t) => {
  const r = rendererPair(t);
  const snapshot = frame([]).snapshot;
  const environment = { width: 2, height: 2, sourceWidth: 2, sourceHeight: 2, cellSize: 50,
    terrainRevision: "1", foodRevision: "2", hydrologyRevision: "3", foodMax: 0.5,
    hydrologyEnabled: false, cells: Array.from({ length: 4 }, (_, index) => ({
      terrain: index % 2 ? "Rock" : "Grass", elevation: index / 4, food: index / 10, waterDepth: 0,
    })) };
  const presentation = { environment, identity: frame().identity, generation: 1, view: FIT_WORLD_VIEW };
  r.canvas.render(snapshot, presentation); r.gpu.render(snapshot, presentation);
  assert.equal(r.uploads.length, 1);
  assert.deepEqual(r.tiles, [[0, 0, 200, 100]]);
  await flush();
  const moved = { ...presentation, view: { centerX: 0.99, centerY: 0.02, zoom: 4 } };
  r.tiles.length = 0;
  r.canvas.render(snapshot, moved); r.gpu.render(snapshot, moved);
  assert.equal(r.uploads.length, 1, "zoom and pan cannot rebuild or replace scientific environment samples");
  assert.equal(r.tiles.length, 4);
  const uniform = r.writes.filter((write) => write.target === r.gpu.environmentUniform).at(-1).values;
  assert.deepEqual(uniform.slice(0, 4), [1, 1, 0, 0]);
  close(uniform[4], 0.99, 1e-6); close(uniform[5], 0.02, 1e-6); close(uniform[6], 0.25);
  assert.equal(r.gpu.environmentUniform.size, 32);
});
