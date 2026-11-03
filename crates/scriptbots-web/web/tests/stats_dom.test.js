// crates/scriptbots-web/web/tests/stats_dom.test.js
// DOM contract controls (bd-2z0.12.6) and actual packaged WASM loading (bd-azi3).

import { createHash } from "node:crypto";
import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const webDir = path.resolve(__dirname, "..");
const packageDir = path.resolve(process.env.SCRIPTBOTS_WASM_PACKAGE_DIR || path.join(webDir, "pkg"));

function readPackage() {
  const manifest = JSON.parse(fs.readFileSync(path.join(packageDir, "package.json"), "utf8"));
  const original = JSON.parse(fs.readFileSync(`${packageDir}.wasm-pack/package.json`, "utf8"));
  const moduleFile = original.module ?? original.main;
  const wasmFiles = original.files.filter((file) => file.endsWith(".wasm"));
  if (wasmFiles.length !== 1 || original.type !== "module") {
    throw new Error("Expected one WASM binary in the generated ES module package");
  }
  const wasmFile = wasmFiles[0];
  const expectedExports = {
    ".": { types: `./${original.types}`, import: `./${moduleFile}`, default: `./${moduleFile}` },
    "./package.json": "./package.json",
    [`./${wasmFile}`]: `./${wasmFile}`,
  };
  if (JSON.stringify(manifest.exports) !== JSON.stringify(expectedExports)) {
    throw new Error("Package exports do not address the generated module, types, and WASM");
  }
  for (const [key, value] of Object.entries(original)) {
    if (key !== "files" && JSON.stringify(manifest[key]) !== JSON.stringify(value)) {
      throw new Error(`Generated package metadata was changed: ${key}`);
    }
  }
  for (const file of original.files) {
    if (!manifest.files.includes(file)) throw new Error(`Generated package file was omitted: ${file}`);
  }
  const files = Object.fromEntries(manifest.files.map((file) => {
    if (!/^[A-Za-z0-9_][A-Za-z0-9_.-]*$/.test(file)) throw new Error(`Unsafe package file: ${file}`);
    const filePath = path.join(packageDir, file);
    if (!fs.lstatSync(filePath).isFile()) throw new Error(`Package entry is not a regular file: ${file}`);
    const bytes = fs.readFileSync(filePath);
    if (bytes.length === 0) throw new Error(`Empty package file: ${file}`);
    if (original.files.includes(file) && !bytes.equals(fs.readFileSync(path.join(`${packageDir}.wasm-pack`, file)))) {
      throw new Error(`Generated package bytes changed: ${file}`);
    }
    return [file, { bytes: bytes.length, sha256: createHash("sha256").update(bytes).digest("hex") }];
  }));
  if (!files[moduleFile] || !files[original.types] || !files[wasmFile]) {
    throw new Error("Generated module, declarations, or binary is absent from the package files");
  }
  return { manifest, moduleFile, typesFile: original.types, wasmFile, files };
}

const camelCaseWasmModule = `
  export default async function initWasm() { return {}; }
  export function version() { return "0.1.0-camel-case-prod"; }
  export function default_init_options() { return { world_width: 1280, world_height: 720 }; }
  export function init_sim(opts) {
    let tick = 0;
    let elapsed = 0;
    let speed = 1;
    let credit = 0;
    return {
      setSpeed(value) { speed = value; },
      drive(now) {
        credit += (now - elapsed) * speed * 60 / 1000;
        elapsed = now;
        const steps = Math.floor(credit);
        credit -= steps;
        this.tick(steps);
        return { scientific_steps: steps, automatic_steps_skipped: 0 };
      },
      snapshot() { return this.tick(0); },
      frame() {
        const snapshot = this.snapshot();
        return { schema: "scriptbots.browser-frame.v1", snapshot, identity: {
          sessionId: "1001", publicationRevision: String(snapshot.tick + 400),
          scientificRevision: String(snapshot.tick + 100), controlRevision: "20",
          configRevision: "10", tick: String(snapshot.tick)
        } };
      },
      tick(steps) {
        tick += steps;
        const pop = 200 + tick;
        const births = 10 + Math.floor(tick / 2);
        const deaths = Math.floor(tick / 4);
        const avgEnergy = 1.500 + (tick * 0.010);
        const avgHealth = 0.900 - (tick * 0.001);
        return {
          tick: tick,
          world: { width: 1280, height: 720, closed: true },
          summary: {
            agentCount: pop,
            births: births,
            deaths: deaths,
            totalEnergy: pop * avgEnergy,
            averageEnergy: avgEnergy,
            averageHealth: avgHealth
          },
          agents: [
            {
              position: [100.0, 100.0],
              health: 1.0,
              color: [0.2, 0.8, 0.2],
              boost: false
            }
          ]
        };
      }
    };
  }
`;

const snakeCaseWasmModule = `
  export default async function initWasm() { return {}; }
  export function version() { return "0.1.0-snake-case-control"; }
  export function default_init_options() { return { world_width: 1280, world_height: 720 }; }
  export function init_sim(opts) {
    let tick = 0;
    let elapsed = 0;
    let speed = 1;
    return {
      setSpeed(value) { speed = value; },
      drive(now) {
        const steps = Math.floor((now - elapsed) * speed * 60 / 1000);
        elapsed = now;
        this.tick(steps);
        return { scientific_steps: steps, automatic_steps_skipped: 0 };
      },
      snapshot() { return this.tick(0); },
      frame() {
        const snapshot = this.snapshot();
        return { schema: "scriptbots.browser-frame.v1", snapshot, identity: {
          sessionId: "1002", publicationRevision: String(snapshot.tick + 400),
          scientificRevision: String(snapshot.tick + 100), controlRevision: "20",
          configRevision: "10", tick: String(snapshot.tick)
        } };
      },
      tick(steps) {
        tick += steps;
        return {
          tick: tick,
          world: { width: 1280, height: 720, closed: true },
          summary: {
            agent_count: 256,
            births: 10,
            deaths: 2,
            average_energy: 1.5,
            average_health: 0.85
          },
          agents: []
        };
      }
    };
  }
`;

/**
 * Creates the HTTP server that serves the production index.html and main.js.
 */
function createServer(packageInfo) {
  return http.createServer((req, res) => {
    const url = new URL(req.url, "http://127.0.0.1");

    if (url.pathname === "/" || url.pathname === "/index.html") {
      res.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
      res.end(fs.readFileSync(path.join(webDir, "index.html")));
    } else if (url.pathname === "/main.js") {
      res.writeHead(200, { "Content-Type": "application/javascript; charset=utf-8" });
      res.end(fs.readFileSync(path.join(webDir, "main.js")));
    } else if (["/renderers/canvas.js", "/renderers/webgpu.js"].includes(url.pathname)) {
      res.writeHead(200, { "Content-Type": "application/javascript; charset=utf-8" });
      res.end(fs.readFileSync(path.join(webDir, url.pathname.slice(1))));
    } else if (url.pathname.startsWith("/pkg/")
        && (url.pathname === "/pkg/package.json" || Object.hasOwn(packageInfo.files, url.pathname.slice(5)))) {
      const file = url.pathname.slice(5);
      const contentType = file.endsWith(".wasm") ? "application/wasm"
        : file.endsWith(".js") ? "application/javascript; charset=utf-8"
        : file.endsWith(".json") ? "application/json; charset=utf-8" : "text/plain; charset=utf-8";
      res.writeHead(200, { "Content-Type": contentType });
      res.end(fs.readFileSync(path.join(packageDir, file)));
    } else {
      res.writeHead(404);
      res.end("Not found");
    }
  });
}

async function installPausedClock(context) {
  // Freeze before app navigation so neither page loading nor evaluate() can
  // advance the first 500 ms stats window outside the declared runFor() steps.
  const epoch = Date.UTC(2026, 0, 1);
  await context.clock.install({ time: epoch });
  await context.clock.pauseAt(epoch + 1_000);
}

async function runRendererTests(browser, baseUrl) {
  const proof = fs.mkdtempSync(path.join(process.env.SCRIPTBOTS_WASM_PROOF_DIR || "/tmp", "renderer-"));
  // The compositor PNG is the visible-frame observation. A later GPU-canvas
  // drawImage read can be blank even when this retained screenshot is not.
  // The actual backdrop and gradient are below80; controls below reject both.
  const brightThreshold = 80;
  const inspectScreenshot = (page, png) => page.evaluate(async ({ bytes, threshold }) => {
    const bitmap = await createImageBitmap(new Blob([new Uint8Array(bytes)], { type: "image/png" }));
    try {
      const canvas = document.createElement("canvas");
      canvas.width = bitmap.width;
      canvas.height = bitmap.height;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(bitmap, 0, 0);
      const pixels = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
      let brightPixels = 0;
      let maxChannel = 0;
      for (let offset = 0; offset < pixels.length; offset += 4) {
        const max = Math.max(pixels[offset], pixels[offset + 1], pixels[offset + 2]);
        if (max > threshold) brightPixels += 1;
        maxChannel = Math.max(maxChannel, max);
      }
      return { width: canvas.width, height: canvas.height, threshold, brightPixels, maxChannel };
    } finally {
      bitmap.close();
    }
  }, { bytes: Array.from(png), threshold: brightThreshold });
  const paintedPixels = (page) => page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => {
    const canvas = document.getElementById("sim-canvas");
    const copy = document.createElement("canvas");
    copy.width = canvas.width;
    copy.height = canvas.height;
    const ctx = copy.getContext("2d");
    ctx.drawImage(canvas, 0, 0);
    const bytes = ctx.getImageData(0, 0, copy.width, copy.height).data;
    let colored = 0;
    for (let offset = 0; offset < bytes.length; offset += 4) {
      if (bytes[offset] > 30 || bytes[offset + 1] > 35 || bytes[offset + 2] > 50) colored += 1;
    }
    resolve(colored);
  })));
  const exports = {};
  const retainDownload = async (page, download, label, expected) => {
    const file = path.join(proof, `${label}-download.png`);
    await download.saveAs(file);
    if (await download.failure()) throw new Error(`PNG download failed: ${await download.failure()}`);
    await page.waitForFunction(() => document.getElementById("capture-status").dataset.state === "ready");
    const status = await page.evaluate(() => ({ ...document.getElementById("capture-status").dataset }));
    for (const [key, value] of Object.entries(expected)) {
      if (status[key] !== value) throw new Error(`PNG ${key} differs from its canonical frame: ${JSON.stringify({ expected, status })}`);
    }
    const filename = download.suggestedFilename();
    const match = /^scriptbots-session-(\d+)-p(\d+)-s(\d+)-c(\d+)-g(\d+)-t(\d+)-(canvas|webgpu)\.png$/.exec(filename);
    const names = ["sessionId", "publicationRevision", "scientificRevision", "controlRevision", "configRevision", "tick", "renderer"];
    if (!match || names.some((key, index) => status[key] !== match[index + 1]) || status.filename !== filename) {
      throw new Error(`PNG filename/status disagree: ${JSON.stringify({ filename, status })}`);
    }
    const bytes = fs.readFileSync(file);
    const image = await inspectScreenshot(page, bytes);
    const size = await page.evaluate(() => {
      const canvas = document.getElementById("sim-canvas");
      return { width: canvas.width, height: canvas.height };
    });
    if (image.width !== size.width || image.height !== size.height || image.brightPixels <= 0) {
      throw new Error(`PNG is empty or has incorrect dimensions: ${JSON.stringify({ image, size })}`);
    }
    return { filename, status, image, sha256: createHash("sha256").update(bytes).digest("hex"), file };
  };
  const saveFrame = async (page, label) => {
    const expected = await page.evaluate(() => window.__rendererSim.frame().identity);
    try {
      const [download] = await Promise.all([page.waitForEvent("download"), page.click("#capture-btn")]);
      return await retainDownload(page, download, label, expected);
    } catch (error) {
      console.error("PNG download failure state", JSON.stringify({ label, expected,
        observed: await page.evaluate(() => ({ hidden: document.hidden,
          capture: { ...document.getElementById("capture-status").dataset },
          message: document.getElementById("capture-status").textContent,
          disabled: document.getElementById("capture-btn").disabled,
          renderer: document.getElementById("renderer-status").textContent,
          identity: window.__rendererSim.frame().identity,
          serializer: document.getElementById("sim-canvas").toBlob.toString(),
          serializationCalls: window.__nativePngCalls,
          anchorClicks: window.__pngAnchorClicks,
          log: document.getElementById("log").textContent })), errors }));
      throw error;
    }
  };
  const context = await browser.newContext({ deviceScaleFactor: 1 });
  const errors = [];
  const page = await context.newPage();
  let downloadCount = 0;
  page.on("download", () => { downloadCount += 1; });
  page.on("pageerror", (error) => errors.push(error.message));
  await context.addInitScript(() => {
    window.__nativePngCalls = [];
    window.__pngAnchorClicks = [];
    const toBlob = HTMLCanvasElement.prototype.toBlob;
    HTMLCanvasElement.prototype.toBlob = function (callback, ...args) {
      if (this.id !== "sim-canvas") return toBlob.call(this, callback, ...args);
      const call = { startedAt: performance.now(), width: this.width, height: this.height };
      window.__nativePngCalls.push(call);
      return toBlob.call(this, (blob) => {
        call.deliveredAt = performance.now();
        call.type = blob?.type;
        callback(blob);
        call.returnedAt = performance.now();
      }, ...args);
    };
    const click = HTMLAnchorElement.prototype.click;
    HTMLAnchorElement.prototype.click = function () {
      const call = { filename: this.download, startedAt: performance.now(), userActivation: navigator.userActivation.isActive };
      window.__pngAnchorClicks.push(call);
      const result = click.call(this);
      call.returnedAt = performance.now();
      return result;
    };
    const presentation = { hold: false, pending: null, native: window.requestAnimationFrame.bind(window) };
    window.requestAnimationFrame = (callback) => {
      if (presentation.hold) {
        if (presentation.pending) throw new Error("More than one presentation callback retained");
        presentation.pending = callback;
        return 0;
      }
      return presentation.native(callback);
    };
    window.__presentationControl = presentation;
    if (typeof GPUAdapter === "undefined") throw new Error("Actual WebGPU is required for this lane");
    const requestDevice = GPUAdapter.prototype.requestDevice;
    GPUAdapter.prototype.requestDevice = async function (...args) {
      window.__rendererAdapter = {
        vendor: this.info.vendor, architecture: this.info.architecture,
        device: this.info.device, description: this.info.description,
      };
      const device = await requestDevice.apply(this, args);
      window.__rendererDevice = device;
      return device;
    };
  });
  await page.goto(baseUrl);
  await page.waitForFunction(() => document.getElementById("renderer-status").dataset.renderer === "canvas");
  if (await page.inputValue("#renderer") !== "canvas") throw new Error("Unmeasured WebGPU became the default renderer");
  await page.selectOption("#renderer", "webgpu");
  await page.waitForFunction(() => Number(document.getElementById("renderer-status").dataset.snapshotTick) > 0);
  await page.evaluate(() => {
    const speed = document.getElementById("steps-per-frame");
    speed.value = "0";
    speed.dispatchEvent(new Event("input", { bubbles: true }));
  });
  const screenshotControl = await page.evaluate(() => {
    const canvas = document.createElement("canvas");
    canvas.id = "screenshot-control";
    canvas.width = 64;
    canvas.height = 64;
    canvas.style.cssText = "position:fixed;top:0;right:0;z-index:1;";
    canvas.style.width = `${canvas.width}px`;
    canvas.style.height = `${canvas.height}px`;
    canvas.style.backgroundColor = getComputedStyle(document.documentElement).backgroundColor;
    canvas.style.backgroundImage = getComputedStyle(document.getElementById("sim-container")).backgroundImage;
    canvas.getContext("2d");
    document.body.append(canvas);
    return { width: canvas.width, height: canvas.height, marker: { x: 16, y: 16, width: 1, height: 1 } };
  });
  const transparentBlank = await inspectScreenshot(page, await page.locator("#screenshot-control").screenshot());
  await page.evaluate(async () => {
    const { CanvasRenderer } = await import("/renderers/canvas.js");
    new CanvasRenderer(document.getElementById("screenshot-control"))
      .render({ world: { width: 64, height: 64 }, agents: [] });
  });
  const emptyFrame = await inspectScreenshot(page, await page.locator("#screenshot-control").screenshot());
  await page.evaluate((marker) => {
    const ctx = document.getElementById("screenshot-control").getContext("2d");
    ctx.fillStyle = "#ff0000";
    ctx.fillRect(marker.x, marker.y, marker.width, marker.height);
  }, screenshotControl.marker);
  const knownMarker = await inspectScreenshot(page, await page.locator("#screenshot-control").screenshot());
  const markerArea = screenshotControl.marker.width * screenshotControl.marker.height;
  if (transparentBlank.brightPixels !== 0 || emptyFrame.brightPixels !== 0
      || knownMarker.brightPixels !== markerArea || knownMarker.width !== screenshotControl.width
      || knownMarker.height !== screenshotControl.height) {
    throw new Error(`Invalid screenshot blank guard: ${JSON.stringify({ transparentBlank, emptyFrame, knownMarker })}`);
  }
  await page.evaluate(() => document.getElementById("screenshot-control").remove());
  // Known render inputs exercise the actual production modules and real GPU;
  // they are geometry fixtures, not simulation or hardware-qualification proof.
  const geometry = await page.evaluate(async () => {
    const { WebGpuRenderer } = await import("/renderers/webgpu.js");
    const { CanvasRenderer } = await import("/renderers/canvas.js");
    const results = {};
    for (const backend of ["canvas", "webgpu"]) {
      const canvas = document.createElement("canvas");
      canvas.width = 1280;
      canvas.height = 720;
      const renderer = backend === "webgpu" ? await WebGpuRenderer.create(canvas) : new CanvasRenderer(canvas);
      const snapshot = {
        tick: 0, world: { width: 2560, height: 1440 },
        agents: [
          { position: [640, 360], health: 1, color: [0.2, 0.8, 0.2], boost: true },
          { position: [1280, 720], health: 1, color: [0.9, 0.1, 0.1], boost: false },
        ],
      };
      let capturedPixels = null;
      const read = async (input, points) => {
        if (input) renderer.render(input);
        const pendingPng = capturedPixels === null && input
          ? new Promise((resolve) => canvas.toBlob(resolve, "image/png")) : null;
        if (renderer.device) await renderer.device.queue.onSubmittedWorkDone();
        const copy = document.createElement("canvas");
        copy.width = canvas.width;
        copy.height = canvas.height;
        const ctx = copy.getContext("2d");
        ctx.drawImage(canvas, 0, 0);
        const pixels = points.map(([x, y]) => Array.from(ctx.getImageData(x, y, 1, 1).data));
        if (pendingPng) {
          const png = await pendingPng;
          if (!png || png.type !== "image/png") throw new Error(`${backend} did not serialize a PNG`);
          const bitmap = await createImageBitmap(png);
          try {
            ctx.clearRect(0, 0, copy.width, copy.height);
            ctx.drawImage(bitmap, 0, 0);
            capturedPixels = points.map(([x, y]) => Array.from(ctx.getImageData(x, y, 1, 1).data));
            if (JSON.stringify(capturedPixels) !== JSON.stringify(pixels)) {
              throw new Error(`${backend} PNG differs from the submitted geometry: ${JSON.stringify({ capturedPixels, pixels })}`);
            }
          } finally {
            bitmap.close();
          }
        }
        return pixels;
      };
      try {
        const pixels = await read(snapshot, [[320, 180], [640, 360], [644, 364], [325, 180]]);
        const [green, red, corner, ring] = pixels;
        if (green[1] <= green[0] + 70 || red[0] <= red[1] + 70
            || corner[0] > 15 || corner[1] > 22 || corner[2] > 38
            || ring[2] <= ring[0] + 70) {
          throw new Error(`${backend} failed world scaling, circular body or boost ring: ${JSON.stringify(pixels)}`);
        }
        const empty = await read({ ...snapshot, agents: [] }, [[320, 180]]);
        if (empty[0][0] > 15 || empty[0][1] > 22 || empty[0][2] > 38 || empty[0][3] !== 255) {
          throw new Error(`${backend} did not clear an empty frame: ${empty}`);
        }
        // Grow beyond both the initial256-agent and subsequent512-agent buffers.
        await read({ ...snapshot, agents: Array(513).fill(snapshot.agents[0]) }, [[320, 180]]);
        canvas.width = 512;
        canvas.height = 256;
        const resized = await read(snapshot, [[128, 64], [256, 128]]);
        if (resized[0][1] <= resized[0][0] + 70 || resized[1][0] <= resized[1][1] + 70) {
          throw new Error(`${backend} did not rescale after resize: ${resized}`);
        }
        let capacityError = null;
        if (renderer.device) {
          try { renderer.ensureAgentCapacity(Number.MAX_SAFE_INTEGER); }
          catch (error) { capacityError = String(error); }
          if (!capacityError?.includes("exceeds WebGPU capacity")) throw new Error("Missing host capacity refusal");
        }
        let admission = null;
        if (renderer.device) {
          admission = [];
          const queue = renderer.device.queue;
          const workDone = queue.onSubmittedWorkDone.bind(queue);
          const submit = queue.submit.bind(queue);
          const writeBuffer = queue.writeBuffer.bind(queue);
          let submissions = 0;
          let writes = 0;
          let heldCompletion;
          let release;
          queue.submit = (...args) => { submissions += 1; return submit(...args); };
          queue.writeBuffer = (...args) => { writes += 1; return writeBuffer(...args); };
          try {
            for (const initial of [snapshot, { ...snapshot, agents: [] }]) {
              submissions = 0;
              writes = 0;
              release = null;
              queue.onSubmittedWorkDone = () => {
                const actual = workDone();
                heldCompletion = new Promise((resolve, reject) => {
                  release = () => actual.then(resolve, reject);
                });
                return heldCompletion;
              };
              const first = renderer.render(initial);
              const firstWrites = writes;
              const refused = Array.from({ length: 100 }, () => renderer.render({ ...snapshot, agents: [] }));
              const blockedSubmissions = submissions;
              const blockedWrites = writes;
              if (first !== true || !refused.every((value) => value === false)
                  || submissions !== 1 || writes !== firstWrites || !release) {
                throw new Error(`Unbounded GPU admission: ${JSON.stringify({ first, refused, submissions, writes, firstWrites })}`);
              }
              queue.onSubmittedWorkDone = workDone;
              release();
              await heldCompletion;
              const newest = { ...snapshot, agents: [snapshot.agents[1]] };
              const admitted = renderer.render(newest);
              await workDone();
              if (admitted !== true || submissions !== 2) throw new Error("GPU admission did not resume after completion");
              const newestPixels = await read(null, [[128, 64], [256, 128]]);
              if (submissions !== 2 || newestPixels[0][0] > 15 || newestPixels[0][1] > 22 || newestPixels[0][2] > 38
                  || newestPixels[1][0] <= newestPixels[1][1] + 70) {
                throw new Error(`GPU resumed with stale input: ${JSON.stringify(newestPixels)}`);
              }
              const observed = [first, ...refused];
              admission.push({ initialAgents: initial.agents.length, attempts: observed.length,
                refused: observed.filter((value) => value === false).length,
                admitted: observed.filter((value) => value === true).length,
                blockedSubmissions, blockedWrites, newestPixels });
            }
          } finally {
            queue.onSubmittedWorkDone = workDone;
            queue.submit = submit;
            queue.writeBuffer = writeBuffer;
          }
        }
        results[backend] = { pixels, capturedPixels, empty, resized, growthAgents: 513, capacityError, admission };
      } finally {
        renderer.destroy();
      }
    }
    return results;
  });
  console.log(JSON.stringify({ schema: "scriptbots.browser-frame-admission-geometry.v1", observation: geometry }));
  await page.waitForFunction(() => document.getElementById("renderer-status")?.dataset.renderer === "webgpu");
  const aspect = await page.evaluate(() => {
    const canvas = document.getElementById("sim-canvas");
    const bounds = canvas.getBoundingClientRect();
    return { displayed: bounds.width / bounds.height, intrinsic: canvas.width / canvas.height };
  });
  if (Math.abs(aspect.displayed - aspect.intrinsic) > 0.01) throw new Error(`Canvas CSS stretches the world: ${JSON.stringify(aspect)}`);
  await page.evaluate(async () => {
    const wasm = await import("/pkg/scriptbots_web.js");
    const { CanvasRenderer } = await import("/renderers/canvas.js");
    const render = CanvasRenderer.prototype.render;
    window.__canvasRenderCalls = 0;
    CanvasRenderer.prototype.render = function (snapshot) {
      if (this.canvas.id === "sim-canvas") window.__canvasRenderCalls += 1;
      return render.call(this, snapshot);
    };
    const frame = wasm.SimHandle.prototype.frame;
    wasm.SimHandle.prototype.frame = function () {
      window.__rendererSim = this;
      return frame.call(this);
    };
    const drive = wasm.SimHandle.prototype.drive;
    wasm.SimHandle.prototype.drive = function (now) {
      const receipt = drive.call(this, now);
      window.__lastRendererDrive = { now, receipt };
      return receipt;
    };
    const setSpeed = wasm.SimHandle.prototype.setSpeed;
    wasm.SimHandle.prototype.setSpeed = function (value) {
      const receipt = setSpeed.call(this, value);
      window.__lastRendererSpeed = { value, receipt };
      return receipt;
    };
    window.__captureUrls = new Set();
    const create = URL.createObjectURL.bind(URL);
    const revoke = URL.revokeObjectURL.bind(URL);
    URL.createObjectURL = (blob) => {
      const url = create(blob);
      window.__captureUrls.add(url);
      return url;
    };
    URL.revokeObjectURL = (url) => {
      window.__captureUrls.delete(url);
      return revoke(url);
    };
  });
  await page.evaluate(() => {
    const speed = document.getElementById("steps-per-frame");
    speed.value = "0";
    speed.dispatchEvent(new Event("input", { bubbles: true }));
  });
  await page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));
  await page.waitForFunction(() => !!document.getElementById("renderer-status").dataset.snapshotTick);
  const pausedTick = await page.getAttribute("#renderer-status", "data-snapshot-tick");
  const snapshotBefore = await page.evaluate(() => JSON.stringify(window.__rendererSim.snapshot()));
  if (String(JSON.parse(snapshotBefore).tick) !== pausedTick) throw new Error("Renderer snapshot tick differs from live host");
  await page.selectOption("#renderer", "canvas");
  await page.waitForFunction(() => document.getElementById("renderer-status").dataset.renderer === "canvas"
    && !!document.getElementById("renderer-status").dataset.snapshotTick);
  await page.evaluate(() => { window.__presentationControl.hold = true; });
  await page.waitForFunction(() => !!window.__presentationControl.pending, undefined, { polling: 50 });
  const pausedPaints = await page.evaluate(() => {
    const presentation = window.__presentationControl;
    const invoke = () => {
      const callback = presentation.pending;
      presentation.pending = null;
      callback(performance.now());
    };
    const before = window.__canvasRenderCalls;
    const snapshot = JSON.stringify(window.__rendererSim.snapshot());
    const repetitions = 120;
    const canvas = document.getElementById("sim-canvas");
    const width = canvas.width;
    for (let index = 0; index < repetitions; index += 1) invoke();
    const afterUnchanged = window.__canvasRenderCalls;
    canvas.width = width + 1;
    invoke();
    const afterResize = window.__canvasRenderCalls;
    canvas.width = width;
    invoke();
    const afterRestore = window.__canvasRenderCalls;
    presentation.hold = false;
    const next = presentation.pending;
    presentation.pending = null;
    presentation.native(next);
    return { repetitions, unchangedCalls: afterUnchanged - before,
      resizeCalls: afterResize - afterUnchanged, restoreCalls: afterRestore - afterResize,
      unchangedState: snapshot === JSON.stringify(window.__rendererSim.snapshot()) };
  });
  if (pausedPaints.unchangedCalls !== 0 || pausedPaints.resizeCalls !== 1 || pausedPaints.restoreCalls !== 1
      || !pausedPaints.unchangedState) throw new Error(`Paused publication was repainted or resize ignored: ${JSON.stringify(pausedPaints)}`);
  await page.locator("#sim-canvas").screenshot({ path: path.join(proof, "main-canvas.png") });
  exports.canvas = await saveFrame(page, "canvas");
  if (await page.evaluate(() => JSON.stringify(window.__rendererSim.snapshot())) !== snapshotBefore) throw new Error("Canvas switch changed exported snapshot");
  await page.selectOption("#renderer", "webgpu");
  await page.waitForFunction(() => document.getElementById("renderer-status").dataset.renderer === "webgpu"
    && !!document.getElementById("renderer-status").dataset.snapshotTick);
  const beforeScreenshot = await page.locator("#sim-canvas").boundingBox();
  let gpuPng;
  try {
    gpuPng = await page.locator("#sim-canvas").screenshot({ path: path.join(proof, "main-webgpu.png") });
  } catch (error) {
    console.error("WebGPU screenshot failure state", JSON.stringify({ beforeScreenshot,
      afterScreenshot: await page.locator("#sim-canvas").boundingBox(),
      page: await page.evaluate(() => ({ hidden: document.hidden, status: document.getElementById("renderer-status").textContent,
        tick: window.__rendererSim.snapshot().tick, log: document.getElementById("log").textContent })), errors }));
    throw error;
  }
  const gpuCanvasCopyPixels = await paintedPixels(page);
  const gpuScreenshot = await inspectScreenshot(page, gpuPng);
  if (gpuScreenshot.brightPixels <= 0) throw new Error("Selected WebGPU screenshot has no bright agent pixels");
  if (await page.evaluate(() => JSON.stringify(window.__rendererSim.snapshot())) !== snapshotBefore) throw new Error("GPU switch changed exported snapshot");
  const adapter = await page.evaluate(() => window.__rendererAdapter);
  exports.webgpu = await saveFrame(page, "webgpu");
  await page.evaluate(async () => {
    const device = window.__rendererDevice;
    device.destroy();
    await device.lost;
  });
  await page.waitForFunction(() => document.getElementById("renderer-status").dataset.renderer === "canvas"
    && document.getElementById("renderer-status").textContent.includes("device lost")
    && !!document.getElementById("renderer-status").dataset.snapshotTick);
  await page.locator("#sim-canvas").screenshot({ path: path.join(proof, "main-device-lost.png") });
  const lostPaintedPixels = await paintedPixels(page);
  if (lostPaintedPixels <= 0) throw new Error("Device loss fallback painted no agents");
  if (await page.evaluate(() => JSON.stringify(window.__rendererSim.snapshot())) !== snapshotBefore) throw new Error("Device loss changed exported snapshot");

  // Acknowledge real GPU work only after the app has advanced science. This
  // controls delivery of completion, not whether the submitted work executes.
  await page.selectOption("#renderer", "canvas");
  await page.selectOption("#renderer", "webgpu");
  await page.waitForFunction(() => document.getElementById("renderer-status").dataset.renderer === "webgpu"
    && !!document.getElementById("renderer-status").dataset.snapshotTick);
  await page.evaluate(() => {
    const queue = window.__rendererDevice.queue;
    const done = queue.onSubmittedWorkDone.bind(queue);
    const submit = queue.submit.bind(queue);
    const hold = { submissions: 0, release: null, completion: null };
    hold.restore = () => { queue.onSubmittedWorkDone = done; queue.submit = submit; };
    queue.submit = (...args) => { hold.submissions += 1; return submit(...args); };
    queue.onSubmittedWorkDone = () => {
      const actual = done();
      hold.completion = new Promise((resolve, reject) => { hold.release = () => actual.then(resolve, reject); });
      return hold.completion;
    };
    window.__frameHold = hold;
    const speed = document.getElementById("steps-per-frame");
    speed.value = "1";
    speed.dispatchEvent(new Event("input", { bubbles: true }));
  });
  await page.waitForFunction(() => !!window.__frameHold.release, undefined, { polling: 50 });
  await page.evaluate(() => {
    const speed = document.getElementById("steps-per-frame");
    speed.value = "0";
    speed.dispatchEvent(new Event("input", { bubbles: true }));
    window.__presentationControl.hold = true;
  });
  await page.waitForFunction(() => !!window.__presentationControl.pending, undefined, { polling: 50 });
  const beforeHold = await page.evaluate(() => ({
    tick: window.__rendererSim.snapshot().tick,
    submittedTick: Number(document.getElementById("renderer-status").dataset.snapshotTick),
  }));
  const beforeHeldDownloads = downloadCount;
  await page.evaluate(() => {
    const button = document.getElementById("capture-btn");
    button.click();
    button.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    if (!button.disabled || document.getElementById("capture-status").dataset.state !== "waiting") {
      throw new Error("PNG request was not bounded while GPU admission was held");
    }
  });
  await page.evaluate(() => {
    const speed = document.getElementById("steps-per-frame");
    speed.value = "2";
    speed.dispatchEvent(new Event("input", { bubbles: true }));
  });
  try {
    await page.waitForFunction((tick) => window.__rendererSim.snapshot().tick > tick + 2, beforeHold.tick, { polling: 50 });
  } catch (error) {
    console.error("Independent clock failure state", JSON.stringify({ beforeHold,
      observed: await page.evaluate(() => ({ hidden: document.hidden, tick: window.__rendererSim.snapshot().tick,
        population: window.__rendererSim.snapshot().agents.length, drive: window.__lastRendererDrive,
        speed: window.__lastRendererSpeed, status: document.getElementById("renderer-status").textContent,
        log: document.getElementById("log").textContent })), errors }));
    throw error;
  }
  const duringHold = await page.evaluate(() => {
    const speed = document.getElementById("steps-per-frame");
    speed.value = "0";
    speed.dispatchEvent(new Event("input", { bubbles: true }));
    const beforePresentation = JSON.stringify(window.__rendererSim.snapshot());
    const presentation = window.__presentationControl;
    const callback = presentation.pending;
    presentation.pending = null;
    callback(performance.now());
    if (JSON.stringify(window.__rendererSim.snapshot()) !== beforePresentation) throw new Error("Presentation changed exported state");
    return { tick: window.__rendererSim.snapshot().tick,
      submittedTick: Number(document.getElementById("renderer-status").dataset.snapshotTick),
      submissions: window.__frameHold.submissions };
  });
  if (duringHold.tick <= beforeHold.tick || duringHold.submittedTick !== beforeHold.submittedTick
      || duringHold.submissions !== 1 || downloadCount !== beforeHeldDownloads
      || await page.getAttribute("#capture-status", "data-state") !== "waiting") {
    throw new Error(`GPU backlog or false snapshot stamp: ${JSON.stringify({ beforeHold, duringHold })}`);
  }
  const heldIdentity = await page.evaluate(() => window.__rendererSim.frame().identity);
  const heldDownload = page.waitForEvent("download");
  await page.evaluate(async () => {
    const hold = window.__frameHold;
    hold.restore();
    hold.release();
    await hold.completion;
    const presentation = window.__presentationControl;
    const callback = presentation.pending;
    presentation.pending = null;
    callback(performance.now());
    presentation.hold = false;
    const next = presentation.pending;
    presentation.pending = null;
    presentation.native(next);
  });
  await page.waitForFunction((tick) => Number(document.getElementById("renderer-status").dataset.snapshotTick) === tick, duringHold.tick);
  const afterHold = Number(await page.getAttribute("#renderer-status", "data-snapshot-tick"));
  exports.afterHeldAdmission = await retainDownload(page, await heldDownload, "after-held-admission", heldIdentity);
  const completionSnapshot = await page.evaluate(() => JSON.stringify(window.__rendererSim.snapshot()));
  await page.evaluate(() => {
    window.__rendererDevice.queue.onSubmittedWorkDone = () => Promise.reject(new Error("intentional completion acknowledgement refusal"));
    document.getElementById("sim-canvas").width += 1;
  });
  await page.waitForFunction(() => document.getElementById("renderer-status").dataset.renderer === "canvas"
    && document.getElementById("renderer-status").textContent.includes("intentional completion acknowledgement refusal")
    && !!document.getElementById("renderer-status").dataset.snapshotTick);
  await page.evaluate(() => { document.getElementById("sim-canvas").width -= 1; });
  const rejectedCompletionPixels = await paintedPixels(page);
  if (rejectedCompletionPixels <= 0 || await page.evaluate(() => JSON.stringify(window.__rendererSim.snapshot())) !== completionSnapshot) {
    throw new Error("Injected completion rejection failed to retain a painted, unchanged paused snapshot");
  }
  // Hold delivery of a real serialization callback, not its pixel production.
  const holdSerialization = () => page.evaluate(() => {
    const canvas = document.getElementById("sim-canvas");
    const original = canvas.toBlob;
    window.__releasePng = null;
    window.__restorePng = () => { canvas.toBlob = original; };
    canvas.toBlob = function (callback, type) {
      return original.call(this, (blob) => { window.__releasePng = () => callback(blob); }, type);
    };
  });
  const beforeEncoding = await page.evaluate(() => window.__rendererSim.frame().identity);
  await holdSerialization();
  await page.click("#capture-btn");
  await page.waitForFunction(() => !!window.__releasePng);
  await page.evaluate(() => {
    const speed = document.getElementById("steps-per-frame");
    speed.value = "2";
    speed.dispatchEvent(new Event("input", { bubbles: true }));
  });
  await page.waitForFunction((tick) => window.__rendererSim.snapshot().tick > tick + 2, Number(beforeEncoding.tick), { polling: 50 });
  await page.evaluate(() => {
    const speed = document.getElementById("steps-per-frame");
    speed.value = "0";
    speed.dispatchEvent(new Event("input", { bubbles: true }));
  });
  const afterEncoding = await page.evaluate(() => window.__rendererSim.frame().identity);
  const movedDownload = page.waitForEvent("download");
  await page.evaluate(() => { window.__restorePng(); window.__releasePng(); });
  exports.whileScienceMoves = await retainDownload(page, await movedDownload, "while-science-moves", beforeEncoding);
  exports.whileScienceMoves.laterIdentity = afterEncoding;
  if (beforeEncoding.tick === afterEncoding.tick || beforeEncoding.publicationRevision === afterEncoding.publicationRevision) {
    throw new Error("PNG callback test did not observe a later science publication");
  }
  const captureFailuresSnapshot = await page.evaluate(() => JSON.stringify(window.__rendererSim.snapshot()));
  const captureFailures = {};
  for (const mode of ["null", "throw", "identity"]) {
    const before = downloadCount;
    await page.evaluate(async (mode) => {
      const canvas = document.getElementById("sim-canvas");
      const wasm = await import("/pkg/scriptbots_web.js");
      const originalBlob = canvas.toBlob;
      const originalFrame = wasm.SimHandle.prototype.frame;
      window.__restoreCaptureFault = () => {
        canvas.toBlob = originalBlob;
        wasm.SimHandle.prototype.frame = originalFrame;
      };
      if (mode === "null") canvas.toBlob = (callback) => callback(null);
      if (mode === "throw") canvas.toBlob = () => { throw new DOMException("intentional PNG serialization refusal", "SecurityError"); };
      if (mode === "identity") wasm.SimHandle.prototype.frame = function () {
        const frame = originalFrame.call(this);
        frame.identity.tick = String(frame.snapshot.tick + 1);
        return frame;
      };
    }, mode);
    await page.click("#capture-btn");
    await page.waitForFunction(() => document.getElementById("capture-status").dataset.state === "error");
    const failure = await page.textContent("#capture-status");
    await page.evaluate(() => window.__restoreCaptureFault());
    if (downloadCount !== before || await page.isDisabled("#capture-btn")
        || await page.evaluate(() => JSON.stringify(window.__rendererSim.snapshot())) !== captureFailuresSnapshot) {
      throw new Error(`PNG ${mode} refusal downloaded or changed paused science`);
    }
    captureFailures[mode] = { injected: true, failure, downloads: downloadCount - before };
  }

  await holdSerialization();
  await page.click("#capture-btn");
  await page.waitForFunction(() => !!window.__releasePng);
  const beforeResetDownloads = downloadCount;
  await page.evaluate(() => {
    window.__beforeResetSim = window.__rendererSim;
    document.getElementById("reset-btn").click();
  });
  await page.waitForFunction(() => window.__rendererSim !== window.__beforeResetSim);
  const resetSnapshot = await page.evaluate(() => JSON.stringify(window.__rendererSim.snapshot()));
  await page.evaluate(() => { window.__restorePng(); window.__releasePng(); });
  await page.waitForFunction(() => document.getElementById("capture-status").dataset.state === "error");
  captureFailures.resetDuringEncoding = { injectedCallbackHold: true, failure: await page.textContent("#capture-status") };
  if (downloadCount !== beforeResetDownloads
      || await page.evaluate(() => JSON.stringify(window.__rendererSim.snapshot())) !== resetSnapshot) {
    throw new Error("Old-world PNG completed after reset or changed the new paused world");
  }
  await holdSerialization();
  await page.click("#capture-btn");
  await page.waitForFunction(() => !!window.__releasePng);
  const beforeTimeoutDownloads = downloadCount;
  await page.waitForFunction(() => document.getElementById("capture-status").dataset.state === "error");
  captureFailures.timeout = { injectedCallbackHold: true, failure: await page.textContent("#capture-status") };
  if (!captureFailures.timeout.failure.includes("10 seconds") || downloadCount !== beforeTimeoutDownloads) {
    throw new Error("PNG serialization deadline did not refuse the held callback");
  }
  const indeterminateEncoder = await page.evaluate(() => {
    const callsBefore = window.__nativePngCalls.length;
    const button = document.getElementById("capture-btn");
    const disabled = button.disabled;
    button.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    return { callsBefore, disabled };
  });
  await page.evaluate(() => new Promise(requestAnimationFrame));
  indeterminateEncoder.callsAfter = await page.evaluate(() => window.__nativePngCalls.length);
  if (!indeterminateEncoder.disabled || indeterminateEncoder.callsBefore !== indeterminateEncoder.callsAfter) {
    throw new Error(`Retry started another encoder before the prior callback returned: ${JSON.stringify(indeterminateEncoder)}`);
  }
  captureFailures.timeout.indeterminateEncoder = indeterminateEncoder;
  await page.evaluate(() => { window.__restorePng(); window.__releasePng(); });
  await page.waitForFunction(() => !document.getElementById("capture-btn").disabled);
  exports.afterTimeout = await saveFrame(page, "after-timeout");
  const afterTimeoutDownloads = downloadCount;
  await page.evaluate(() => window.__releasePng());
  if (downloadCount !== afterTimeoutDownloads
      || await page.getAttribute("#capture-status", "data-filename") !== exports.afterTimeout.filename) {
    throw new Error("Duplicate stale PNG callback replaced the successful retry");
  }
  captureFailures.timeout.duplicateCallbackInjected = true;
  const urlLifecycle = await page.evaluate(() => {
    const retained = window.__captureUrls.size;
    window.dispatchEvent(new PageTransitionEvent("pagehide"));
    return { retained, afterPagehideSignal: window.__captureUrls.size, control: "injected pagehide event" };
  });
  if (urlLifecycle.retained !== 1 || urlLifecycle.afterPagehideSignal !== 0) {
    throw new Error(`PNG object URLs are unbounded or not released: ${JSON.stringify(urlLifecycle)}`);
  }
  await context.close();
  const fallbacks = {};
  for (const failure of ["unavailable", "adapter-null", "requestDevice-rejected", "requestDevice-pending"]) {
    const fallbackContext = await browser.newContext();
    const fallbackPage = await fallbackContext.newPage();
    fallbackPage.on("pageerror", (error) => errors.push(error.message));
    await fallbackContext.addInitScript((mode) => {
      if (mode === "unavailable") Object.defineProperty(navigator, "gpu", { value: undefined });
      else if (mode === "adapter-null") Object.defineProperty(navigator, "gpu", { value: { requestAdapter: async () => null } });
      else if (mode === "requestDevice-pending") GPUAdapter.prototype.requestDevice = () => new Promise(() => {});
      else GPUAdapter.prototype.requestDevice = async () => { throw new Error("intentional device creation refusal"); };
    }, failure);
    await fallbackPage.goto(baseUrl);
    await fallbackPage.selectOption("#renderer", "webgpu");
    await fallbackPage.waitForFunction(() => document.getElementById("renderer-status").dataset.renderer === "canvas"
      && Number(document.getElementById("renderer-status").dataset.snapshotTick) > 0);
    const status = await fallbackPage.textContent("#renderer-status");
    const expected = { unavailable: "not supported", "adapter-null": "Unable to acquire GPU adapter",
      "requestDevice-rejected": "intentional device creation refusal", "requestDevice-pending": "starting WebGPU" }[failure];
    if (!status.includes(expected)) throw new Error(`Fallback hid its cause: ${status}`);
    const painting = await paintedPixels(fallbackPage);
    if (painting <= 0) throw new Error(`Fallback had no painted agents: ${failure}`);
    fallbacks[failure] = { status, paintedPixels: painting };
    await fallbackPage.locator("#sim-canvas").screenshot({ path: path.join(proof, `${failure}.png`) });
    await fallbackContext.close();
  }
  if (errors.length !== 0) throw new Error(`Renderer page errors: ${JSON.stringify(errors)}`);
  const result = { schema: "scriptbots.browser-renderers.v2", status: "pass", browser: browser.version(), adapter,
    geometry, aspect, pausedTick, pausedPaints, exportedSnapshotSha256: createHash("sha256").update(snapshotBefore).digest("hex"),
    backpressure: { beforeHold, duringHold, afterHold,
      presentationControl: "one producer callback retained and invoked explicitly; real independent timer" },
    injectedCompletionRejection: { paintedPixels: rejectedCompletionPixels,
      exportedSnapshotSha256: createHash("sha256").update(completionSnapshot).digest("hex") },
    exports, captureFailures, urlLifecycle, screenshotControls: { transparentBlank, emptyFrame, knownMarker }, gpuCanvasCopyPixels, gpuScreenshot,
    lostPaintedPixels, deviceLoss: "actual GPUDevice.destroy, followed by Canvas painting", fallbacks, errors, proof };
  console.log(JSON.stringify(result));
  return true;
}

async function runTests() {
  const observations = {
    camelCasePositive: null,
    snakeCaseNegative: null,
    schedulingNegative: null,
    realPackage: null,
    consoleLogs: [],
    pageErrors: [],
  };

  const packageInfo = readPackage();
  const server = createServer(packageInfo);
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const port = server.address().port;
  const baseUrl = `http://127.0.0.1:${port}`;

  const browser = await chromium.launch({
    headless: true,
    executablePath: process.env.SCRIPTBOTS_CHROMIUM_EXECUTABLE || undefined,
    args: process.env.SCRIPTBOTS_WEB_RENDERER_TESTS === "1"
      ? ["--enable-unsafe-webgpu", "--use-angle=swiftshader", "--enable-features=Vulkan", "--use-vulkan=swiftshader", "--disable-vulkan-surface"] : [],
  });

  try {
    if (process.env.SCRIPTBOTS_WEB_RENDERER_TESTS === "1") return await runRendererTests(browser, baseUrl);
    // =========================================================================
    // Test 1: Production camelCase Contract & Multiple Window TPS Deltas
    // =========================================================================
    {
      const context = await browser.newContext();
      await installPausedClock(context);
      const page = await context.newPage();

      const pageLogs = [];
      const pageErrors = [];
      page.on("console", (msg) => pageLogs.push(msg.text()));
      page.on("pageerror", (err) => pageErrors.push(err.message));

      await page.route("**/pkg/scriptbots_web.js", (route) =>
        route.fulfill({
          status: 200,
          contentType: "application/javascript; charset=utf-8",
          body: camelCaseWasmModule,
        }),
      );

      await page.goto(`${baseUrl}/`);

      // Verify initialization text in header
      const versionText = await page.textContent("#version");
      if (versionText !== "0.1.0-camel-case-prod") {
        throw new Error(`Expected version '0.1.0-camel-case-prod', got '${versionText}'`);
      }

      // Check pre-500ms state: at t = 200ms, stats boundary must NOT have fired yet.
      await context.clock.runFor(200);
      const earlyPop = await page.textContent("#metric-population");
      const earlyTps = await page.textContent("#metric-tps");
      const earlyEnergy = await page.textContent("#metric-energy");
      const earlyHealth = await page.textContent("#metric-health");

      if (earlyPop !== "–" || earlyTps !== "–" || earlyEnergy !== "–" || earlyHealth !== "–") {
        throw new Error(
          `Scheduling defect: stats updated before 500ms boundary: pop=${earlyPop}, tps=${earlyTps}`,
        );
      }

      // Advance clock past 500ms: run additional 350ms (total 550ms) -> Window 1 fires!
      await context.clock.runFor(350);

      const w1 = {
        pop: await page.textContent("#metric-population"),
        energy: await page.textContent("#metric-energy"),
        health: await page.textContent("#metric-health"),
        tps: await page.textContent("#metric-tps"),
        fps: await page.textContent("#metric-fps"),
        tick: await page.textContent("#metric-tick"),
        births: await page.textContent("#metric-births"),
        deaths: await page.textContent("#metric-deaths"),
      };

      // Assert Window 1 DOM node values
      if (w1.pop === "–" || w1.tps === "–" || w1.energy === "–" || w1.health === "–") {
        throw new Error(
          `Window 1 failed to refresh DOM metrics after 550ms: ${JSON.stringify(w1)}`,
        );
      }

      const popNum1 = Number.parseInt(w1.pop.replace(/,/g, ""), 10);
      const tpsNum1 = Number.parseFloat(w1.tps);
      const energyNum1 = Number.parseFloat(w1.energy);
      const healthNum1 = Number.parseFloat(w1.health);
      const tickNum1 = Number.parseInt(w1.tick.replace(/,/g, ""), 10);

      if (Number.isNaN(popNum1) || popNum1 <= 200) {
        throw new Error(`Invalid population in window 1: ${w1.pop}`);
      }
      if (Number.isNaN(tpsNum1) || tpsNum1 <= 0) {
        throw new Error(`Invalid TPS in window 1: ${w1.tps}`);
      }
      if (Number.isNaN(energyNum1) || energyNum1 <= 1.0) {
        throw new Error(`Invalid average energy in window 1: ${w1.energy}`);
      }
      if (Number.isNaN(healthNum1) || healthNum1 <= 0.0 || healthNum1 > 1.0) {
        throw new Error(`Invalid average health in window 1: ${w1.health}`);
      }

      // Check intermediate frame (t = 800ms, statsDt = 250ms < 500ms): DOM must remain at w1 values
      await context.clock.runFor(250);
      const interPop = await page.textContent("#metric-population");
      if (interPop !== w1.pop) {
        throw new Error(
          `Stats refreshed prematurely at t=800ms: expected ${w1.pop}, got ${interPop}`,
        );
      }

      // Advance clock past second 500ms boundary: run additional 350ms (total 1150ms) -> Window 2 fires!
      await context.clock.runFor(350);

      const w2 = {
        pop: await page.textContent("#metric-population"),
        energy: await page.textContent("#metric-energy"),
        health: await page.textContent("#metric-health"),
        tps: await page.textContent("#metric-tps"),
        fps: await page.textContent("#metric-fps"),
        tick: await page.textContent("#metric-tick"),
        births: await page.textContent("#metric-births"),
        deaths: await page.textContent("#metric-deaths"),
      };

      const popNum2 = Number.parseInt(w2.pop.replace(/,/g, ""), 10);
      const tpsNum2 = Number.parseFloat(w2.tps);
      const tickNum2 = Number.parseInt(w2.tick.replace(/,/g, ""), 10);
      const energyNum2 = Number.parseFloat(w2.energy);
      const healthNum2 = Number.parseFloat(w2.health);

      // Verify that Window 2 progressed beyond Window 1
      if (popNum2 <= popNum1) {
        throw new Error(`Population did not advance in window 2: w1=${popNum1}, w2=${popNum2}`);
      }
      if (tickNum2 <= tickNum1) {
        throw new Error(`Tick did not advance in window 2: w1=${tickNum1}, w2=${tickNum2}`);
      }
      if (Number.isNaN(tpsNum2) || tpsNum2 <= 0) {
        throw new Error(`Invalid TPS in window 2: ${w2.tps}`);
      }

      // Calculate the expected delta ticks and delta TPS for window 2
      const deltaTicksWindow2 = tickNum2 - tickNum1;
      const expectedWindow2MinTps = 80.0; // At least 80 TPS for a healthy simulated loop
      if (tpsNum2 < expectedWindow2MinTps) {
        throw new Error(
          `TPS underreported in window 2: got ${tpsNum2}, expected >= ${expectedWindow2MinTps}`,
        );
      }

      observations.camelCasePositive = {
        passed: true,
        window1: {
          ...w1,
          popNum: popNum1,
          tpsNum: tpsNum1,
          tickNum: tickNum1,
          energyNum: energyNum1,
          healthNum: healthNum1,
        },
        window2: {
          ...w2,
          popNum: popNum2,
          tpsNum: tpsNum2,
          tickNum: tickNum2,
          energyNum: energyNum2,
          healthNum: healthNum2,
        },
        deltaTicksWindow2,
        pageErrorsCount: pageErrors.length,
      };
      observations.consoleLogs = pageLogs;

      await context.close();
    }

    // =========================================================================
    // Test 2: Negative Control 1 — Snake_case Contract Rejection
    // =========================================================================
    {
      const context = await browser.newContext();
      await installPausedClock(context);
      const page = await context.newPage();

      const pageErrors = [];
      page.on("pageerror", (err) => pageErrors.push(err.message));

      await page.route("**/pkg/scriptbots_web.js", (route) =>
        route.fulfill({
          status: 200,
          contentType: "application/javascript; charset=utf-8",
          body: snakeCaseWasmModule,
        }),
      );

      await page.goto(`${baseUrl}/`);

      // Run clock past 500ms boundary so updateStats executes with snake_case fields.
      // In Playwright, clock.runFor rejects directly when an unhandled timer/RAF error occurs.
      let clockError = null;
      try {
        await context.clock.runFor(600);
      } catch (err) {
        clockError = err;
      }

      const allErrors = [...(clockError ? [clockError.message] : []), ...pageErrors];

      // Expect an unhandled TypeError because agentCount is undefined:
      // snapshot.summary.agentCount.toLocaleString() throws TypeError
      const expectedTypeError = allErrors.find(
        (err) =>
          err.includes("Cannot read properties of undefined (reading 'toLocaleString')") ||
          err.includes(
            "undefined is not an object (evaluating 'snapshot.summary.agentCount.toLocaleString')",
          ),
      );

      if (!expectedTypeError) {
        throw new Error(
          `Snake_case negative control failed to catch contract violation! Errors: ${JSON.stringify(allErrors)}`,
        );
      }

      // Verify the DOM stayed unpopulated and the presentation loop froze.
      // The independent science timer is not stopped by a paint contract error.
      const popAfterCrash = await page.textContent("#metric-population");
      if (popAfterCrash !== "–") {
        throw new Error(
          `Population was populated despite snake_case contract error: ${popAfterCrash}`,
        );
      }

      // Advancing time must NOT present more frames after the fatal paint error.
      const tickBefore = await page.textContent("#metric-tick");
      await context.clock.runFor(500);
      const tickAfter = await page.textContent("#metric-tick");
      if (tickBefore !== tickAfter) {
        throw new Error("Loop continued running after fatal contract error");
      }

      observations.snakeCaseNegative = {
        passed: true,
        caughtError: expectedTypeError,
        popRemainedPlaceholder: popAfterCrash === "–",
        loopFrozen: tickBefore === tickAfter,
      };

      await context.close();
    }

    // =========================================================================
    // Test 3: Negative Control 2 — Scheduling-Path First-Frame Defect Proof
    // =========================================================================
    {
      const context = await browser.newContext();
      await installPausedClock(context);
      const page = await context.newPage();

      await page.route("**/pkg/scriptbots_web.js", (route) =>
        route.fulfill({
          status: 200,
          contentType: "application/javascript; charset=utf-8",
          body: camelCaseWasmModule,
        }),
      );

      await page.goto(`${baseUrl}/`);

      // Sample strictly before the 500ms boundary from page init
      await context.clock.runFor(150);
      const earlyPop150 = await page.textContent("#metric-population");
      const earlyTps150 = await page.textContent("#metric-tps");
      if (earlyPop150 !== "–" || earlyTps150 !== "–") {
        throw new Error(
          `Premature stats update detected before 500ms at +150ms: pop=${earlyPop150}, tps=${earlyTps150}`,
        );
      }

      await context.clock.runFor(200);
      const earlyPop350 = await page.textContent("#metric-population");
      const earlyTps350 = await page.textContent("#metric-tps");
      if (earlyPop350 !== "–" || earlyTps350 !== "–") {
        throw new Error(
          `Premature stats update detected before 500ms at +350ms: pop=${earlyPop350}, tps=${earlyTps350}`,
        );
      }

      // Now advance across the 500ms threshold (+250ms, making +600ms total after page init)
      await context.clock.runFor(250);
      const popAfter500 = await page.textContent("#metric-population");
      const tpsAfter500 = await page.textContent("#metric-tps");

      if (popAfter500 === "–" || tpsAfter500 === "–") {
        throw new Error("Stats failed to update once 500ms boundary was reached");
      }

      observations.schedulingNegative = {
        passed: true,
        earlyChecks: [
          { at: "+150ms", pop: earlyPop150, tps: earlyTps150 },
          { at: "+350ms", pop: earlyPop350, tps: earlyTps350 },
        ],
        allSamplesRemainedPlaceholder: true,
        refreshedAfterBoundary: true,
        boundaryPop: popAfter500,
        boundaryTps: tpsAfter500,
      };

      await context.close();
    }

    // The production demo loads the exact generated package; no module replacement.
    {
      const context = await browser.newContext();
      const page = await context.newPage();
      const pageErrors = [];
      const consoleLogs = [];
      page.on("pageerror", (err) => pageErrors.push(err.message));
      page.on("console", (msg) => consoleLogs.push({ type: msg.type(), text: msg.text() }));
      const moduleUrl = `${baseUrl}/pkg/${packageInfo.moduleFile}`;
      const wasmUrl = `${baseUrl}/pkg/${packageInfo.wasmFile}`;
      const [moduleResponse, wasmResponse] = await Promise.all([
        page.waitForResponse(moduleUrl),
        page.waitForResponse(wasmUrl),
        page.goto(`${baseUrl}/`),
      ]);
      const loadedFiles = [];
      for (const [file, response] of [[packageInfo.moduleFile, moduleResponse], [packageInfo.wasmFile, wasmResponse]]) {
        const bytes = await response.body();
        const sha256 = createHash("sha256").update(bytes).digest("hex");
        if (response.status() !== 200 || sha256 !== packageInfo.files[file].sha256) {
          throw new Error(`Browser did not load the exact generated package bytes: ${file}`);
        }
        loadedFiles.push({ file, url: response.url(), status: response.status(), bytes: bytes.length, sha256 });
      }
      if (wasmResponse.headers()["content-type"] !== "application/wasm") {
        throw new Error("Packaged WASM must be served with its streaming MIME type");
      }
      const expectedVersion = `${packageInfo.manifest.name} ${packageInfo.manifest.version}`;
      await page.waitForFunction((version) => document.getElementById("version").textContent === version, expectedVersion);
      await page.waitForFunction(() => Number(document.getElementById("metric-tick").textContent.replaceAll(",", "")) > 0)
        .catch((error) => {
          console.error("Real packaged demo did not advance:", JSON.stringify({
            loadedFiles, pageErrors, consoleLogs,
          }));
          throw error;
        });
      const readMetrics = () => page.evaluate(() => Object.fromEntries(
        ["tick", "population", "tps", "energy", "health", "births", "deaths"].map((field) =>
          [field, document.getElementById(`metric-${field}`).textContent]),
      ));
      const window1 = await readMetrics();
      const firstTick = Number(window1.tick.replaceAll(",", ""));
      await page.waitForFunction((tick) => Number(document.getElementById("metric-tick").textContent.replaceAll(",", "")) > tick, firstTick);
      const window2 = await readMetrics();
      for (const window of [window1, window2]) {
        for (const value of Object.values(window)) {
          if (value.trim().length === 0 || !Number.isFinite(Number(value.replaceAll(",", "")))
              || Number(value.replaceAll(",", "")) < 0) {
            throw new Error(`Real packaged simulation produced an invalid DOM metric: ${JSON.stringify(window)}`);
          }
        }
        if (Number(window.tps) <= 0) throw new Error("Real packaged simulation did not advance its TPS window");
      }
      const api = await page.evaluate(async () => {
        const response = await fetch("/pkg/package.json", { signal: AbortSignal.timeout(30_000) });
        if (!response.ok) throw new Error("The exported package descriptor is unavailable");
        const manifest = await response.json();
        const wasm = await import(new URL(manifest.exports["."].import, response.url).href);
        await wasm.default();
        const options = { seed: 7, population: 8, seed_strategy: "none", snapshot_format: "json" };
        const steps = 2;
        const handle = wasm.init_sim(options);
        try {
          const initial = handle.snapshot();
          const advanced = handle.tick(steps);
          handle.reset(options.seed);
          const reset = handle.snapshot();
          if (initial.tick !== 0 || initial.summary.agentCount !== options.population
              || initial.agents.length !== options.population || advanced.tick !== initial.tick + steps
              || reset.tick !== initial.tick || reset.summary.agentCount !== options.population) {
            throw new Error("Packaged init/snapshot/tick/reset did not preserve their public contracts");
          }
          let unrepresentableTimeError = null;
          try {
            handle.drive(Number.MAX_SAFE_INTEGER / 1_000_000 + 1);
          } catch (error) {
            unrepresentableTimeError = String(error);
          }
          if (!unrepresentableTimeError?.includes("exact nanosecond range")
              || handle.snapshot().tick !== reset.tick) {
            throw new Error("Unrepresentable receipt time was not refused before scientific work");
          }
          const late = handle.drive(20_000);
          const afterLate = handle.snapshot();
          if (late.scientific_steps <= 0 || late.automatic_steps_skipped <= 0
              || late.automatic_steps_due !== late.scientific_steps + late.automatic_steps_skipped
              || afterLate.tick !== reset.tick + late.scientific_steps
              || handle.drive(20_000).scientific_steps !== 0) {
            throw new Error("Packaged host did not retain bounded catch-up and same-time idempotence");
          }
          let backwardsTimeError = null;
          try {
            handle.drive(19_999);
          } catch (error) {
            backwardsTimeError = String(error);
          }
          if (!backwardsTimeError?.includes("browser elapsed time moved backwards")
              || handle.snapshot().tick !== afterLate.tick
              || handle.drive(20_000).scientific_steps !== 0) {
            throw new Error("Packaged host admitted a clock rewind or changed science after refusal");
          }
          let invalidGeometryError = null;
          try {
            const invalid = wasm.init_sim({ ...options, world_width: initial.world.width + 1 });
            invalid.free();
            throw new Error("Packaged initialization admitted invalid food-grid geometry");
          } catch (error) {
            invalidGeometryError = error.message;
            if (!invalidGeometryError.includes("failed to initialize ScriptBots world state")
                || !invalidGeometryError.includes("world_width must be divisible by food_cell_size")) {
              throw error;
            }
          }
          return {
            package: { name: manifest.name, version: manifest.version, exports: manifest.exports },
            options, steps,
            initial: { tick: initial.tick, summary: initial.summary, agentCount: initial.agents.length },
            advanced: { tick: advanced.tick, summary: advanced.summary, agentCount: advanced.agents.length },
            reset: { tick: reset.tick, summary: reset.summary, agentCount: reset.agents.length },
            boundedCatchUp: { receipt: late, observedTick: afterLate.tick, unrepresentableTimeError, backwardsTimeError },
            invalidGeometryError,
          };
        } finally {
          handle.free();
        }
      });
      if (pageErrors.length !== 0 || consoleLogs.some((entry) => entry.text.includes("Failed to initialise wasm module"))) {
        throw new Error(`Real packaged demo failed: ${JSON.stringify({ pageErrors, consoleLogs })}`);
      }
      await context.close();

      // Real package and production main.js, with RAF held by the browser test.
      // No simulation/module replacement: a render-authorized tick throws.
      const clockContext = await browser.newContext();
      await installPausedClock(clockContext);
      await clockContext.addInitScript(() => {
        window.__heldFrames = [];
        window.requestAnimationFrame = (callback) => window.__heldFrames.push(callback);
      });
      const clockPage = await clockContext.newPage();
      const clockErrors = [];
      clockPage.on("pageerror", (error) => clockErrors.push(error.message));
      await clockPage.goto(`${baseUrl}/`);
      await clockPage.waitForFunction((version) => document.getElementById("version").textContent === version, expectedVersion);
      await clockPage.evaluate(async () => {
        const wasm = await import("/pkg/scriptbots_web.js");
        const frame = wasm.SimHandle.prototype.frame;
        wasm.SimHandle.prototype.frame = function () {
          window.__clockHandle = this;
          return frame.call(this);
        };
        wasm.SimHandle.prototype.tick = function () {
          throw new Error("RAF attempted to authorize scientific work");
        };
        const setSpeed = wasm.SimHandle.prototype.setSpeed;
        window.__clockSpeedCommands = [];
        wasm.SimHandle.prototype.setSpeed = function (speed) {
          const status = setSpeed.call(this, speed);
          window.__clockSpeedCommands.push({ now: performance.now(), speed, status });
          return status;
        };
        const drive = wasm.SimHandle.prototype.drive;
        window.__clockDrives = [];
        wasm.SimHandle.prototype.drive = function (now) {
          const receipt = drive.call(this, now);
          window.__clockDrives.push({ now, receipt });
          if (window.__clockDrives.length > 8) window.__clockDrives.shift();
          return receipt;
        };
      });
      await clockContext.clock.runFor(600);
      const clockBefore = await clockPage.evaluate(() => {
        if (window.__heldFrames.length !== 1) throw new Error("Expected one outstanding presentation callback");
        window.__heldFrames.shift()(performance.now());
        return window.__clockHandle.snapshot().tick;
      });
      if (clockBefore <= 0) throw new Error("Science did not advance while RAF was held");
      const firstPresentedTps = Number(await clockPage.textContent("#metric-tps"));
      if (!Number.isFinite(firstPresentedTps) || firstPresentedTps <= 0) {
        throw new Error("First delayed presentation omitted the already observed science interval from TPS");
      }
      await clockContext.clock.runFor(300);
      const clockAfter = await clockPage.evaluate(() => {
        const before = window.__clockHandle.snapshot().tick;
        for (let index = 0; index < 120; index += 1) {
          window.__heldFrames.shift()(performance.now());
        }
        const after = window.__clockHandle.snapshot().tick;
        if (after !== before) throw new Error("Extra presentation callbacks advanced science");
        const speed = document.getElementById("steps-per-frame");
        speed.value = "0";
        speed.dispatchEvent(new Event("input", { bubbles: true }));
        return { before, after, pausedTick: window.__clockHandle.snapshot().tick };
      });
      if (clockAfter.before <= clockBefore) throw new Error("Independent timer stopped with RAF held");
      await clockContext.clock.runFor(300);
      const pausedTick = await clockPage.evaluate(() => window.__clockHandle.snapshot().tick);
      if (pausedTick !== clockAfter.pausedTick) throw new Error("Zero speed did not pause science");
      await clockPage.evaluate(() => {
        const speed = document.getElementById("steps-per-frame");
        speed.value = "2";
        speed.dispatchEvent(new Event("input", { bubbles: true }));
        Object.defineProperty(document, "hidden", { configurable: true, get: () => true });
        document.dispatchEvent(new Event("visibilitychange"));
      });
      await clockContext.clock.runFor(5_000);
      const hiddenTick = await clockPage.evaluate(() => window.__clockHandle.snapshot().tick);
      if (hiddenTick !== pausedTick) throw new Error("Hidden page accrued scientific work");
      await clockPage.evaluate(() => {
        Object.defineProperty(document, "hidden", { configurable: true, get: () => false });
        document.dispatchEvent(new Event("visibilitychange"));
      });
      await clockContext.clock.runFor(100);
      const resumedTick = await clockPage.evaluate(() => window.__clockHandle.snapshot().tick);
      if (resumedTick <= hiddenTick || clockErrors.length !== 0) {
        const clockState = await clockPage.evaluate(() => ({
          now: performance.now(),
          hidden: document.hidden,
          speed: document.getElementById("steps-per-frame").value,
          commands: window.__clockSpeedCommands,
          drives: window.__clockDrives,
          log: document.getElementById("log").textContent,
        }));
        throw new Error(`Independent browser clock did not resume cleanly: ${JSON.stringify({ hiddenTick, resumedTick, clockErrors, clockState })}`);
      }
      await clockContext.close();

      // The same actual glue must refuse an invalid WASM version, rather than
      // allowing the DOM fixture to stand in for a compiled module.
      const corruptContext = await browser.newContext();
      const corruptPage = await corruptContext.newPage();
      const corruptConsole = [];
      let corruptRequests = 0;
      corruptPage.on("console", (msg) => corruptConsole.push({ type: msg.type(), text: msg.text() }));
      await corruptPage.route(wasmUrl, (route) => {
        corruptRequests += 1;
        return route.fulfill({ status: 200, contentType: "application/wasm", body: Buffer.from([0, 97, 115, 109, 255, 255, 255, 255]) });
      });
      await corruptPage.goto(`${baseUrl}/`);
      await corruptPage.waitForFunction(() => document.getElementById("log").textContent.includes("Bootstrap failed:"));
      const corruptDom = await corruptPage.evaluate(() => ({
        version: document.getElementById("version").textContent,
        tick: document.getElementById("metric-tick").textContent,
        population: document.getElementById("metric-population").textContent,
        log: document.getElementById("log").textContent,
      }));
      if (corruptRequests !== 1 || corruptDom.version !== "initializing…" || corruptDom.tick !== "–"
          || corruptDom.population !== "–" || !corruptConsole.some((entry) =>
            entry.type === "error" && entry.text.includes("Failed to initialise wasm module") && entry.text.includes("CompileError"))) {
        throw new Error(`Corrupt packaged WASM did not fail at bootstrap: ${JSON.stringify({ corruptRequests, corruptDom, corruptConsole })}`);
      }
      await corruptContext.close();
      observations.realPackage = {
        passed: true,
        directory: packageDir,
        name: packageInfo.manifest.name,
        version: packageInfo.manifest.version,
        exports: packageInfo.manifest.exports,
        types: packageInfo.typesFile,
        files: packageInfo.files,
        loadedFiles,
        window1, window2, api, pageErrors, consoleLogs,
        independentClock: {
          passed: true, clockBefore, firstPresentedTps, clockAfter, pausedTick, hiddenTick, resumedTick,
          heldRaf: true, extraPresentations: 120, clockErrors,
          visibilityControl: "visibilitychange with injected document.hidden; not an OS suspension proof",
        },
        corruptedWasmNegative: { passed: true, requests: corruptRequests, dom: corruptDom, consoleLogs: corruptConsole },
      };
    }

    // All tests passed successfully!
    const caseNames = ["camelCasePositive", "snakeCaseNegative", "schedulingNegative", "realPackage"];
    if (!caseNames.every((name) => observations[name]?.passed === true)) {
      throw new Error("A required browser case has no successful observation");
    }
    const result = {
      schema: "scriptbots.browser-stats-dom.v1",
      timestamp: new Date().toISOString(),
      status: "pass",
      browser: { version: browser.version(), executable: process.env.SCRIPTBOTS_CHROMIUM_EXECUTABLE || null },
      cases_passed: caseNames.length,
      cases_failed: 0,
      observations,
    };

    console.log(JSON.stringify(result));
    return true;
  } finally {
    await browser.close();
    server.close();
  }
}

runTests().catch((err) => {
  console.error("FATAL: Test run failed:", err);
  process.exit(1);
});
