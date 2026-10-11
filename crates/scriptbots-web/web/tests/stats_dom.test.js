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
    return {
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
    return {
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
  });

  try {
    // =========================================================================
    // Test 1: Production camelCase Contract & Multiple Window TPS Deltas
    // =========================================================================
    {
      const context = await browser.newContext();
      await context.clock.install();
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
      await context.clock.install();
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
      const hasExpectedTypeError = allErrors.some(
        (err) =>
          err.includes("Cannot read properties of undefined (reading 'toLocaleString')") ||
          err.includes(
            "undefined is not an object (evaluating 'snapshot.summary.agentCount.toLocaleString')",
          ),
      );

      if (!hasExpectedTypeError) {
        throw new Error(
          `Snake_case negative control failed to catch contract violation! Errors: ${JSON.stringify(allErrors)}`,
        );
      }

      // Verify DOM element was never populated with a valid number and loop froze
      const popAfterCrash = await page.textContent("#metric-population");
      if (popAfterCrash !== "–") {
        throw new Error(
          `Population was populated despite snake_case contract error: ${popAfterCrash}`,
        );
      }

      // Advancing clock further must NOT execute more frames because the loop halted
      const tickBefore = await page.textContent("#metric-tick");
      await context.clock.runFor(500);
      const tickAfter = await page.textContent("#metric-tick");
      if (tickBefore !== tickAfter) {
        throw new Error("Loop continued running after fatal contract error");
      }

      observations.snakeCaseNegative = {
        passed: true,
        caughtError: pageErrors[0],
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
      await context.clock.install();
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
          return {
            package: { name: manifest.name, version: manifest.version, exports: manifest.exports },
            options, steps,
            initial: { tick: initial.tick, summary: initial.summary, agentCount: initial.agents.length },
            advanced: { tick: advanced.tick, summary: advanced.summary, agentCount: advanced.agents.length },
            reset: { tick: reset.tick, summary: reset.summary, agentCount: reset.agents.length },
          };
        } finally {
          handle.free();
        }
      });
      if (pageErrors.length !== 0 || consoleLogs.some((entry) => entry.text.includes("Failed to initialise wasm module"))) {
        throw new Error(`Real packaged demo failed: ${JSON.stringify({ pageErrors, consoleLogs })}`);
      }
      await context.close();

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
