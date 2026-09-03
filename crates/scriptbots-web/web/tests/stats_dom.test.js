// crates/scriptbots-web/web/tests/stats_dom.test.js
// Real-browser DOM execution test for the WASM browser stats refresh (bd-2z0.12.6).

import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const webDir = path.resolve(__dirname, "..");

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
function createServer() {
  return http.createServer((req, res) => {
    const url = new URL(req.url, "http://127.0.0.1");

    if (url.pathname === "/" || url.pathname === "/index.html") {
      res.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
      res.end(fs.readFileSync(path.join(webDir, "index.html")));
    } else if (url.pathname === "/main.js") {
      res.writeHead(200, { "Content-Type": "application/javascript; charset=utf-8" });
      res.end(fs.readFileSync(path.join(webDir, "main.js")));
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
    consoleLogs: [],
    pageErrors: [],
  };

  const server = createServer();
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const port = server.address().port;
  const baseUrl = `http://127.0.0.1:${port}`;

  const browser = await chromium.launch({ headless: true });

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
        })
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
          `Scheduling defect: stats updated before 500ms boundary: pop=${earlyPop}, tps=${earlyTps}`
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
        throw new Error(`Window 1 failed to refresh DOM metrics after 550ms: ${JSON.stringify(w1)}`);
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
        throw new Error(`Stats refreshed prematurely at t=800ms: expected ${w1.pop}, got ${interPop}`);
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
          `TPS underreported in window 2: got ${tpsNum2}, expected >= ${expectedWindow2MinTps}`
        );
      }

      observations.camelCasePositive = {
        passed: true,
        window1: { ...w1, popNum: popNum1, tpsNum: tpsNum1, tickNum: tickNum1, energyNum: energyNum1, healthNum: healthNum1 },
        window2: { ...w2, popNum: popNum2, tpsNum: tpsNum2, tickNum: tickNum2, energyNum: energyNum2, healthNum: healthNum2 },
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
        })
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

      const allErrors = [
        ...(clockError ? [clockError.message] : []),
        ...pageErrors,
      ];

      // Expect an unhandled TypeError because agentCount is undefined:
      // snapshot.summary.agentCount.toLocaleString() throws TypeError
      const hasExpectedTypeError = allErrors.some((err) =>
        err.includes("Cannot read properties of undefined (reading 'toLocaleString')") ||
        err.includes("undefined is not an object (evaluating 'snapshot.summary.agentCount.toLocaleString')")
      );

      if (!hasExpectedTypeError) {
        throw new Error(
          `Snake_case negative control failed to catch contract violation! Errors: ${JSON.stringify(allErrors)}`
        );
      }

      // Verify DOM element was never populated with a valid number and loop froze
      const popAfterCrash = await page.textContent("#metric-population");
      if (popAfterCrash !== "–") {
        throw new Error(
          `Population was populated despite snake_case contract error: ${popAfterCrash}`
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
        })
      );

      await page.goto(`${baseUrl}/`);

      // Sample strictly before the 500ms boundary from page init
      await context.clock.runFor(150);
      const earlyPop150 = await page.textContent("#metric-population");
      const earlyTps150 = await page.textContent("#metric-tps");
      if (earlyPop150 !== "–" || earlyTps150 !== "–") {
        throw new Error(
          `Premature stats update detected before 500ms at +150ms: pop=${earlyPop150}, tps=${earlyTps150}`
        );
      }

      await context.clock.runFor(200);
      const earlyPop350 = await page.textContent("#metric-population");
      const earlyTps350 = await page.textContent("#metric-tps");
      if (earlyPop350 !== "–" || earlyTps350 !== "–") {
        throw new Error(
          `Premature stats update detected before 500ms at +350ms: pop=${earlyPop350}, tps=${earlyTps350}`
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

    // All tests passed successfully!
    const result = {
      schema: "scriptbots.browser-stats-dom.v1",
      timestamp: new Date().toISOString(),
      status: "pass",
      cases_passed: 3,
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
