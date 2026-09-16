// crates/scriptbots-web/web/tests/permalink_fork_dom.test.js
// Real-browser Playwright test for permalink loading, fork-this-world UX,
// parent-diff display, and honest build mismatch banner (bd-16g.8.2).

import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const webDir = path.resolve(__dirname, "..");

const mockWasmModule = `
  export default async function initWasm() { return {}; }
  export function version() { return "0.1.0-permalink-fork-test"; }
  export function default_init_options() { return { world_width: 1280, world_height: 720 }; }
  
  let currentWorldLink = "sbw1.root_world_link";

  export function build_identity() {
    return {
      toolchain_digest: "0x1111111111111111",
      lockfile_digest: "0x2222222222222222",
      core_digest: "0x3333333333333333"
    };
  }

  export function check_build_match(link) {
    if (link.includes("mismatch")) {
      return {
        status: "mismatch",
        link_toolchain_digest: "0xaaaaaaaaaaaaaaaa",
        link_lockfile_digest: "0xbbbbbbbbbbbbbbbb",
        link_core_digest: "0xcccccccccccccccc",
        local_toolchain_digest: "0x1111111111111111",
        local_lockfile_digest: "0x2222222222222222",
        local_core_digest: "0x3333333333333333"
      };
    }
    if (link.includes("compatible")) {
      return {
        status: "compatible",
        link_toolchain_digest: "0xaaaaaaaaaaaaaaaa",
        link_lockfile_digest: "0xbbbbbbbbbbbbbbbb",
        link_core_digest: "0x3333333333333333",
        local_toolchain_digest: "0x1111111111111111",
        local_lockfile_digest: "0x2222222222222222",
        local_core_digest: "0x3333333333333333"
      };
    }
    return {
      status: "exact",
      link_toolchain_digest: "0x1111111111111111",
      link_lockfile_digest: "0x2222222222222222",
      link_core_digest: "0x3333333333333333",
      local_toolchain_digest: "0x1111111111111111",
      local_lockfile_digest: "0x2222222222222222",
      local_core_digest: "0x3333333333333333"
    };
  }

  export function permalink_diff(link) {
    if (link.includes("with_diff")) {
      return [
        { knob: "food_growth_rate", parent_value: 0.01, this_value: 0.05 },
        { knob: "population_minimum", parent_value: 24, this_value: 48 }
      ];
    }
    return [];
  }

  export function permalink_of(handle) {
    return handle.__link || currentWorldLink;
  }

  export function fork(handle, patch) {
    return "sbw1.child_fork_link_with_diff";
  }

  export function init_from_permalink(link) {
    if (link.startsWith("invalid") || link.includes("corrupt")) {
      throw new Error("permalink magic mismatch (first bytes: " + link.slice(0, 8) + ")");
    }
    let tick = 0;
    return {
      __link: link,
      permalinkOf() { return link; },
      tick(steps) {
        tick += steps;
        return {
          tick: tick,
          world: { width: 1280, height: 720, closed: true },
          summary: {
            agentCount: 150,
            births: 5,
            deaths: 2,
            totalEnergy: 300,
            averageEnergy: 2.0,
            averageHealth: 0.95
          },
          agents: [
            {
              position: [200.0, 200.0],
              health: 0.95,
              color: [0.1, 0.7, 0.3],
              boost: false
            }
          ]
        };
      }
    };
  }

  export function init_sim(opts) {
    let tick = 0;
    return {
      __link: currentWorldLink,
      permalinkOf() { return currentWorldLink; },
      tick(steps) {
        tick += steps;
        return {
          tick: tick,
          world: { width: 1280, height: 720, closed: true },
          summary: {
            agentCount: 200,
            births: 10,
            deaths: 4,
            totalEnergy: 400,
            averageEnergy: 2.0,
            averageHealth: 0.9
          },
          agents: []
        };
      }
    };
  }
`;

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
  const server = createServer();
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const port = server.address().port;
  const baseUrl = `http://127.0.0.1:${port}`;

  const browser = await chromium.launch({ headless: true });

  try {
    // =========================================================================
    // Test 1: Honest Build Mismatch Warning Banner & Simulation Execution
    // =========================================================================
    {
      const context = await browser.newContext();
      await context.clock.install();
      const page = await context.newPage();

      await page.route("**/pkg/scriptbots_web.js", (route) =>
        route.fulfill({
          status: 200,
          contentType: "application/javascript; charset=utf-8",
          body: mockWasmModule,
        })
      );

      // Load with mismatch permalink
      await page.goto(`${baseUrl}/?world=sbw1.mismatch_sample`);

      // Verify the banner is rendered in the DOM with mismatch warning
      const bannerDisplay = await page.$eval("#mismatch-banner", (el) => window.getComputedStyle(el).display);
      if (bannerDisplay === "none") {
        throw new Error("Expected #mismatch-banner to be visible on mismatch, but display was none");
      }

      const bannerText = await page.textContent("#mismatch-banner");
      if (!bannerText.includes("Build Mismatch Warning")) {
        throw new Error(`Expected mismatch warning title in banner text, got: ${bannerText}`);
      }
      if (!bannerText.includes("0xcccccccccccccccc") || !bannerText.includes("0x3333333333333333")) {
        throw new Error(`Expected core digests in banner details, got: ${bannerText}`);
      }

      // Assert that the simulation is NOT blocked: advance clock past 500ms and verify tick updates
      await context.clock.runFor(600);
      const tickText = await page.textContent("#metric-tick");
      if (tickText === "–" || tickText === "0") {
        throw new Error(`Simulation should still run despite mismatch, got tick: ${tickText}`);
      }

      await context.close();
    }

    // =========================================================================
    // Test 2: Parent Diff Display
    // =========================================================================
    {
      const context = await browser.newContext();
      const page = await context.newPage();

      await page.route("**/pkg/scriptbots_web.js", (route) =>
        route.fulfill({
          status: 200,
          contentType: "application/javascript; charset=utf-8",
          body: mockWasmModule,
        })
      );

      // Load with permalink that has diffs
      await page.goto(`${baseUrl}/?world=sbw1.with_diff_test`);

      const tableDisplay = await page.$eval("#diff-table", (el) => window.getComputedStyle(el).display);
      if (tableDisplay === "none") {
        throw new Error("Expected #diff-table to be visible for permalink with diffs");
      }

      const rowCount = await page.$$eval("#diff-tbody tr", (trs) => trs.length);
      if (rowCount !== 2) {
        throw new Error(`Expected exactly 2 diff rows, found: ${rowCount}`);
      }

      const row1Text = await page.textContent("#diff-tbody tr:nth-child(1)");
      if (!row1Text.includes("food_growth_rate") || !row1Text.includes("0.05")) {
        throw new Error(`Row 1 mismatch: ${row1Text}`);
      }

      await context.close();
    }

    // =========================================================================
    // Test 3: Fork Flow Produces Child Link Without Mutating Running World
    // =========================================================================
    {
      const context = await browser.newContext();
      const page = await context.newPage();

      await page.route("**/pkg/scriptbots_web.js", (route) =>
        route.fulfill({
          status: 200,
          contentType: "application/javascript; charset=utf-8",
          body: mockWasmModule,
        })
      );

      await page.goto(`${baseUrl}/?world=sbw1.root_world_link`);

      // Input patch and trigger fork
      await page.fill("#fork-patch-input", '{"food_growth_rate": 0.05}');
      await page.click("#fork-btn");

      // Verify fork-output displays child link
      const outputDisplay = await page.$eval("#fork-output", (el) => window.getComputedStyle(el).display);
      if (outputDisplay === "none") {
        throw new Error("Expected #fork-output to be visible after fork");
      }

      const childLink = await page.textContent("#fork-child-link");
      if (!childLink.includes("sbw1.child_fork_link_with_diff")) {
        throw new Error(`Unexpected child link: ${childLink}`);
      }

      // Verify running world permalink input is unchanged
      const currentInput = await page.$eval("#permalink-input", (el) => el.value);
      if (currentInput !== "sbw1.root_world_link") {
        throw new Error(`Running world was mutated! Expected sbw1.root_world_link, got: ${currentInput}`);
      }

      await context.close();
    }

    // =========================================================================
    // Test 4: Negative Control: Rejected Bad Permalink Leaves Instance Functional
    // =========================================================================
    {
      const context = await browser.newContext();
      const page = await context.newPage();
      const pageErrors = [];
      page.on("pageerror", (err) => pageErrors.push(err.message));

      await page.route("**/pkg/scriptbots_web.js", (route) =>
        route.fulfill({
          status: 200,
          contentType: "application/javascript; charset=utf-8",
          body: mockWasmModule,
        })
      );

      await page.goto(`${baseUrl}/`);

      // Attempt to load an invalid permalink
      await page.fill("#permalink-input", "invalid_corrupted_link");
      await page.click("#load-permalink-btn");

      // Verify error in status
      const statusText = await page.textContent("#permalink-status");
      if (!statusText.toLowerCase().includes("error")) {
        throw new Error(`Expected error status, got: ${statusText}`);
      }

      // Assert no unhandled JS page crashes
      if (pageErrors.length > 0) {
        throw new Error(`Unhandled page errors during bad link: ${pageErrors.join("; ")}`);
      }

      // Assert simulation can reset normally and runs without poison
      await page.click("#reset-btn");
      const postResetStatus = await page.textContent("#permalink-status");
      if (postResetStatus.toLowerCase().includes("error")) {
        throw new Error(`Instance remained poisoned after reset: ${postResetStatus}`);
      }

      await context.close();
    }

    const result = {
      schema: "scriptbots.browser-permalink-fork-dom.v1",
      timestamp: new Date().toISOString(),
      status: "pass",
      cases_passed: 4,
      cases_failed: 0,
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
