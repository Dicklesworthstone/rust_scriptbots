// crates/scriptbots-web/web/tests/permalink_fork_dom.test.js
// Real-browser Playwright test for permalink loading, fork-this-world UX,
// parent-diff display, honest build mismatch banner, local owner replacement,
// and presentation recovery through the production page and Canvas renderer.
// The WASM double exercises browser lifecycle behavior, not checkpoint codec proof.

import fs from "node:fs";
import http from "node:http";
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
  
  const currentWorldLink = "sbw1.root_world_link";
  const owners = window.__ownerTest = { records: [], nextFailure: null, failDriveId: null, frameFailure: null, forkCalls: 0 };
  const timers = window.__ownerTimers = { pending: new Set(), maximum: 0 };
  const frames = window.__ownerFrames = { pending: new Set(), maximum: 0, callbacks: 0 };
  const schedule = window.setTimeout.bind(window);
  const cancel = window.clearTimeout.bind(window);
  window.setTimeout = (callback, delay, ...args) => {
    if (callback?.name !== "pumpSimulation") return schedule(callback, delay, ...args);
    const id = schedule(() => { timers.pending.delete(id); callback(...args); }, delay);
    timers.pending.add(id);
    timers.maximum = Math.max(timers.maximum, timers.pending.size);
    return id;
  };
  window.clearTimeout = (id) => { timers.pending.delete(id); cancel(id); };
  const scheduleFrame = window.requestAnimationFrame.bind(window);
  const cancelFrame = window.cancelAnimationFrame.bind(window);
  window.requestAnimationFrame = (callback) => {
    if (callback?.name !== "presentSimulation") return scheduleFrame(callback);
    const id = scheduleFrame((now) => {
      frames.pending.delete(id);
      frames.callbacks += 1;
      callback(now);
    });
    frames.pending.add(id);
    frames.maximum = Math.max(frames.maximum, frames.pending.size);
    return id;
  };
  window.cancelAnimationFrame = (id) => { frames.pending.delete(id); cancelFrame(id); };

  export function build_identity() {
    return {
      toolchain_digest: "0x1111111111111111",
      lockfile_digest: "0x2222222222222222",
      core_digest: "0x3333333333333333"
    };
  }

  export function check_build_match(link) {
    if (link.startsWith("sbb1.")) {
      const status = link.includes("unavailable") ? "unavailable" : link.includes("mismatch") ? "mismatch" : "recorded_match";
      return {
        status, identity_scope: "recorded_browser_build_inputs",
        evidence: "<em>Recorded recipe fixture; source input fingerprints only.</em>",
        link_toolchain_digest: status === "unavailable" ? "unavailable" : "0x1111111111111111",
        link_lockfile_digest: "0x2222222222222222",
        link_core_digest: status === "mismatch" ? "0xcccccccccccccccc" : "0x3333333333333333",
        local_toolchain_digest: "0x1111111111111111",
        local_lockfile_digest: "0x2222222222222222",
        local_core_digest: "0x3333333333333333"
      };
    }
    if (link.includes("mismatch")) {
      return {
        status: "mismatch",
        link_toolchain_digest: link.includes("markup") ? "<img src=x onerror=window.__bannerExecuted=true>" : "0xaaaaaaaaaaaaaaaa",
        link_lockfile_digest: "0xbbbbbbbbbbbbbbbb",
        link_core_digest: link.includes("markup") ? "<img src=x onerror=window.__bannerExecuted=true>" : "0xcccccccccccccccc",
        local_toolchain_digest: "0x1111111111111111",
        local_lockfile_digest: "0x2222222222222222",
        local_core_digest: "0x3333333333333333"
      };
    }
    if (link.includes("compatible")) {
      return {
        status: "compatible",
        link_toolchain_digest: link.includes("markup") ? "<img src=x onerror=window.__bannerExecuted=true>" : "0xaaaaaaaaaaaaaaaa",
        link_lockfile_digest: "0xbbbbbbbbbbbbbbbb",
        link_core_digest: "0x3333333333333333",
        local_toolchain_digest: "0x1111111111111111",
        local_lockfile_digest: "0x2222222222222222",
        local_core_digest: "0x3333333333333333"
      };
    }
    return {
      status: "exact",
      identity_scope: "legacy_gallery_reference",
      evidence: "Gallery fixture identifiers are not measured compiler or source identity.",
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
    return handle.permalinkOf();
  }

  export function fork(handle, patch) {
    owners.forkCalls += 1;
    return handle.launchInfo().kind === "browser_launch" ? "sbb1.child_recipe_with_diff" : "sbw1.child_fork_link_with_diff";
  }

  export function permalink_info(link) {
    return link.startsWith("sbb1.") ? {
      kind: "browser_launch", semantics: "initial_conditions_restart", shareable: true,
      seed: "18446744073709551615", initial_population: 151, brain_preset: "dwraon", recipe_version: 1, reason: null
    } : {
      kind: "gallery_reference", semantics: "gallery_configuration_restart", shareable: true,
      seed: "1", initial_population: null, brain_preset: null, recipe_version: null,
      reason: "No browser founder recipe is encoded in this gallery reference."
    };
  }

  export function init_from_permalink(link) {
    if (link.startsWith("invalid") || link.includes("corrupt")) {
      throw new Error("permalink magic mismatch (first bytes: " + link.slice(0, 8) + ")");
    }
    return createHandle("permalink", link, 0, link.startsWith("sbb1.") ? "dwraon" : "none");
  }

  export function init_sim(opts) {
    const preset = opts.default_brain || (opts.seed_strategy === "wander" ? "wander" : "none");
    return createHandle("reset", currentWorldLink, 0, preset, opts.seed, opts.population);
  }

  export function checkpoint_byte_limit() { return 1024; }
  export function restore_checkpoint(bytes) {
    if (![1, 9].includes(bytes.length) || bytes[0] !== 42) throw new Error("intentional checkpoint refusal");
    const tick = bytes.length === 1 ? 37 : Number(new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).getBigUint64(1, true));
    return createHandle("checkpoint", "sbw1.continuation", tick);
  }
  export function decode_remote_frame_binary() { throw new Error("No remote frame in this lifecycle fixture"); }

  function createHandle(kind, link, initialTick = 0, preset = "mlp", seed = 1, population = 200) {
    const failure = owners.nextFailure;
    owners.nextFailure = null;
    if (failure === "create") throw new Error("intentional create failure");
    const record = { id: owners.records.length + 1, kind, preset, seed: String(seed), population, tick: initialTick,
      freeCalls: 0, driveCalls: 0, frameCalls: 0, narrativeReads: 0, permalinkCalls: 0, elapsed: 0, speed: 1 };
    owners.records.push(record);
    let credit = 0;
    function check(stage) {
      if (record.freeCalls !== 0) throw new Error("owner used after free: " + record.id);
      if (failure === stage) throw new Error("intentional " + stage + " failure");
    }
    return {
      free() {
        record.freeCalls += 1;
        if (record.freeCalls !== 1) throw new Error("owner freed twice: " + record.id);
      },
      permalinkOf() { check("permalink"); record.permalinkCalls += 1; return link; },
      brainPreset() { check("preset"); return preset; },
      launchInfo: failure === "launch_missing" ? undefined : function () {
        check("launch");
        if (failure === "launch_schema") return { kind: "browser_launch", semantics: "saved_owner_continuation", shareable: true };
        if (failure === "launch_seed") return { ...permalink_info("sbb1.fixture"), seed: "18446744073709551616" };
        if (kind === "checkpoint") return {
          kind: "checkpoint_continuation", semantics: "saved_owner_continuation", shareable: false,
          seed: null, initial_population: null, brain_preset: null, recipe_version: null,
          reason: "Use a saved-owner continuation; the original launch recipe is unavailable."
        };
        if (kind === "permalink") return permalink_info(link);
        return { kind: "browser_launch", semantics: "initial_conditions_restart", shareable: failure !== "launch_unshareable",
          seed: String(seed), initial_population: population, brain_preset: preset, recipe_version: 1,
          reason: failure === "launch_unshareable" ? "This owner recipe exceeds the link size limit." : null };
      },
      narrativePage(afterEvent, expectedSessionId) {
        check("narrative");
        record.narrativeReads += 1;
        if (expectedSessionId !== String(2000 + record.id) || !["0", "1"].includes(afterEvent)) {
          throw new Error("narrative fixture refused a foreign session or cursor");
        }
        return { schema: "scriptbots.remote-frame.v2", kind: "events", identity: {
          sessionId: expectedSessionId, publicationRevision: String(record.tick + 400),
          scientificRevision: String(record.tick + 100), controlRevision: "20", configRevision: "10", tick: String(record.tick)
        }, playback: { paused: record.speed === 0, speedMultiplier: record.speed }, coalescedPublications: "0", environment: null,
          payload: { afterEvent, earliestRetained: "1", latestEvent: "1", gap: null,
            events: afterEvent === "1" ? [] : [{ sequence: "1", record: { schemaVersion: 1, tick: "0",
              kind: "extinction", humanText: "<em>owner " + record.id + " narrative fixture</em>", metric: "population",
              severity: 1, magnitude: -1, window: ["0", "0"], before: 1, after: 0, score: 1, subject: null } }] } };
      },
      checkpoint() {
        check("checkpoint");
        const bytes = new Uint8Array(9);
        bytes[0] = 42;
        new DataView(bytes.buffer).setBigUint64(1, BigInt(record.tick), true);
        return bytes;
      },
      setSpeed(value) { check("speed"); record.speed = value; },
      drive(now) {
        check("drive");
        record.driveCalls += 1;
        if (owners.failDriveId === record.id) throw new Error("intentional clock failure");
        if (now < record.elapsed) throw new Error("owner clock rewound");
        credit += (now - record.elapsed) * record.speed * 60 / 1000;
        record.elapsed = now;
        const steps = Math.floor(credit);
        credit -= steps;
        record.tick += steps;
        return { scientific_steps: steps, automatic_steps_skipped: 0 };
      },
      frame() {
        record.frameCalls += 1;
        check("frame");
        if (owners.frameFailure?.ownerId === record.id && owners.frameFailure.remaining !== 0) {
          if (owners.frameFailure.remaining > 0) owners.frameFailure.remaining -= 1;
          throw new Error("intentional presentation frame failure " + record.frameCalls);
        }
        const snapshot = this.snapshot();
        return { schema: "scriptbots.browser-frame.v1", snapshot, identity: {
          sessionId: String(2000 + record.id), publicationRevision: String(snapshot.tick + 400),
          scientificRevision: String(snapshot.tick + 100), controlRevision: "20",
          configRevision: "10", tick: String(snapshot.tick + (failure === "identity" ? 1 : 0))
        } };
      },
      tick(steps) {
        check("tick");
        record.tick += steps;
        return this.snapshot();
      },
      snapshot() {
        check("snapshot");
        const fromPermalink = kind === "permalink";
        return {
          tick: record.tick,
          epoch: "0",
          world: { width: 1280, height: 720, closed: true },
          summary: {
            agentCount: fromPermalink ? 150 : 200,
            births: fromPermalink ? 5 : 10,
            deaths: fromPermalink ? 2 : 4,
            totalEnergy: fromPermalink ? 300 : 400,
            averageEnergy: 2.0,
            averageHealth: fromPermalink ? 0.95 : 0.9
          },
          agents: fromPermalink ? [{ position: [200.0, 200.0], health: 0.95,
            color: [0.1, 0.7, 0.3], boost: false }] : []
        };
      }
    };
  }
`;

async function runOwnerLifecycleCases(page, context) {
  const observations = { refusals: [], recoveries: [], replacements: [], presentationRecoveries: [] };
  await page.evaluate(() => {
    const read = File.prototype.arrayBuffer;
    File.prototype.arrayBuffer = function () {
      window.__ownerFileRead = read.call(this);
      return window.__ownerFileRead;
    };
  });
  const state = () => page.evaluate(() => ({
    records: window.__ownerTest.records,
    session: document.getElementById("renderer-status").dataset.sessionId,
    pendingTimers: window.__ownerTimers.pending.size,
    maximumTimers: window.__ownerTimers.maximum,
    pendingFrames: window.__ownerFrames.pending.size,
    maximumFrames: window.__ownerFrames.maximum,
    frameCallbacks: window.__ownerFrames.callbacks,
    presentedTick: document.getElementById("renderer-status").dataset.snapshotTick,
    presentationState: document.getElementById("renderer-status").dataset.presentationState,
    rendererStatus: document.getElementById("renderer-status").textContent,
    speed: document.getElementById("steps-per-frame").value,
    preset: document.getElementById("brain-preset").value,
  }));
  const activeOwner = (value) => {
    const live = value.records.filter((owner) => owner.freeCalls === 0);
    if (live.length !== 1 || value.records.some((owner) => owner.freeCalls > 1)
        || value.session !== String(2000 + live[0].id)
        || value.pendingFrames !== 1 || value.maximumFrames !== 1) {
      throw new Error(`Expected one live, presented owner: ${JSON.stringify(value)}`);
    }
    return live[0];
  };
  const setSpeed = (speed) => page.evaluate((value) => {
    const input = document.getElementById("steps-per-frame");
    input.value = String(value);
    input.dispatchEvent(new Event("input", { bubbles: true }));
  }, speed);
  const checkpointFile = { name: "lifecycle.sbcheckpoint", mimeType: "application/octet-stream", buffer: Buffer.from([42]) };
  const replace = async (operation) => {
    if (operation === "checkpoint") {
      await page.evaluate(() => { window.__ownerFileRead = null; });
      await page.setInputFiles("#checkpoint-load", checkpointFile);
      await page.evaluate(async () => {
        if (!window.__ownerFileRead) throw new Error("Checkpoint import did not read its file");
        await window.__ownerFileRead;
        if (document.getElementById("checkpoint-load").disabled) throw new Error("Checkpoint import did not finish");
      });
    } else {
      await page.evaluate((kind) => {
        if (kind === "permalink") {
          document.getElementById("permalink-input").value = "sbw1.replacement";
          document.getElementById("load-permalink-btn").click();
        } else {
          document.getElementById("reset-btn").click();
        }
      }, operation);
    }
    await context.clock.runFor(32);
  };

  observations.brainFamilies = [];
  const defaultFamily = activeOwner(await state()).preset;
  if (defaultFamily !== "mlp") throw new Error("Browser did not pass its default real MLP family to the owner");
  for (const preset of ["dwraon", "assembly", "wander", "none", "mlp"]) {
    await page.selectOption("#brain-preset", preset);
    await replace("reset");
    const current = activeOwner(await state());
    const disabled = await page.evaluate(() => ["checkpoint-save", "checkpoint-browser-save"].map((id) => document.getElementById(id).disabled));
    if (current.preset !== preset || disabled.some((value) => value !== (preset === "wander"))) {
      throw new Error(`Family selection or continuation capability is incorrect: ${JSON.stringify({ preset, current, disabled })}`);
    }
    observations.brainFamilies.push({ preset, owner: current.id, saveDisabled: disabled });
  }

  // Keep reading the active owner after presentation faults, without flooding
  // the UI/log or creating a second scientific timer or RAF chain.
  await setSpeed(1);
  await context.clock.runFor(32);
  const beforeFrameFailure = await state();
  const frameOwner = activeOwner(beforeFrameFailure);
  await page.evaluate((ownerId) => {
    window.__presentationReports = 0;
    const status = document.getElementById("renderer-status");
    window.__presentationObserver = new MutationObserver(() => {
      if (status.textContent.startsWith("Presentation interrupted:")) window.__presentationReports += 1;
    });
    window.__presentationObserver.observe(status, { childList: true, characterData: true, subtree: true });
    window.__ownerTest.frameFailure = { ownerId, remaining: -1 };
  }, frameOwner.id);
  await context.clock.runFor(160);
  const failingFrames = await state();
  const reporting = await page.evaluate(() => ({
    updates: window.__presentationReports,
    logs: document.getElementById("log").textContent.split("Presentation interrupted:").length - 1,
  }));
  if (activeOwner(failingFrames).id !== frameOwner.id
      || activeOwner(failingFrames).frameCalls <= frameOwner.frameCalls + 1
      || activeOwner(failingFrames).tick <= frameOwner.tick || failingFrames.pendingTimers !== 1
      || failingFrames.presentationState !== "error" || reporting.updates !== 1 || reporting.logs !== 1) {
    throw new Error(`Presentation failure stopped scheduling or flooded reporting: ${JSON.stringify({ failingFrames, reporting })}`);
  }
  await page.evaluate(() => { window.__ownerTest.frameFailure = null; });
  await context.clock.runFor(64);
  const frameRecovered = await state();
  if (activeOwner(frameRecovered).id !== frameOwner.id || frameRecovered.presentationState
      || Number(frameRecovered.presentedTick) <= frameOwner.tick
      || frameRecovered.pendingTimers !== 1 || frameRecovered.maximumTimers !== 1
      || !await page.evaluate(() => document.getElementById("log").textContent.includes("Presentation resumed"))) {
    throw new Error("The existing owner failed to resume presentation after frame reads recovered");
  }
  await page.evaluate(() => window.__presentationObserver.disconnect());
  observations.presentationRecoveries.push({ fault: "repeated frame read", owner: frameOwner.id,
    attemptedFrames: activeOwner(failingFrames).frameCalls - frameOwner.frameCalls, reports: reporting,
    presentedTick: frameRecovered.presentedTick });

  // Fail after actually modifying the Canvas during capture of a paused world.
  // Recovery must repaint the same publication even though capture was refused;
  // merely continuing callbacks or trusting the old paint identity leaves red.
  await setSpeed(0);
  await context.clock.runFor(32);
  const pausedBeforePaint = await state();
  const pausedOwner = activeOwner(pausedBeforePaint);
  await page.evaluate(() => {
    const fill = CanvasRenderingContext2D.prototype.fillRect;
    window.__canvasFaults = 0;
    CanvasRenderingContext2D.prototype.fillRect = function (...args) {
      if (this.canvas.id !== "sim-canvas") return fill.apply(this, args);
      CanvasRenderingContext2D.prototype.fillRect = fill;
      this.fillStyle = "#ff0000";
      fill.apply(this, args);
      window.__canvasFaults += 1;
      throw new Error("intentional Canvas draw failure");
    };
    document.getElementById("capture-btn").click();
  });
  await context.clock.runFor(16);
  const interruptedPaint = await page.evaluate(() => ({
    faults: window.__canvasFaults,
    capture: document.getElementById("capture-status").dataset.state,
    pixel: Array.from(document.getElementById("sim-canvas").getContext("2d").getImageData(0, 0, 1, 1).data),
  }));
  if (interruptedPaint.faults !== 1 || interruptedPaint.capture !== "error"
      || interruptedPaint.pixel.join(",") !== "255,0,0,255") {
    throw new Error(`Canvas failure did not reach a partial paint and refuse capture: ${JSON.stringify(interruptedPaint)}`);
  }
  await context.clock.runFor(64);
  const repainted = await state();
  const pixel = await page.evaluate(() => Array.from(document.getElementById("sim-canvas")
    .getContext("2d").getImageData(0, 0, 1, 1).data));
  if (activeOwner(repainted).id !== pausedOwner.id || activeOwner(repainted).tick !== pausedOwner.tick
      || repainted.presentedTick !== pausedBeforePaint.presentedTick || repainted.presentationState
      || pixel.join(",") !== "11,17,32,255" || repainted.pendingTimers !== 1 || repainted.maximumTimers !== 1) {
    throw new Error(`Canvas failed to repaint the paused publication: ${JSON.stringify({ repainted, pixel })}`);
  }
  observations.presentationRecoveries.push({ fault: "partial Canvas draw", owner: pausedOwner.id,
    unchangedTick: repainted.presentedTick, interruptedPixel: interruptedPaint.pixel, recoveredPixel: pixel });

  for (const operation of ["reset", "permalink", "checkpoint"]) {
    const previous = activeOwner(await state());
    await page.evaluate((ownerId) => {
      window.__ownerTest.frameFailure = { ownerId, remaining: -1 };
    }, previous.id);
    await context.clock.runFor(32);
    if ((await state()).presentationState !== "error") throw new Error("Replacement case did not observe a presentation fault");
    await replace(operation);
    const recovered = await state();
    const next = activeOwner(recovered);
    if (next.id === previous.id || recovered.records[previous.id - 1].freeCalls !== 1
        || recovered.presentationState || recovered.pendingTimers !== 1 || recovered.maximumTimers !== 1) {
      throw new Error(`${operation} did not recover a failed presentation: ${JSON.stringify(recovered)}`);
    }
    observations.presentationRecoveries.push({ fault: "persistent owner frame read", operation,
      previousOwner: previous.id, presentedOwner: next.id, presentedTick: recovered.presentedTick });
  }
  await page.evaluate(() => { window.__ownerTest.frameFailure = null; });

  // Refusals after allocation are different from a decoder rejecting input:
  // each failed candidate must be freed without moving the paused old world.
  await setSpeed(0);
  for (const operation of ["reset", "permalink", "checkpoint"]) {
    for (const failure of ["create", "speed", "frame", "snapshot", "identity", "launch", "launch_schema", "launch_seed",
      operation === "checkpoint" ? "preset" : "permalink"]) {
      const before = await state();
      const previous = activeOwner(before);
      await page.evaluate((stage) => { window.__ownerTest.nextFailure = stage; }, failure);
      await replace(operation);
      const after = await state();
      const retained = activeOwner(after);
      const candidates = after.records.slice(before.records.length);
      if (retained.id !== previous.id || retained.tick !== previous.tick
          || after.speed !== before.speed || after.preset !== before.preset
          || after.pendingTimers !== 1 || after.maximumTimers !== 1
          || candidates.length !== (failure === "create" ? 0 : 1)
          || candidates.some((owner) => owner.freeCalls !== 1)) {
        throw new Error(`${operation}/${failure} changed the active world or leaked its candidate: ${JSON.stringify(after)}`);
      }
      observations.refusals.push({ operation, failure, retainedOwner: retained.id,
        rejectedOwner: candidates[0]?.id ?? null, retainedTick: retained.tick });
    }
  }

  // A large but valid owner remains usable when the bounded URL codec cannot
  // admit its original recipe. Neither copy nor variant may invent a link.
  await page.evaluate(() => { window.__ownerTest.nextFailure = "launch_unshareable"; });
  await replace("reset");
  const unshareable = activeOwner(await state());
  const unshareableUi = await page.evaluate(() => ({
    text: document.getElementById("launch-info").textContent,
    controls: ["copy-permalink-btn", "fork-btn"].map((id) => document.getElementById(id).disabled),
    link: document.getElementById("permalink-input").value,
  }));
  if (unshareable.permalinkCalls !== 0 || !unshareableUi.controls.every(Boolean)
      || unshareableUi.link !== "" || !unshareableUi.text.includes("exceeds the link size limit")
      || !unshareableUi.text.includes(`Initial seed ${unshareable.seed}`)) {
    throw new Error(`A valid unshareable recipe lost its owner or offered an invented link: ${JSON.stringify(unshareableUi)}`);
  }
  await replace("reset");
  if (await page.locator("#copy-permalink-btn").isDisabled() || await page.locator("#fork-btn").isDisabled()) {
    throw new Error("A fresh bounded recipe did not restore sharing after an unshareable owner");
  }
  observations.unshareableRecipe = { owner: unshareable.id, ui: unshareableUi };

  // A stopped pump must stay stopped until a replacement succeeds. Check drive
  // calls as well as ticks: duplicate timers can drive the same elapsed boundary
  // without producing additional science, so tick progress alone is insufficient.
  for (const operation of ["reset", "permalink", "checkpoint"]) {
    await setSpeed(1);
    await context.clock.runFor(64);
    const previous = activeOwner(await state());
    await page.evaluate((id) => {
      window.__ownerTest.failDriveId = id;
      window.__ownerTest.frameFailure = { ownerId: id, remaining: 1 };
    }, previous.id);
    await context.clock.runFor(32);
    const stopped = await state();
    if (stopped.pendingTimers !== 0) throw new Error("Clock fault left a pump scheduled");
    await context.clock.runFor(80);
    const stillStopped = await state();
    if (activeOwner(stillStopped).driveCalls !== activeOwner(stopped).driveCalls
        || activeOwner(stillStopped).frameCalls <= activeOwner(stopped).frameCalls
        || stillStopped.pendingTimers !== 0 || stillStopped.presentationState
        || !await page.evaluate(() => document.getElementById("log").textContent.includes("Simulation clock stopped"))) {
      throw new Error("Presentation recovery restarted or concealed a stopped scientific clock");
    }
    await page.evaluate(() => {
      // The clock fault was transient. Refusing a candidate must still preserve
      // the old owner's stopped state rather than implicitly trying it again.
      window.__ownerTest.failDriveId = null;
      window.__ownerTest.frameFailure = null;
      window.__ownerTest.nextFailure = "speed";
    });
    await replace(operation);
    const refused = await state();
    if (activeOwner(refused).id !== previous.id || refused.pendingTimers !== 0
        || activeOwner(refused).driveCalls !== activeOwner(stopped).driveCalls
        || activeOwner(refused).tick !== activeOwner(stopped).tick) {
      throw new Error(`${operation} refusal restarted or changed the stopped owner`);
    }
    await replace(operation);
    const replaced = await state();
    const next = activeOwner(replaced);
    if (next.id === previous.id || replaced.records[previous.id - 1].freeCalls !== 1
        || replaced.pendingTimers !== 1 || replaced.maximumTimers !== 1
        || (operation === "checkpoint" && (next.tick !== 37 || replaced.speed !== "0" || replaced.preset !== "mlp"))) {
      throw new Error(`${operation} failed to replace the stopped world: ${JSON.stringify(replaced)}`);
    }
    await setSpeed(1);
    await context.clock.runFor(96);
    const resumed = await state();
    const running = activeOwner(resumed);
    if (running.tick <= next.tick || running.driveCalls <= next.driveCalls
        || resumed.records[previous.id - 1].driveCalls !== activeOwner(stopped).driveCalls
        || resumed.pendingTimers !== 1 || resumed.maximumTimers !== 1) {
      throw new Error(`${operation} did not resume exactly one new clock: ${JSON.stringify(resumed)}`);
    }
    observations.recoveries.push({ operation, previousOwner: previous.id, newOwner: next.id,
      tickBeforeResume: next.tick, tickAfterResume: running.tick, pendingTimers: resumed.pendingTimers });
  }

  for (const operation of ["permalink", "reset", "checkpoint", "reset"]) {
    const previous = activeOwner(await state());
    await replace(operation);
    const after = await state();
    const next = activeOwner(after);
    if (next.id === previous.id || after.records[previous.id - 1].freeCalls !== 1
        || after.pendingTimers !== 1 || after.maximumTimers !== 1) {
      throw new Error(`Running ${operation} replacement leaked an owner or duplicated its timer`);
    }
    observations.replacements.push({ operation, previousOwner: previous.id, newOwner: next.id });
  }

  const beginDelayedImport = async () => {
    await page.evaluate(() => {
      const read = File.prototype.arrayBuffer;
      window.__releaseOwnerRead = null;
      File.prototype.arrayBuffer = function () {
        return new Promise((resolve, reject) => {
          window.__releaseOwnerRead = () => {
            File.prototype.arrayBuffer = read;
            return read.call(this).then(resolve, reject);
          };
        });
      };
    });
    await page.setInputFiles("#checkpoint-load", checkpointFile);
    if (!await page.evaluate(() => typeof window.__releaseOwnerRead === "function")) {
      throw new Error("Delayed import did not reach its asynchronous read boundary");
    }
  };
  const releaseCancelledImport = async () => {
    await page.evaluate(() => window.__releaseOwnerRead());
    if (!await page.evaluate(() => document.getElementById("checkpoint-status").textContent.startsWith("Load cancelled"))) {
      throw new Error("Stale continuation read was not cancelled");
    }
  };

  await setSpeed(0);
  await beginDelayedImport();
  await replace("reset");
  const reset = await state();
  await releaseCancelledImport();
  await context.clock.runFor(32);
  const afterStaleReset = await state();
  if (activeOwner(afterStaleReset).id !== activeOwner(reset).id
      || activeOwner(afterStaleReset).tick !== activeOwner(reset).tick
      || afterStaleReset.records.length !== reset.records.length) {
    throw new Error("Stale continuation replaced or allocated over the reset world");
  }
  observations.staleResetPreservedOwner = activeOwner(reset).id;

  // A connecting remote transport is enough to leave local mode. No server or
  // remote protocol behavior is asserted by this inert socket double.
  await beginDelayedImport();
  await page.evaluate(() => {
    window.WebSocket = class { addEventListener() {} close() {} };
    document.getElementById("remote-connect").click();
  });
  const remote = await state();
  if (remote.pendingTimers !== 0 || !await page.isDisabled("#reset-btn")) {
    throw new Error("Remote mode did not suspend local scheduling and replacement controls");
  }
  await releaseCancelledImport();
  await page.evaluate(() => {
    // Dispatch directly to exercise the guards even though the controls are disabled.
    document.getElementById("reset-btn").dispatchEvent(new Event("click"));
    document.getElementById("load-permalink-btn").dispatchEvent(new Event("click"));
  });
  await context.clock.runFor(64);
  const remoteAfter = await state();
  if (activeOwner(remoteAfter).id !== activeOwner(remote).id
      || activeOwner(remoteAfter).tick !== activeOwner(remote).tick
      || activeOwner(remoteAfter).driveCalls !== activeOwner(remote).driveCalls
      || remoteAfter.records.length !== remote.records.length || remoteAfter.pendingTimers !== 0) {
    throw new Error("Remote mode changed, freed, or drove the retained local owner");
  }
  await page.evaluate(() => document.getElementById("remote-local").click());
  await setSpeed(1);
  await context.clock.runFor(96);
  const returned = await state();
  if (activeOwner(returned).id !== activeOwner(remote).id
      || activeOwner(returned).tick <= activeOwner(remote).tick
      || returned.pendingTimers !== 1 || returned.maximumTimers !== 1) {
    throw new Error("Return to local failed to resume its retained world with one timer");
  }
  observations.staleRemotePreservedOwner = activeOwner(returned).id;

  // Real IndexedDB and DOM, with the same explicit WASM owner double. These
  // cases observe storage refusal and ownership races; stats_dom.test.js also
  // exercises reload and continuation through the generated Rust/WASM package.
  await setSpeed(0);
  await replace("checkpoint");
  const browserClick = async (selector, prefix) => {
    let timeout;
    try {
      return await Promise.race([
        page.evaluate(({ selector, prefix }) => new Promise((resolve) => {
          const status = document.getElementById("checkpoint-browser-status");
          const observer = new MutationObserver(() => {
            if (status.textContent.startsWith(prefix)) {
              observer.disconnect();
              resolve(status.textContent);
            }
          });
          observer.observe(status, { childList: true, characterData: true, subtree: true });
          document.querySelector(selector).click();
        }), { selector, prefix }),
        new Promise((_, reject) => {
          timeout = setTimeout(() => reject(new Error(`${selector} did not produce ${prefix}`)), 10_000);
        }),
      ]);
    } finally {
      clearTimeout(timeout);
    }
  };
  await browserClick("#checkpoint-browser-save", "Saved continuation in this browser:");
  await page.evaluate(async () => {
    const { BrowserContinuationStore } = await import("/continuation-store.js");
    window.__storedGoodRecord = await new BrowserContinuationStore(1024).read();
    window.__replaceStoredRecord = (record) => new Promise((resolve, reject) => {
      const open = indexedDB.open("scriptbots-continuations", 1);
      open.onerror = () => reject(open.error);
      open.onsuccess = () => {
        const database = open.result;
        const transaction = database.transaction("continuations", "readwrite");
        transaction.oncomplete = () => { database.close(); resolve(); };
        transaction.onabort = () => { database.close(); reject(transaction.error); };
        transaction.objectStore("continuations").put(record, "latest");
      };
    });
  });

  const storedRefusals = [];
  for (const defect of ["tick", "corrupt", "oversized", "version"]) {
    await page.evaluate(async (defect) => {
      const record = { ...window.__storedGoodRecord };
      if (defect === "tick") record.tick = String(BigInt(record.tick) + 1n);
      if (defect === "corrupt") record.blob = new Blob([new Uint8Array([0])]);
      if (defect === "oversized") record.blob = new Blob([new Uint8Array(1025)]);
      if (defect === "version") record.schema = "scriptbots.browser-continuation.v999";
      await window.__replaceStoredRecord(record);
      const read = Blob.prototype.arrayBuffer;
      window.__storedReadCount = 0;
      window.__restoreBlobRead = () => { Blob.prototype.arrayBuffer = read; };
      Blob.prototype.arrayBuffer = function () { window.__storedReadCount += 1; return read.call(this); };
    }, defect);
    const before = await state();
    const message = await browserClick("#checkpoint-browser-restore", "Browser restore failed:");
    await context.clock.runFor(32);
    const after = await state();
    const reads = await page.evaluate(() => { window.__restoreBlobRead(); return window.__storedReadCount; });
    const candidates = after.records.slice(before.records.length);
    if (activeOwner(after).id !== activeOwner(before).id || activeOwner(after).tick !== activeOwner(before).tick
        || candidates.length !== (defect === "tick" ? 1 : 0) || candidates.some((owner) => owner.freeCalls !== 1)
        || after.pendingTimers !== 1 || after.maximumTimers !== 1
        || reads !== (["tick", "corrupt"].includes(defect) ? 1 : 0)) {
      throw new Error(`Stored ${defect} did not preserve the owner or byte-allocation boundary`);
    }
    storedRefusals.push({ defect, reads, candidates: candidates.length, message });
  }
  await page.evaluate(() => window.__replaceStoredRecord(window.__storedGoodRecord));

  await page.evaluate(() => {
    const put = IDBObjectStore.prototype.put;
    IDBObjectStore.prototype.put = function (...args) {
      IDBObjectStore.prototype.put = put;
      throw new DOMException("Controlled quota refusal", "QuotaExceededError");
    };
  });
  const beforeQuota = await state();
  const quotaMessage = await browserClick("#checkpoint-browser-save", "Browser save failed:");
  if (!quotaMessage.includes("quota") || activeOwner(await state()).id !== activeOwner(beforeQuota).id) {
    throw new Error("Browser quota refusal changed the active owner or lost its typed explanation");
  }
  await browserClick("#checkpoint-browser-restore", "Restored browser continuation at tick 37;");
  await context.clock.runFor(32);
  if (activeOwner(await state()).tick !== 37) throw new Error("Quota refusal replaced the saved checkpoint");

  const delayedRestore = async () => {
    await page.evaluate(() => {
      const read = Blob.prototype.arrayBuffer;
      let started;
      window.__browserReadStarted = new Promise((resolve) => { started = resolve; });
      Blob.prototype.arrayBuffer = function () {
        Blob.prototype.arrayBuffer = read;
        return new Promise((resolve, reject) => {
          window.__releaseBrowserRead = () => read.call(this).then(resolve, reject);
          started();
        });
      };
      document.getElementById("checkpoint-browser-restore").click();
    });
    await page.evaluate(() => window.__browserReadStarted);
  };
  const staleStoredReads = [];
  for (const change of ["reset", "remote"]) {
    await delayedRestore();
    if (change === "reset") await replace("reset");
    else await page.evaluate(() => document.getElementById("remote-connect").click());
    const changed = await state();
    await page.evaluate(() => window.__releaseBrowserRead());
    const after = await state();
    if (!await page.evaluate(() => document.getElementById("checkpoint-browser-status").textContent.startsWith("Browser restore cancelled"))
        || activeOwner(after).id !== activeOwner(changed).id || after.records.length !== changed.records.length
        || after.pendingTimers !== (change === "remote" ? 0 : 1)) {
      throw new Error(`Delayed browser restore ignored ${change}`);
    }
    if (change === "remote") {
      const disabled = await page.evaluate(() => ["checkpoint-browser-save", "checkpoint-browser-restore"].every((id) => {
        const button = document.getElementById(id);
        button.dispatchEvent(new Event("click"));
        return button.disabled;
      }));
      if (!disabled || (await state()).records.length !== changed.records.length) {
        throw new Error("Remote mode admitted a local browser continuation operation");
      }
      await page.evaluate(() => document.getElementById("remote-local").click());
    }
    staleStoredReads.push({ change, retainedOwner: activeOwner(after).id });
  }
  observations.browserStorage = { storedRefusals, quotaMessage, staleStoredReads };

  // Production DOM/controller integration with the explicit WASM double above.
  // This proves presentation and owner lifetime, not real detector generation.
  await setSpeed(0);
  await context.clock.runFor(300);
  const narrativeBefore = activeOwner(await state());
  const narrativeText = await page.locator("#remote-events li").allTextContents();
  if (narrativeText.length !== 1 || narrativeText[0] !== `#1 · tick 0: <em>owner ${narrativeBefore.id} narrative fixture</em>`
      || await page.locator("#remote-events em").count() !== 0
      || await page.getAttribute("#narrative-timeline", "data-session-id") !== String(2000 + narrativeBefore.id)) {
    throw new Error("Local timeline did not consume exact canonical fixture prose as text for the active owner");
  }
  await context.clock.runFor(1_000);
  const narrativeQuiet = activeOwner(await state());
  if (narrativeQuiet.tick !== narrativeBefore.tick || narrativeQuiet.narrativeReads !== narrativeBefore.narrativeReads) {
    throw new Error("Paused timeline presentation advanced ticks or reread an unchanged publication");
  }
  await page.evaluate(() => document.getElementById("narrative-refresh").click());
  await context.clock.runFor(1);
  const narrativeRefreshed = activeOwner(await state());
  if (narrativeRefreshed.tick !== narrativeBefore.tick || narrativeRefreshed.narrativeReads !== narrativeBefore.narrativeReads + 1
      || await page.locator("#remote-events li").count() !== 1) {
    throw new Error("Explicit local history refresh changed science, duplicated events, or failed to read");
  }
  await replace("checkpoint");
  const narrativeReplaced = activeOwner(await state());
  const replacementText = await page.locator("#remote-events li").allTextContents();
  if (replacementText.length !== 1 || replacementText[0] !== `#1 · tick 0: <em>owner ${narrativeReplaced.id} narrative fixture</em>`
      || await page.getAttribute("#narrative-timeline", "data-session-id") !== String(2000 + narrativeReplaced.id)) {
    throw new Error("Timeline retained old-owner prose when a replacement reused event ordinal one");
  }
  observations.narrative = { evidence: "production DOM and JavaScript with explicit WASM double",
    before: narrativeBefore, refreshed: narrativeRefreshed, replacement: narrativeReplaced,
    narrativeText, replacementText };
  const continuationRecipe = await page.evaluate(() => {
    const before = window.__ownerTest.forkCalls;
    for (const id of ["copy-permalink-btn", "fork-btn"]) document.getElementById(id).dispatchEvent(new Event("click"));
    return { disabled: ["copy-permalink-btn", "fork-btn", "fork-patch-input"].every((id) => document.getElementById(id).disabled),
      kind: document.getElementById("launch-info").dataset.kind, text: document.getElementById("launch-info").textContent,
      forkCalls: window.__ownerTest.forkCalls - before,
      exports: window.__ownerTest.records.find((owner) => owner.freeCalls === 0).permalinkCalls };
  });
  if (!continuationRecipe.disabled || continuationRecipe.kind !== "checkpoint_continuation"
      || !continuationRecipe.text.includes("original launch recipe is unavailable")
      || continuationRecipe.forkCalls !== 0 || continuationRecipe.exports !== 0) {
    throw new Error(`Continuation offered an unrecorded initial-condition recipe: ${JSON.stringify(continuationRecipe)}`);
  }
  observations.continuationRecipe = continuationRecipe;
  await page.evaluate(() => { window.__ownerTest.nextFailure = "launch_missing"; });
  await replace("checkpoint");
  if (!(await page.locator("#copy-permalink-btn").isDisabled()) || !(await page.locator("#fork-btn").isDisabled())
      || await page.getAttribute("#launch-info", "data-kind") !== "checkpoint_continuation"
      || activeOwner(await state()).permalinkCalls !== 0) {
    throw new Error("An older package without launchInfo offered a recipe for a saved continuation");
  }
  const finalState = await state();
  observations.releasedOwners = finalState.records.filter((owner) => owner.freeCalls === 1).length;
  observations.maximumPendingTimers = finalState.maximumTimers;
  observations.maximumPendingFrames = finalState.maximumFrames;
  return observations;
}

function createServer() {
  return http.createServer((req, res) => {
    const url = new URL(req.url, "http://127.0.0.1");
    if (url.pathname === "/" || url.pathname === "/index.html") {
      res.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
      res.end(fs.readFileSync(path.join(webDir, "index.html")));
    } else if (url.pathname === "/main.js") {
      res.writeHead(200, { "Content-Type": "application/javascript; charset=utf-8" });
      res.end(fs.readFileSync(path.join(webDir, "main.js")));
    } else if (["/remote.js", "/experiments.js", "/experiment-protocol.js", "/experiment-view.js", "/checkpoint-lab.js", "/study-report.js", "/continuation-store.js", "/inspector.js", "/genome.js", "/interventions.js", "/narrative.js", "/recipes.js", "/renderers/canvas.js", "/renderers/webgpu.js", "/renderers/environment.js"].includes(url.pathname)) {
      res.writeHead(200, { "Content-Type": "application/javascript; charset=utf-8" });
      res.end(fs.readFileSync(path.join(webDir, url.pathname.slice(1))));
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

  const browser = await chromium.launch({
    headless: true,
    executablePath: process.env.SCRIPTBOTS_CHROMIUM_EXECUTABLE || undefined,
  });

  try {
    let ownerLifecycle;
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
        }),
      );

      // Load with mismatch permalink
      await page.goto(`${baseUrl}/?world=sbw1.mismatch_sample`);

      // Verify the banner is rendered in the DOM with mismatch warning
      const bannerDisplay = await page.$eval(
        "#mismatch-banner",
        (el) => window.getComputedStyle(el).display,
      );
      if (bannerDisplay === "none") {
        throw new Error(
          "Expected #mismatch-banner to be visible on mismatch, but display was none",
        );
      }

      const bannerText = await page.textContent("#mismatch-banner");
      if (!bannerText.includes("Build Mismatch Warning")) {
        throw new Error(`Expected mismatch warning title in banner text, got: ${bannerText}`);
      }
      if (
        !bannerText.includes("0xcccccccccccccccc") ||
        !bannerText.includes("0x3333333333333333")
      ) {
        throw new Error(`Expected core digests in banner details, got: ${bannerText}`);
      }

      // Assert that the simulation is NOT blocked: advance clock past 500ms and verify tick updates
      await context.clock.runFor(600);
      const tickText = await page.textContent("#metric-tick");
      if (tickText === "–" || tickText === "0") {
        throw new Error(`Simulation should still run despite mismatch, got tick: ${tickText}`);
      }

      // DOM-boundary fixture: actual WASM formats these digests from u64, so
      // this proves safe text rendering, not an exploitable permalink path.
      for (const kind of ["mismatch", "compatible"]) {
        await page.fill("#permalink-input", `sbw1.${kind}_markup`);
        await page.click("#load-permalink-btn");
        const literal = await page.textContent("#mismatch-banner");
        if (!literal.includes("<img src=x onerror=window.__bannerExecuted=true>")
            || await page.locator("#mismatch-banner img").count() !== 0
            || await page.evaluate(() => !!window.__bannerExecuted)) {
          throw new Error(`${kind} banner interpreted build details as HTML`);
        }
      }

      for (const [status, heading] of [["recorded_match", "Recorded Build Inputs Match"],
        ["unavailable", "Build Identity Unavailable"], ["mismatch", "Build Mismatch Warning"]]) {
        await page.fill("#permalink-input", `sbb1.${status}_recipe`);
        await page.click("#load-permalink-btn");
        const banner = await page.textContent("#mismatch-banner");
        const info = await page.textContent("#launch-info");
        if (!banner.includes(heading) || !banner.includes("<em>Recorded recipe fixture; source input fingerprints only.</em>")
            || await page.locator("#mismatch-banner em").count() !== 0
            || !info.includes("18446744073709551615") || !info.includes("151 founders") || !info.includes("dwraon brains")
            || !info.includes("tick zero") || await page.locator("#copy-permalink-btn").isDisabled()) {
          throw new Error(`Browser recipe metadata or scoped build status was misrepresented: ${JSON.stringify({ status, banner, info })}`);
        }
        if (status === "unavailable" && (!banner.includes("equality is unknown") || banner.includes("Build Mismatch Warning"))) {
          throw new Error("Unavailable recorded identity was treated as failed reproduction");
        }
        const tickBefore = BigInt(await page.textContent("#metric-tick"));
        await context.clock.runFor(64);
        if (BigInt(await page.textContent("#metric-tick")) <= tickBefore) throw new Error(`${status} prevented a recipe from running`);
      }

      await page.fill("#permalink-input", "sbw1.gallery_reference");
      await page.click("#load-permalink-btn");
      if (!(await page.textContent("#mismatch-banner")).includes("Gallery Build Reference")
          || !(await page.textContent("#launch-info")).includes("does not encode a browser founder population")) {
        throw new Error("Legacy gallery metadata was promoted to a measured browser recipe");
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
        }),
      );

      // Load with permalink that has diffs
      await page.goto(`${baseUrl}/?world=sbw1.with_diff_test`);

      const tableDisplay = await page.$eval(
        "#diff-table",
        (el) => window.getComputedStyle(el).display,
      );
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
        }),
      );

      await page.goto(`${baseUrl}/?world=sbw1.root_world_link`);

      // Input patch and trigger fork
      await page.fill("#fork-patch-input", '{"food_growth_rate": 0.05}');
      await page.click("#fork-btn");

      // Verify fork-output displays child link
      const outputDisplay = await page.$eval(
        "#fork-output",
        (el) => window.getComputedStyle(el).display,
      );
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
        throw new Error(
          `Running world was mutated! Expected sbw1.root_world_link, got: ${currentInput}`,
        );
      }

      await page.fill("#permalink-input", "sbb1.root_recipe");
      await page.click("#load-permalink-btn");
      const parentOwner = await page.evaluate(() => window.__ownerTest.records.find((owner) => owner.freeCalls === 0).id);
      await page.fill("#fork-patch-input", '{"food_growth_rate": 0.05}');
      await page.click("#fork-btn");
      const childRecipe = await page.textContent("#fork-recipe-info");
      if ((await page.textContent("#fork-child-link")) !== "sbb1.child_recipe_with_diff"
          || !childRecipe.includes("18446744073709551615") || !childRecipe.includes("151 founders")
          || !childRecipe.includes("dwraon brains") || !childRecipe.includes("tick zero")
          || await page.inputValue("#permalink-input") !== "sbb1.root_recipe"
          || await page.evaluate(() => window.__ownerTest.records.find((owner) => owner.freeCalls === 0).id) !== parentOwner) {
        throw new Error("Variant recipe changed its running parent or lost founder recipe metadata");
      }
      await page.click("#load-forked-btn");
      if (await page.inputValue("#permalink-input") !== "sbb1.child_recipe_with_diff"
          || await page.evaluate(() => window.__ownerTest.records.find((owner) => owner.freeCalls === 0).id) === parentOwner
          || await page.$eval("#fork-output", (element) => window.getComputedStyle(element).display) !== "none") {
        throw new Error("Starting a variant did not replace its owner or clear the old child preview");
      }

      await context.close();
    }

    // =========================================================================
    // Test 4: Rejected Inputs, Atomic Owner Replacement, and Clock Recovery
    // =========================================================================
    {
      const context = await browser.newContext();
      const epoch = Date.UTC(2026, 0, 1);
      await context.clock.install({ time: epoch });
      await context.clock.pauseAt(epoch + 1_000);
      const page = await context.newPage();
      const pageErrors = [];
      page.on("pageerror", (err) => pageErrors.push(err.message));

      await page.route("**/pkg/scriptbots_web.js", (route) =>
        route.fulfill({
          status: 200,
          contentType: "application/javascript; charset=utf-8",
          body: mockWasmModule,
        }),
      );

      await page.goto(`${baseUrl}/?world=invalid_startup_link`);
      await context.clock.runFor(32);
      if (!await page.evaluate(() => document.getElementById("permalink-status").textContent.startsWith("Error:")
          && window.__ownerTest.records.length === 0 && window.__ownerTimers.pending.size === 0)) {
        throw new Error("Invalid startup permalink did not leave a recoverable ownerless page");
      }
      await page.evaluate(() => document.getElementById("reset-btn").click());
      await context.clock.runFor(32);
      if (!await page.evaluate(() => window.__ownerTest.records.length === 1
          && window.__ownerTest.records[0].freeCalls === 0 && window.__ownerTest.records[0].driveCalls > 0
          && document.getElementById("renderer-status").dataset.sessionId === "2001"
          && window.__ownerTimers.pending.size === 1)) {
        throw new Error("Reset failed to recover from an invalid startup permalink");
      }

      // Attempt to load an invalid permalink
      await page.evaluate(() => {
        document.getElementById("permalink-input").value = "invalid_corrupted_link";
        document.getElementById("load-permalink-btn").click();
      });

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
      await page.evaluate(() => document.getElementById("reset-btn").click());
      await context.clock.runFor(32);
      const postResetStatus = await page.textContent("#permalink-status");
      if (postResetStatus.toLowerCase().includes("error")) {
        throw new Error(`Instance remained poisoned after reset: ${postResetStatus}`);
      }

      ownerLifecycle = await runOwnerLifecycleCases(page, context);
      ownerLifecycle.invalidStartupRecovered = true;
      if (pageErrors.length > 0) {
        throw new Error(`Unhandled page errors during owner replacement: ${pageErrors.join("; ")}`);
      }
      await context.close();
    }

    const result = {
      schema: "scriptbots.browser-permalink-fork-dom.v1",
      timestamp: new Date().toISOString(),
      status: "pass",
      cases_passed: 4,
      cases_failed: 0,
      observations: { ownerLifecycle },
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
