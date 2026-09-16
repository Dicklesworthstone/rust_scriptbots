import initWasm, * as wasm from "./pkg/scriptbots_web.js";

const {
    init_sim,
    default_init_options,
    version,
    init_from_permalink,
    permalink_of,
    fork,
    permalink_diff,
    build_identity,
    check_build_match,
} = wasm;

const canvas = document.getElementById("sim-canvas");
const ctx = canvas.getContext("2d", { alpha: false, desynchronized: true });
const fpsEl = document.getElementById("metric-fps");
const tpsEl = document.getElementById("metric-tps");
const tickEl = document.getElementById("metric-tick");
const popEl = document.getElementById("metric-population");
const birthsEl = document.getElementById("metric-births");
const deathsEl = document.getElementById("metric-deaths");
const energyEl = document.getElementById("metric-energy");
const healthEl = document.getElementById("metric-health");
const resetButton = document.getElementById("reset-btn");
const stepsSlider = document.getElementById("steps-per-frame");
const populationSlider = document.getElementById("population");
const logView = document.getElementById("log");
const versionEl = document.getElementById("version");

const mismatchBannerEl = document.getElementById("mismatch-banner");
const permalinkInput = document.getElementById("permalink-input");
const loadPermalinkBtn = document.getElementById("load-permalink-btn");
const copyPermalinkBtn = document.getElementById("copy-permalink-btn");
const permalinkStatus = document.getElementById("permalink-status");
const forkPatchInput = document.getElementById("fork-patch-input");
const forkBtn = document.getElementById("fork-btn");
const forkOutput = document.getElementById("fork-output");
const forkChildLink = document.getElementById("fork-child-link");
const loadForkedBtn = document.getElementById("load-forked-btn");
const copyForkBtn = document.getElementById("copy-fork-btn");
const diffContainer = document.getElementById("diff-container");
const diffEmpty = document.getElementById("diff-empty");
const diffTable = document.getElementById("diff-table");
const diffTbody = document.getElementById("diff-tbody");

let currentPermalink = null;
let latestForkedLink = null;

const metrics = {
    lastFrameTs: performance.now(),
    lastStatsTs: performance.now(),
    previousTick: null,
    fps: 0,
    tps: 0,
    tick: 0,
    statsTickAcc: 0,
};

const perfWindow = {
    frameAcc: 0,
    tickAcc: 0,
    lastLog: performance.now(),
};

let simHandle = null;
let stepsPerFrame = Number(stepsSlider.value);
let queuedReset = false;
let population = Number(populationSlider.value);

stepsSlider.addEventListener("input", () => {
    stepsPerFrame = Number(stepsSlider.value);
    appendLog(`Steps/frame set to ${stepsPerFrame}`);
});

populationSlider.addEventListener("input", () => {
    population = Number(populationSlider.value);
});

resetButton.addEventListener("click", () => {
    queuedReset = true;
    if (permalinkStatus) {
        permalinkStatus.textContent = "";
    }
    appendLog("Reset requested");
});

function appendLog(message) {
    const line = `[${new Date().toLocaleTimeString()}] ${message}`;
    logView.textContent = `${line}\n${logView.textContent}`.slice(0, 4096);
}

function resetPerformanceWindows() {
    const now = performance.now();
    metrics.lastFrameTs = now;
    metrics.lastStatsTs = now;
    metrics.previousTick = null;
    metrics.fps = 0;
    metrics.tps = 0;
    metrics.tick = 0;
    metrics.statsTickAcc = 0;
    perfWindow.frameAcc = 0;
    perfWindow.tickAcc = 0;
    perfWindow.lastLog = now;
}

function scaleFactor(world) {
    return {
        x: canvas.width / world.width,
        y: canvas.height / world.height,
    };
}

function drawSnapshot(snapshot) {
    ctx.fillStyle = "#0b1120";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const scale = scaleFactor(snapshot.world);

    for (const agent of snapshot.agents) {
        const px = agent.position[0] * scale.x;
        const py = agent.position[1] * scale.y;
        const radius = Math.max(1.5, 3.5 * Math.sqrt(agent.health + 0.15));

        ctx.beginPath();
        ctx.fillStyle = `rgba(${Math.round(agent.color[0] * 255)}, ${Math.round(agent.color[1] * 255)}, ${Math.round(agent.color[2] * 255)}, 0.9)`;
        ctx.arc(px, py, radius, 0, Math.PI * 2);
        ctx.fill();

        if (agent.boost) {
            ctx.beginPath();
            ctx.strokeStyle = "rgba(30, 144, 255, 0.65)";
            ctx.lineWidth = 1.2;
            ctx.arc(px, py, radius + 2.0, 0, Math.PI * 2);
            ctx.stroke();
        }
    }
}

function updateStats(snapshot, now) {
    const frameDt = now - metrics.lastFrameTs;
    metrics.lastFrameTs = now;

    if (frameDt > 0) {
        const instantaneous = 1000 / frameDt;
        metrics.fps = metrics.fps * 0.85 + instantaneous * 0.15;
    }

    const statsDt = now - metrics.lastStatsTs;
    const previousTick = metrics.previousTick ?? snapshot.tick;
    const deltaTicks = snapshot.tick - previousTick;
    metrics.previousTick = snapshot.tick;
    metrics.tick = snapshot.tick;
    metrics.statsTickAcc += deltaTicks;

    if (statsDt >= 500) {
        metrics.tps = metrics.statsTickAcc * (1000 / statsDt);
        metrics.statsTickAcc = 0;
        metrics.lastStatsTs = now;

        fpsEl.textContent = metrics.fps.toFixed(1);
        tpsEl.textContent = metrics.tps.toFixed(1);
        tickEl.textContent = snapshot.tick.toLocaleString();
        popEl.textContent = snapshot.summary.agentCount.toLocaleString();
        birthsEl.textContent = snapshot.summary.births.toLocaleString();
        deathsEl.textContent = snapshot.summary.deaths.toLocaleString();
        energyEl.textContent = snapshot.summary.averageEnergy.toFixed(3);
        healthEl.textContent = snapshot.summary.averageHealth.toFixed(3);
    }

    perfWindow.frameAcc += 1;
    perfWindow.tickAcc += deltaTicks;
    const logDt = now - perfWindow.lastLog;
    if (logDt >= 5000) {
        const durationSeconds = logDt / 1000;
        const avgFps = perfWindow.frameAcc / durationSeconds;
        const avgTps = perfWindow.tickAcc / durationSeconds;
        appendLog(
            `5s avg — FPS: ${avgFps.toFixed(1)} | TPS: ${avgTps.toFixed(1)} | Population: ${snapshot.summary.agentCount}`,
        );
        perfWindow.frameAcc = 0;
        perfWindow.tickAcc = 0;
        perfWindow.lastLog = now;
    }
}

function stepSimulation(now) {
    if (!simHandle) {
        requestAnimationFrame(stepSimulation);
        return;
    }

    if (queuedReset) {
        resetSimulation(population).catch((err) => {
            console.error(err);
            appendLog(`Reset failed: ${err}`);
        });
        queuedReset = false;
    }

    let snapshot = null;
    for (let i = 0; i < stepsPerFrame; i += 1) {
        snapshot = simHandle.tick(1);
    }

    drawSnapshot(snapshot);
    updateStats(snapshot, now);
    requestAnimationFrame(stepSimulation);
}

function updateMismatchBanner(matchResult) {
    if (!mismatchBannerEl) return;

    if (!matchResult || matchResult.status === "exact") {
        mismatchBannerEl.className = "";
        mismatchBannerEl.style.display = "none";
        mismatchBannerEl.innerHTML = "";
        return;
    }

    if (matchResult.status === "compatible") {
        mismatchBannerEl.className = "banner-compatible";
        mismatchBannerEl.style.display = "block";
        mismatchBannerEl.innerHTML = `
            <div class="banner-title">
                <span>&#9432;</span>
                <span>Compatible Build Note</span>
            </div>
            <div>This world was recorded on a compatible build. Toolchain or lockfile differ, but core simulation digest matches.</div>
            <div class="banner-details">
                <div>Link Build: toolchain=${matchResult.link_toolchain_digest}, lockfile=${matchResult.link_lockfile_digest}, core=${matchResult.link_core_digest}</div>
                <div>Running Build: toolchain=${matchResult.local_toolchain_digest}, lockfile=${matchResult.local_lockfile_digest}, core=${matchResult.local_core_digest}</div>
            </div>
        `;
        return;
    }

    if (matchResult.status === "mismatch") {
        mismatchBannerEl.className = "banner-mismatch";
        mismatchBannerEl.style.display = "block";
        mismatchBannerEl.innerHTML = `
            <div class="banner-title">
                <span>&#9888;</span>
                <span>Build Mismatch Warning</span>
            </div>
            <div>This world was recorded on a different build (core digest ${matchResult.link_core_digest}). You are on build (core digest ${matchResult.local_core_digest}). Trajectories may diverge after tick 0.</div>
            <div class="banner-details">
                <div>Link Build: toolchain=${matchResult.link_toolchain_digest}, lockfile=${matchResult.link_lockfile_digest}, core=${matchResult.link_core_digest}</div>
                <div>Running Build: toolchain=${matchResult.local_toolchain_digest}, lockfile=${matchResult.local_lockfile_digest}, core=${matchResult.local_core_digest}</div>
                <div style="margin-top: 0.35rem; font-style: italic; opacity: 0.85;">Cross-target floating point equivalence between native and WebAssembly is an open research area (bd-2wj).</div>
            </div>
        `;
    }
}

function updateDiffTable(linkStr) {
    if (!diffTable || !diffTbody || !diffEmpty) return;

    if (!linkStr || !permalink_diff) {
        diffTable.style.display = "none";
        diffEmpty.style.display = "block";
        diffEmpty.textContent = "No knob differences from base scenario.";
        return;
    }

    try {
        const rows = permalink_diff(linkStr);
        if (!rows || rows.length === 0) {
            diffTable.style.display = "none";
            diffEmpty.style.display = "block";
            diffEmpty.textContent = "No knob differences from base scenario (0 diffs).";
            diffTbody.innerHTML = "";
            return;
        }

        diffEmpty.style.display = "none";
        diffTable.style.display = "table";
        diffTbody.innerHTML = "";

        for (const row of rows) {
            const tr = document.createElement("tr");
            tr.style.borderBottom = "1px solid rgba(255,255,255,0.05)";

            const tdKnob = document.createElement("td");
            tdKnob.style.padding = "0.25rem 0.4rem";
            tdKnob.style.fontFamily = "monospace";
            tdKnob.textContent = row.knob;

            const tdParent = document.createElement("td");
            tdParent.style.padding = "0.25rem 0.4rem";
            tdParent.textContent = (row.parent_value ?? row.parentValue ?? 0).toString();

            const tdThis = document.createElement("td");
            tdThis.style.padding = "0.25rem 0.4rem";
            tdThis.style.fontWeight = "600";
            tdThis.style.color = "#93c5fd";
            tdThis.textContent = (row.this_value ?? row.thisValue ?? 0).toString();

            tr.appendChild(tdKnob);
            tr.appendChild(tdParent);
            tr.appendChild(tdThis);
            diffTbody.appendChild(tr);
        }
    } catch (err) {
        console.warn("Could not compute permalink diff:", err);
        diffTable.style.display = "none";
        diffEmpty.style.display = "block";
        diffEmpty.textContent = `Could not compute diff: ${err}`;
    }
}

async function loadWorldFromPermalink(linkStr) {
    if (!linkStr) return;
    try {
        if (permalinkStatus) {
            permalinkStatus.textContent = "Loading permalink...";
            permalinkStatus.style.color = "#93c5fd";
        }

        let matchResult = null;
        if (check_build_match) {
            try {
                matchResult = check_build_match(linkStr);
            } catch (e) {
                console.warn("Failed to check build match:", e);
            }
        }

        if (!init_from_permalink) {
            throw new Error("init_from_permalink is not exported by wasm module");
        }

        const handle = init_from_permalink(linkStr);
        simHandle = handle;
        currentPermalink = linkStr;
        if (permalinkInput) {
            permalinkInput.value = linkStr;
        }

        try {
            if (permalink_of) {
                currentPermalink = permalink_of(simHandle);
            } else if (simHandle.permalinkOf) {
                currentPermalink = simHandle.permalinkOf();
            }
        } catch (_) {}

        updateMismatchBanner(matchResult);
        updateDiffTable(linkStr);
        resetPerformanceWindows();

        if (permalinkStatus) {
            permalinkStatus.textContent = "World loaded from permalink";
            permalinkStatus.style.color = "#34d399";
        }
        appendLog(`Loaded permalink world (${linkStr.slice(0, 20)}…)`);

        try {
            const url = new URL(window.location.href);
            url.searchParams.set("world", linkStr);
            window.history.replaceState({}, "", url.toString());
        } catch (_) {}
    } catch (err) {
        console.error("Failed to load permalink:", err);
        if (permalinkStatus) {
            permalinkStatus.textContent = `Error: ${err}`;
            permalinkStatus.style.color = "#f87171";
        }
        appendLog(`Failed to load permalink: ${err}`);
    }
}

async function handleFork() {
    if (!simHandle) {
        appendLog("Cannot fork: simulation not running");
        return;
    }

    let patch = {};
    const patchText = forkPatchInput ? forkPatchInput.value.trim() : "";
    if (patchText) {
        try {
            patch = JSON.parse(patchText);
        } catch (e) {
            if (permalinkStatus) {
                permalinkStatus.textContent = "Invalid JSON in fork knob patch";
                permalinkStatus.style.color = "#f87171";
            }
            appendLog(`Fork error: invalid JSON: ${e.message}`);
            return;
        }
    }

    try {
        let childLink = null;
        if (fork) {
            childLink = fork(simHandle, patch);
        } else if (simHandle.fork) {
            childLink = simHandle.fork(patch);
        } else {
            throw new Error("fork is not exported by wasm module");
        }

        latestForkedLink = childLink;
        if (forkChildLink) {
            forkChildLink.textContent = childLink;
        }
        if (forkOutput) {
            forkOutput.style.display = "block";
        }
        if (permalinkStatus) {
            permalinkStatus.textContent = "Fork created successfully";
            permalinkStatus.style.color = "#34d399";
        }
        appendLog(`Created fork child link (${childLink.slice(0, 20)}…)`);
    } catch (err) {
        console.error("Fork failed:", err);
        if (permalinkStatus) {
            permalinkStatus.textContent = `Fork error: ${err}`;
            permalinkStatus.style.color = "#f87171";
        }
        appendLog(`Fork failed: ${err}`);
    }
}

if (loadPermalinkBtn) {
    loadPermalinkBtn.addEventListener("click", () => {
        const link = permalinkInput ? permalinkInput.value.trim() : "";
        if (link) {
            loadWorldFromPermalink(link);
        }
    });
}

if (copyPermalinkBtn) {
    copyPermalinkBtn.addEventListener("click", () => {
        if (!currentPermalink && simHandle) {
            try {
                if (permalink_of) {
                    currentPermalink = permalink_of(simHandle);
                } else if (simHandle.permalinkOf) {
                    currentPermalink = simHandle.permalinkOf();
                }
            } catch (_) {}
        }
        if (currentPermalink) {
            navigator.clipboard.writeText(currentPermalink).then(() => {
                if (permalinkStatus) {
                    permalinkStatus.textContent = "Permalink copied to clipboard";
                    permalinkStatus.style.color = "#34d399";
                }
            }).catch(() => {
                if (permalinkInput) {
                    permalinkInput.select();
                }
            });
        }
    });
}

if (forkBtn) {
    forkBtn.addEventListener("click", handleFork);
}

if (loadForkedBtn) {
    loadForkedBtn.addEventListener("click", () => {
        if (latestForkedLink) {
            loadWorldFromPermalink(latestForkedLink);
        }
    });
}

if (copyForkBtn) {
    copyForkBtn.addEventListener("click", () => {
        if (latestForkedLink) {
            navigator.clipboard.writeText(latestForkedLink).then(() => {
                if (permalinkStatus) {
                    permalinkStatus.textContent = "Fork link copied to clipboard";
                    permalinkStatus.style.color = "#34d399";
                }
            });
        }
    });
}

async function resetSimulation(populationOverride) {
    const defaults = default_init_options ? default_init_options() : {};
    const seed = Math.floor(Math.random() * 1_000_000);
    const options = {
        ...defaults,
        seed,
        population: populationOverride,
        world_width: defaults.world_width ?? 1280,
        world_height: defaults.world_height ?? 720,
    };

    simHandle = init_sim(options);
    resetPerformanceWindows();

    currentPermalink = null;
    try {
        if (permalink_of) {
            currentPermalink = permalink_of(simHandle);
        } else if (simHandle && simHandle.permalinkOf) {
            currentPermalink = simHandle.permalinkOf();
        }
        if (currentPermalink && permalinkInput) {
            permalinkInput.value = currentPermalink;
        }
    } catch (_) {}

    updateMismatchBanner(null);
    updateDiffTable(currentPermalink);

    if (permalinkStatus) {
        permalinkStatus.textContent = "";
    }
    if (forkOutput) {
        forkOutput.style.display = "none";
    }

    try {
        const url = new URL(window.location.href);
        url.searchParams.delete("world");
        url.searchParams.delete("permalink");
        window.history.replaceState({}, "", url.toString());
    } catch (_) {}

    appendLog(`Simulation reset (seed=${seed}, population=${populationOverride})`);
}

async function bootstrap() {
    try {
        await initWasm();
        const tag = version ? version() : "unknown";
        if (versionEl) {
            versionEl.textContent = tag;
        }
        appendLog(`Loaded ${tag}`);

        const urlParams = new URLSearchParams(window.location.search);
        const worldParam = urlParams.get("world") || urlParams.get("permalink");
        if (worldParam && init_from_permalink) {
            await loadWorldFromPermalink(worldParam);
        } else {
            await resetSimulation(population);
        }

        requestAnimationFrame(stepSimulation);
    } catch (err) {
        console.error("Failed to initialise wasm module", err);
        appendLog(`Bootstrap failed: ${err}`);
    }
}

bootstrap();
