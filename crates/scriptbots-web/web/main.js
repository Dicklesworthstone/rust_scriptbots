import initWasm, { init_sim, default_init_options, version } from "./pkg/scriptbots_web.js";

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

const metrics = {
    frameCount: 0,
    lastFrameTs: performance.now(),
    lastStatsTs: performance.now(),
    fps: 0,
    tps: 0,
    tick: 0,
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
    appendLog("Reset requested");
});

function appendLog(message) {
    const line = `[${new Date().toLocaleTimeString()}] ${message}`;
    logView.textContent = `${line}\n${logView.textContent}`.slice(0, 4096);
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
    metrics.frameCount += 1;
    const frameDt = now - metrics.lastFrameTs;
    metrics.lastFrameTs = now;

    if (frameDt > 0) {
        const instantaneous = 1000 / frameDt;
        metrics.fps = metrics.fps * 0.85 + instantaneous * 0.15;
    }

    const statsDt = now - metrics.lastStatsTs;
    metrics.tick = snapshot.tick;

    if (statsDt >= 500) {
        const deltaTicks = snapshot.tick - (metrics.previousTick ?? 0);
        metrics.tps = deltaTicks * (1000 / statsDt);
        metrics.previousTick = snapshot.tick;
        metrics.lastStatsTs = now;

        fpsEl.textContent = metrics.fps.toFixed(1);
        tpsEl.textContent = metrics.tps.toFixed(1);
        tickEl.textContent = snapshot.tick.toLocaleString();
        popEl.textContent = snapshot.summary.agent_count.toLocaleString();
        birthsEl.textContent = snapshot.summary.births.toLocaleString();
        deathsEl.textContent = snapshot.summary.deaths.toLocaleString();
        energyEl.textContent = snapshot.summary.average_energy.toFixed(3);
        healthEl.textContent = snapshot.summary.average_health.toFixed(3);
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

async function resetSimulation(populationOverride) {
    const defaults = default_init_options();
    const seed = Math.floor(Math.random() * 1_000_000);
    const options = {
        ...defaults,
        seed,
        population: populationOverride,
        world_width: defaults.world_width ?? 1280,
        world_height: defaults.world_height ?? 720,
    };

    options.config = {
        ...(defaults.config ?? {}),
        rng_seed: seed,
        population_minimum: 0,
        population_spawn_interval: 0,
    };

    simHandle = init_sim(options);
    appendLog(`Simulation reset (seed=${seed}, population=${populationOverride})`);
}

async function bootstrap() {
    try {
        await initWasm();
        const tag = version();
        versionEl.textContent = tag;
        appendLog(`Loaded ${tag}`);
        await resetSimulation(population);
        requestAnimationFrame(stepSimulation);
    } catch (err) {
        console.error("Failed to initialise wasm module", err);
        appendLog(`Bootstrap failed: ${err}`);
    }
}

bootstrap();
