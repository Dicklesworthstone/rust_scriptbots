// One shared palette/raster policy for Canvas and WebGPU. This is a presentation
// of actual bounded source samples, never a replacement for scientific metrics.
const MAX_SIDE = 128;
const TERRAIN = Object.freeze({
  DeepWater: [17, 48, 70], ShallowWater: [39, 88, 105], Sand: [147, 126, 82],
  Grass: [54, 79, 42], Bloom: [66, 98, 39], Rock: [101, 105, 99],
});
const MODES = new Set(["combined", "terrain", "food", "water", "agents"]);
const clamp = (value) => Math.max(0, Math.min(1, value));

function revision(value) {
  return typeof value === "string" && /^(0|[1-9][0-9]{0,19})$/.test(value)
    && BigInt(value) < (1n << 64n);
}

function mix(color, target, amount) {
  return color.map((channel, index) => channel + (target[index] - channel) * amount);
}

export function environmentDescription(environment) {
  if (!environment) return "Environment layers are unavailable for this frame.";
  const sampled = environment.width !== environment.sourceWidth || environment.height !== environment.sourceHeight;
  return `${environment.sourceWidth} × ${environment.sourceHeight} source cells${sampled
    ? `; ${environment.width} × ${environment.height} center-sampled overview` : "; exact cell view"}. ${environment.hydrologyEnabled
    ? "Water shows observed depth." : "Hydrology is disabled."}`;
}

/** A single bounded reusable RGBA grid, invalidated by source identity and content revisions. */
export class EnvironmentRaster {
  constructor() { this.current = null; }

  update(environment, identity, world, mode = "combined", epoch = 0, generation = 0) {
    if (!MODES.has(mode)) throw new Error("Unknown environment layer");
    if (!environment || mode === "agents") {
      this.current = null;
      return null;
    }
    const env = environment;
    if (![env.width, env.height, env.sourceWidth, env.sourceHeight, env.cellSize,
      world?.width, world?.height].every((value) => Number.isSafeInteger(value) && value > 0)
        || env.width !== Math.min(env.sourceWidth, MAX_SIDE)
        || env.height !== Math.min(env.sourceHeight, MAX_SIDE)
        || env.sourceWidth * env.cellSize !== world.width
        || env.sourceHeight * env.cellSize !== world.height
        || !Array.isArray(env.cells) || env.cells.length !== env.width * env.height
        || ![env.terrainRevision, env.foodRevision, env.hydrologyRevision].every(revision)
        || !Number.isFinite(env.foodMax) || env.foodMax <= 0
        || typeof env.hydrologyEnabled !== "boolean"
        || typeof identity?.sessionId !== "string") {
      throw new Error("Environment grid disagrees with its world or exceeds its bounds");
    }
    // Local owners can reuse the seed-derived session ID after loading or reset.
    // The frontend's source generation distinguishes those independent lifetimes.
    const key = JSON.stringify([generation, identity.sessionId, epoch, env.width, env.height, env.sourceWidth,
      env.sourceHeight, env.cellSize, world.width, world.height, env.terrainRevision,
      env.foodRevision, env.hydrologyRevision, env.foodMax, env.hydrologyEnabled, mode]);
    if (this.current?.key === key) return this.current;
    const pixels = new Uint8ClampedArray(env.cells.length * 4);
    env.cells.forEach((cell, index) => {
      if (!Object.hasOwn(TERRAIN, cell.terrain) || !Number.isFinite(cell.elevation)
          || !Number.isFinite(cell.food) || cell.food < 0
          || !Number.isFinite(cell.waterDepth) || cell.waterDepth < 0
          || (!env.hydrologyEnabled && cell.waterDepth !== 0)) {
        throw new Error("Environment contains invalid source samples");
      }
      let color = mode === "food" || mode === "water" ? [18, 27, 37]
        : TERRAIN[cell.terrain].map((channel) => channel * (0.82 + clamp(cell.elevation) * 0.35));
      if (mode === "combined" || mode === "food") {
        color = mix(color, [120, 180, 48], clamp(cell.food / env.foodMax) * (mode === "food" ? 1 : 0.7));
      }
      if (mode === "combined" || mode === "water") {
        color = mix(color, [39, 132, 195], cell.waterDepth / (cell.waterDepth + 0.15) * (mode === "water" ? 1 : 0.85));
      }
      pixels.set([...color.map(Math.round), 255], index * 4);
    });
    this.current = { key, width: env.width, height: env.height, pixels,
      coverage: [env.sourceWidth * env.cellSize / world.width, env.sourceHeight * env.cellSize / world.height] };
    return this.current;
  }
}
