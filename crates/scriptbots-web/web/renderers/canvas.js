import { EnvironmentRaster } from "./environment.js";
import { agentRadius, agentUid } from "../inspector.js";
import { worldProjection, visitWorldCopies, visitWorldTiles } from "../world-view.js";

export class CanvasRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d", { alpha: false, desynchronized: true });
    if (!this.ctx) throw new Error("Canvas 2D is unavailable");
    this.clearColor = "#0b1120";
    this.environment = new EnvironmentRaster();
    this.environmentCanvas = null;
    this.environmentKey = null;
  }

  render(snapshot, presentation = {}) {
    if (!this.ctx || !snapshot) {
      return false;
    }
    const ctx = this.ctx;
    const canvas = this.canvas;
    const projection = worldProjection(snapshot.world, canvas.width, canvas.height, presentation.view);
    const environment = this.environment.update(presentation.environment, presentation.identity,
      snapshot.world, presentation.layer, snapshot.epoch, presentation.generation);
    if (environment && this.environmentKey !== environment.key) {
      this.environmentCanvas ??= typeof OffscreenCanvas === "function"
        ? new OffscreenCanvas(environment.width, environment.height) : document.createElement("canvas");
      this.environmentCanvas.width = environment.width;
      this.environmentCanvas.height = environment.height;
      const context = this.environmentCanvas.getContext("2d");
      if (!context) throw new Error("Environment canvas is unavailable");
      context.putImageData(new ImageData(environment.pixels, environment.width, environment.height), 0, 0);
      this.environmentKey = environment.key;
    }
    ctx.fillStyle = this.clearColor;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    if (environment) {
      ctx.imageSmoothingEnabled = false;
      visitWorldTiles(projection, (x, y, width, height) => {
        ctx.drawImage(this.environmentCanvas, x, y,
          width * environment.coverage[0], height * environment.coverage[1]);
      });
    }

    for (const agent of snapshot.agents) {
      const radius = agentRadius(agent) * projection.zoom;
      const selected = typeof presentation.selectedUid === "string" && agentUid(agent) === presentation.selectedUid;
      visitWorldCopies(agent.position, projection, radius + 5.5, (px, py) => {
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
        if (selected) {
          ctx.beginPath();
          ctx.strokeStyle = "rgb(251, 191, 36)";
          ctx.lineWidth = 2;
          ctx.arc(px, py, radius + 4.5, 0, Math.PI * 2);
          ctx.stroke();
        }
      });
    }
    return true;
  }

  destroy() {
    // Nothing to clean up for the Canvas backend.
  }
}
