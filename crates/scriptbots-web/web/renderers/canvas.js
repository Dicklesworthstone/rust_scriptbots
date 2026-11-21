import { EnvironmentRaster } from "./environment.js";
import { agentRadius, agentUid } from "../inspector.js";

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
      ctx.drawImage(this.environmentCanvas, 0, 0,
        canvas.width * environment.coverage[0], canvas.height * environment.coverage[1]);
    }

    const scaleX = canvas.width / snapshot.world.width;
    const scaleY = canvas.height / snapshot.world.height;

    for (const agent of snapshot.agents) {
      const px = agent.position[0] * scaleX;
      const py = agent.position[1] * scaleY;
      const radius = agentRadius(agent);

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
      if (typeof presentation.selectedUid === "string" && agentUid(agent) === presentation.selectedUid) {
        ctx.beginPath();
        ctx.strokeStyle = "rgb(251, 191, 36)";
        ctx.lineWidth = 2;
        ctx.arc(px, py, radius + 4.5, 0, Math.PI * 2);
        ctx.stroke();
      }
    }
    return true;
  }

  destroy() {
    // Nothing to clean up for the Canvas backend.
  }
}
