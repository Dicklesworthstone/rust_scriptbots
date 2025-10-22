export class CanvasRenderer {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext("2d", { alpha: false, desynchronized: true });
        this.clearColor = "#0b1120";
    }

    render(snapshot) {
        if (!this.ctx || !snapshot) {
            return;
        }
        const ctx = this.ctx;
        const canvas = this.canvas;
        ctx.fillStyle = this.clearColor;
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        const scaleX = canvas.width / snapshot.world.width;
        const scaleY = canvas.height / snapshot.world.height;

        for (const agent of snapshot.agents) {
            const px = agent.position[0] * scaleX;
            const py = agent.position[1] * scaleY;
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

    destroy() {
        // Nothing to clean up for the Canvas backend.
    }
}
