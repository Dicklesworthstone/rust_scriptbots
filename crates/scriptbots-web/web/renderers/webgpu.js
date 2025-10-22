const WGSL_SOURCE = /* wgsl */ `
struct ViewUniforms {
    canvas : vec2<f32>,
    padding : vec2<f32>,
};

struct Agent {
    position : vec2<f32>;
    radius : f32;
    pad0 : f32;
    color : vec4<f32>;
};

@group(0) @binding(0)
var<uniform> view : ViewUniforms;

@group(0) @binding(1)
var<storage, read> agents : array<Agent>;

struct VertexOutput {
    @builtin(position) position : vec4<f32>;
    @location(0) color : vec3<f32>;
};

@vertex
fn vs(@builtin(vertex_index) vertexIndex : u32,
      @builtin(instance_index) instanceIndex : u32) -> VertexOutput {
    var positions = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(1.0, 1.0)
    );

    let agent = agents[instanceIndex];
    let corner = positions[vertexIndex];
    let radius = agent.radius;
    let world = agent.position + corner * radius;
    let clip_x = (world.x / view.canvas.x) * 2.0 - 1.0;
    let clip_y = 1.0 - (world.y / view.canvas.y) * 2.0;

    var output : VertexOutput;
    output.position = vec4<f32>(clip_x, clip_y, 0.0, 1.0);
    output.color = agent.color.rgb;
    return output;
}

@fragment
fn fs(input : VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(input.color, 0.95);
}
`;

const FLOAT_BYTES = Float32Array.BYTES_PER_ELEMENT;
const AGENT_STRIDE_FLOATS = 8; // vec4 position/radius + vec4 color
const AGENT_STRIDE_BYTES = AGENT_STRIDE_FLOATS * FLOAT_BYTES;
const AGENT_STRIDE_ALIGNED = Math.ceil(AGENT_STRIDE_BYTES / 16) * 16;

export class WebGpuRenderer {
    static async isSupported() {
        return typeof navigator !== "undefined" && navigator.gpu !== undefined;
    }

    static async create(canvas) {
        if (!(await WebGpuRenderer.isSupported())) {
            throw new Error("WebGPU not supported in this environment");
        }

        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
            throw new Error("Unable to acquire GPU adapter");
        }
        const device = await adapter.requestDevice();
        const context = canvas.getContext("webgpu");
        if (!context) {
            throw new Error("Failed to acquire WebGPU context");
        }

        const format = navigator.gpu.getPreferredCanvasFormat();
        context.configure({
            device,
            format,
            alphaMode: "opaque",
        });

        const shaderModule = device.createShaderModule({ code: WGSL_SOURCE });
        const pipeline = device.createRenderPipeline({
            layout: "auto",
            vertex: {
                module: shaderModule,
                entryPoint: "vs",
            },
            fragment: {
                module: shaderModule,
                entryPoint: "fs",
                targets: [
                    {
                        format,
                        blend: {
                            color: {
                                srcFactor: "src-alpha",
                                dstFactor: "one-minus-src-alpha",
                                operation: "add",
                            },
                            alpha: {
                                srcFactor: "one",
                                dstFactor: "one-minus-src-alpha",
                                operation: "add",
                            },
                        },
                    },
                ],
            },
            primitive: {
                topology: "triangle-list",
            },
        });

        const viewUniformBuffer = device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        const renderer = new WebGpuRenderer(canvas, device, context, pipeline, viewUniformBuffer, format);
        renderer.updateViewUniform();
        return renderer;
    }

    constructor(canvas, device, context, pipeline, viewUniformBuffer, format) {
        this.canvas = canvas;
        this.device = device;
        this.context = context;
        this.pipeline = pipeline;
        this.viewUniformBuffer = viewUniformBuffer;
        this.format = format;

        this.agentCapacity = 0;
        this.agentArray = new Float32Array();
        this.agentBuffer = null;
        this.bindGroup = null;
    }

    updateViewUniform() {
        const data = new Float32Array([this.canvas.width, this.canvas.height, 0, 0]);
        this.device.queue.writeBuffer(this.viewUniformBuffer, 0, data.buffer, data.byteOffset, data.byteLength);
    }

    ensureAgentCapacity(count) {
        if (count <= this.agentCapacity) {
            return;
        }
        let capacity = this.agentCapacity === 0 ? 256 : this.agentCapacity;
        while (capacity < count) {
            capacity *= 2;
        }
        const bufferSize = capacity * AGENT_STRIDE_ALIGNED;
        this.agentBuffer = this.device.createBuffer({
            size: bufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.agentArray = new Float32Array((AGENT_STRIDE_ALIGNED / FLOAT_BYTES) * capacity);
        this.bindGroup = this.device.createBindGroup({
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.viewUniformBuffer } },
                { binding: 1, resource: { buffer: this.agentBuffer } },
            ],
        });
        this.agentCapacity = capacity;
    }

    render(snapshot) {
        if (!snapshot) {
            return;
        }

        this.updateViewUniform();
        const agents = snapshot.agents;
        this.ensureAgentCapacity(Math.max(agents.length, 1));

        if (agents.length === 0) {
            this.clearFrame();
            return;
        }

        const floatsPerAgent = AGENT_STRIDE_ALIGNED / FLOAT_BYTES;
        for (let i = 0; i < agents.length; i += 1) {
            const base = i * floatsPerAgent;
            const agent = agents[i];
            const radius = Math.max(1.5, Math.sqrt(agent.health + 0.15) * 3.5);
            this.agentArray[base + 0] = agent.position[0];
            this.agentArray[base + 1] = agent.position[1];
            this.agentArray[base + 2] = radius;
            this.agentArray[base + 3] = 0.0;
            this.agentArray[base + 4] = agent.color[0];
            this.agentArray[base + 5] = agent.color[1];
            this.agentArray[base + 6] = agent.color[2];
            this.agentArray[base + 7] = agent.boost ? 1.0 : 0.0;
        }

        const byteLength = agents.length * AGENT_STRIDE_ALIGNED;
        this.device.queue.writeBuffer(
            this.agentBuffer,
            0,
            this.agentArray.buffer,
            0,
            byteLength,
        );

        const textureView = this.context.getCurrentTexture().createView();
        const encoder = this.device.createCommandEncoder();
        const pass = encoder.beginRenderPass({
            colorAttachments: [
                {
                    view: textureView,
                    clearValue: { r: 0.043, g: 0.066, b: 0.126, a: 1 },
                    loadOp: "clear",
                    storeOp: "store",
                },
            ],
        });
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, this.bindGroup);
        pass.draw(6, agents.length, 0, 0);
        pass.end();
        this.device.queue.submit([encoder.finish()]);
    }

    clearFrame() {
        const textureView = this.context.getCurrentTexture().createView();
        const encoder = this.device.createCommandEncoder();
        const pass = encoder.beginRenderPass({
            colorAttachments: [
                {
                    view: textureView,
                    clearValue: { r: 0.043, g: 0.066, b: 0.126, a: 1 },
                    loadOp: "clear",
                    storeOp: "store",
                },
            ],
        });
        pass.end();
        this.device.queue.submit([encoder.finish()]);
    }

    destroy() {
        this.agentBuffer?.destroy();
        this.viewUniformBuffer.destroy();
    }
}
