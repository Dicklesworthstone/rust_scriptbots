import { EnvironmentRaster } from "./environment.js";

const ENVIRONMENT_WGSL = /* wgsl */ `
@group(0) @binding(0) var grid: texture_2d<f32>;
@group(0) @binding(1) var<uniform> coverage: vec4<f32>;
struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
}
@vertex fn vs(@builtin(vertex_index) index: u32) -> VertexOutput {
  let positions = array<vec2<f32>, 3>(vec2<f32>(0.0, 0.0), vec2<f32>(2.0, 0.0), vec2<f32>(0.0, 2.0));
  var output: VertexOutput;
  output.uv = positions[index];
  output.position = vec4<f32>(output.uv.x * 2.0 - 1.0, 1.0 - output.uv.y * 2.0, 0.0, 1.0);
  return output;
}
@fragment fn fs(input: VertexOutput) -> @location(0) vec4<f32> {
  if (input.uv.x >= coverage.x || input.uv.y >= coverage.y) {
    return vec4<f32>(0.043, 0.066, 0.126, 1.0);
  }
  let size = textureDimensions(grid);
  let cell = min(vec2<u32>(input.uv / coverage.xy * vec2<f32>(size)), size - vec2<u32>(1u));
  return textureLoad(grid, vec2<i32>(cell), 0);
}
`;

const WGSL_SOURCE = /* wgsl */ `
struct ViewUniforms {
    canvas : vec2<f32>,
    padding : vec2<f32>,
}

struct Agent {
    position : vec2<f32>,
    radius : f32,
    pad0 : f32,
    color : vec4<f32>,
}

@group(0) @binding(0)
var<uniform> view : ViewUniforms;

@group(0) @binding(1)
var<storage, read> agents : array<Agent>;

struct VertexOutput {
    @builtin(position) position : vec4<f32>,
    @location(0) color : vec3<f32>,
    @location(1) offset : vec2<f32>,
    @location(2) @interpolate(flat) radius : f32,
    @location(3) @interpolate(flat) boost : f32,
}

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
    let extent = agent.radius + select(0.0, 2.6, agent.color.a > 0.5);
    let offset = corner * extent;
    let world = agent.position + offset;
    let clip_x = (world.x / view.canvas.x) * 2.0 - 1.0;
    let clip_y = 1.0 - (world.y / view.canvas.y) * 2.0;

    var output : VertexOutput;
    output.position = vec4<f32>(clip_x, clip_y, 0.0, 1.0);
    output.color = agent.color.rgb;
    output.offset = offset;
    output.radius = agent.radius;
    output.boost = agent.color.a;
    return output;
}

@fragment
fn fs(input : VertexOutput) -> @location(0) vec4<f32> {
    let distance = length(input.offset);
    if (distance <= input.radius) {
        return vec4<f32>(input.color, 0.9);
    }
    if (input.boost > 0.5 && abs(distance - input.radius - 2.0) <= 0.6) {
        return vec4<f32>(30.0 / 255.0, 144.0 / 255.0, 1.0, 0.65);
    }
    discard;
    return vec4<f32>(0.0);
}
`;

const FLOAT_BYTES = Float32Array.BYTES_PER_ELEMENT;
const AGENT_STRIDE_FLOATS = 8; // vec4 position/radius + vec4 color
const AGENT_STRIDE_BYTES = AGENT_STRIDE_FLOATS * FLOAT_BYTES;
const AGENT_STRIDE_ALIGNED = Math.ceil(AGENT_STRIDE_BYTES / 16) * 16;

export class WebGpuRenderer {
  #pendingFrame = null;
  #reportFailure;

  static async isSupported() {
    return typeof navigator !== "undefined" && navigator.gpu !== undefined;
  }

  static async create(canvas, onFailure = () => {}) {
    if (!(await WebGpuRenderer.isSupported())) {
      throw new Error("WebGPU not supported in this environment");
    }

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      throw new Error("Unable to acquire GPU adapter");
    }
    const device = await adapter.requestDevice();
    let context = null;
    try {
      context = canvas.getContext("webgpu");
      if (!context) throw new Error("Failed to acquire WebGPU context");

      const format = navigator.gpu.getPreferredCanvasFormat();
      context.configure({ device, format, alphaMode: "opaque" });

      const shaderModule = device.createShaderModule({ code: WGSL_SOURCE });
      const compilation = await shaderModule.getCompilationInfo();
      const shaderErrors = compilation.messages.filter((message) => message.type === "error");
      if (shaderErrors.length > 0) {
        const details = shaderErrors.map((message) => `${message.lineNum}:${message.linePos} ${message.message}`);
        throw new Error(`WebGPU shader: ${details.join("; ")}`);
      }
      const pipeline = await device.createRenderPipelineAsync({
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
        primitive: { topology: "triangle-list" },
      });
      const environmentShader = device.createShaderModule({ code: ENVIRONMENT_WGSL });
      const environmentErrors = (await environmentShader.getCompilationInfo()).messages
        .filter((message) => message.type === "error");
      if (environmentErrors.length) throw new Error(`WebGPU environment shader: ${environmentErrors.map((error) => error.message).join("; ")}`);
      const environmentPipeline = await device.createRenderPipelineAsync({
        layout: "auto",
        vertex: { module: environmentShader, entryPoint: "vs" },
        fragment: { module: environmentShader, entryPoint: "fs", targets: [{ format }] },
        primitive: { topology: "triangle-list" },
      });

      const viewUniformBuffer = device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });

      const renderer = new WebGpuRenderer(canvas, device, context, pipeline, viewUniformBuffer, format, onFailure, environmentPipeline);
      renderer.updateViewUniform();
      return renderer;
    } catch (error) {
      context?.unconfigure();
      device.destroy();
      throw error;
    }
  }

  constructor(canvas, device, context, pipeline, viewUniformBuffer, format, onFailure, environmentPipeline) {
    this.canvas = canvas;
    this.device = device;
    this.context = context;
    this.pipeline = pipeline;
    this.viewUniformBuffer = viewUniformBuffer;
    this.format = format;
    this.environmentPipeline = environmentPipeline;
    this.environment = new EnvironmentRaster();
    this.environmentTexture = null;
    this.environmentSize = "";
    this.environmentKey = null;
    this.environmentGroup = null;
    this.environmentUniform = device.createBuffer({ size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.disposed = false;
    this.failure = null;
    const fail = (reason) => {
      if (this.disposed || this.failure) return;
      this.failure = reason;
      onFailure(reason);
    };
    this.#reportFailure = fail;
    device.lost.then((info) => fail(`WebGPU device lost: ${info.message || info.reason}`));
    device.addEventListener("uncapturederror", (event) => {
      event.preventDefault();
      fail(`WebGPU error: ${event.error.message}`);
    });

    this.agentCapacity = 0;
    this.agentArray = new Float32Array();
    this.agentBuffer = null;
    this.bindGroup = null;
  }

  updateViewUniform() {
    const data = new Float32Array([this.canvas.width, this.canvas.height, 0, 0]);
    this.device.queue.writeBuffer(
      this.viewUniformBuffer,
      0,
      data.buffer,
      data.byteOffset,
      data.byteLength,
    );
  }

  ensureAgentCapacity(count) {
    const maxAgents = Math.floor(Math.min(
      this.device.limits.maxBufferSize,
      this.device.limits.maxStorageBufferBindingSize,
    ) / AGENT_STRIDE_ALIGNED);
    if (!Number.isSafeInteger(count) || count < 0 || count > maxAgents) {
      throw new RangeError(`Agent count ${count} exceeds WebGPU capacity ${maxAgents}`);
    }
    if (count <= this.agentCapacity) return;
    let capacity = Math.min(maxAgents, this.agentCapacity === 0 ? 256 : this.agentCapacity);
    while (capacity < count) {
      capacity = Math.min(capacity * 2, maxAgents);
    }
    const bufferSize = capacity * AGENT_STRIDE_ALIGNED;
    const agentBuffer = this.device.createBuffer({
      size: bufferSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const bindGroup = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.viewUniformBuffer } },
        { binding: 1, resource: { buffer: agentBuffer } },
      ],
    });
    this.agentBuffer?.destroy();
    this.agentBuffer = agentBuffer;
    this.agentArray = new Float32Array((AGENT_STRIDE_ALIGNED / FLOAT_BYTES) * capacity);
    this.bindGroup = bindGroup;
    this.agentCapacity = capacity;
  }

  render(snapshot, presentation = {}) {
    if (!snapshot) {
      return false;
    }
    if (this.disposed || this.failure) throw new Error(this.failure || "Renderer was destroyed");
    // Refused snapshots are never retained or uploaded. The next callback
    // supplies the then-current snapshot after the existing frame completes.
    if (this.#pendingFrame) return false;

    const environment = this.environment.update(presentation.environment, presentation.identity,
      snapshot.world, presentation.layer, snapshot.epoch, presentation.generation);
    if (environment && this.environmentKey !== environment.key) {
      const size = `${environment.width}x${environment.height}`;
      if (this.environmentSize !== size) {
        this.environmentTexture?.destroy();
        this.environmentTexture = this.device.createTexture({
          size: [environment.width, environment.height], format: "rgba8unorm",
          usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
        this.environmentGroup = this.device.createBindGroup({
          layout: this.environmentPipeline.getBindGroupLayout(0), entries: [
            { binding: 0, resource: this.environmentTexture.createView() },
            { binding: 1, resource: { buffer: this.environmentUniform } },
          ],
        });
        this.environmentSize = size;
      }
      this.device.queue.writeTexture({ texture: this.environmentTexture }, environment.pixels,
        { bytesPerRow: environment.width * 4, rowsPerImage: environment.height },
        { width: environment.width, height: environment.height });
      this.device.queue.writeBuffer(this.environmentUniform, 0, new Float32Array([...environment.coverage, 0, 0]));
      this.environmentKey = environment.key;
    }

    this.updateViewUniform();
    const agents = snapshot.agents;
    this.ensureAgentCapacity(Math.max(agents.length, 1));

    const floatsPerAgent = AGENT_STRIDE_ALIGNED / FLOAT_BYTES;
    const scaleX = this.canvas.width / snapshot.world.width;
    const scaleY = this.canvas.height / snapshot.world.height;
    for (let i = 0; i < agents.length; i += 1) {
      const base = i * floatsPerAgent;
      const agent = agents[i];
      const radius = Math.max(1.5, Math.sqrt(agent.health + 0.15) * 3.5);
      this.agentArray[base + 0] = agent.position[0] * scaleX;
      this.agentArray[base + 1] = agent.position[1] * scaleY;
      this.agentArray[base + 2] = radius;
      this.agentArray[base + 3] = 0.0;
      this.agentArray[base + 4] = agent.color[0];
      this.agentArray[base + 5] = agent.color[1];
      this.agentArray[base + 6] = agent.color[2];
      this.agentArray[base + 7] = agent.boost ? 1.0 : 0.0;
    }

    const byteLength = agents.length * AGENT_STRIDE_ALIGNED;
    if (byteLength) this.device.queue.writeBuffer(this.agentBuffer, 0, this.agentArray.buffer, 0, byteLength);

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
    if (environment) {
      pass.setPipeline(this.environmentPipeline);
      pass.setBindGroup(0, this.environmentGroup);
      pass.draw(3);
    }
    if (agents.length) {
      pass.setPipeline(this.pipeline);
      pass.setBindGroup(0, this.bindGroup);
      pass.draw(6, agents.length, 0, 0);
    }
    pass.end();
    this.#submitFrame(encoder.finish());
    return true;
  }

  #submitFrame(commandBuffer) {
    this.device.queue.submit([commandBuffer]);
    this.#pendingFrame = this.device.queue.onSubmittedWorkDone();
    this.#pendingFrame.then(
      () => { this.#pendingFrame = null; },
      (error) => {
        this.#pendingFrame = null;
        this.#reportFailure(`WebGPU completion failed: ${error.message}`);
      },
    );
  }

  #clearFrame() {
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
    this.#submitFrame(encoder.finish());
  }

  destroy() {
    if (this.disposed) return;
    this.disposed = true;
    this.agentBuffer?.destroy();
    this.environmentTexture?.destroy();
    this.environmentUniform.destroy();
    this.viewUniformBuffer.destroy();
    this.context.unconfigure();
    this.device.destroy();
  }
}
