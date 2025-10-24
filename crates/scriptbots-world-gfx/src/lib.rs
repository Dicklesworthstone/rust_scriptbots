#![forbid(unsafe_code)]

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec2, Vec3};
use std::sync::Arc;
use tracing::{info, warn};

/// Public snapshot format the renderer expects. Keep minimal; the app will adapt
/// its internal world snapshot to this view before passing to the renderer.
#[derive(Clone, Debug)]
pub struct WorldSnapshot<'a> {
    pub world_size: (f32, f32),
    pub terrain: TerrainView<'a>,
    pub agents: &'a [AgentInstance],
}

#[derive(Clone, Debug)]
pub struct TerrainView<'a> {
    pub dims: (u32, u32),
    pub cell_size: u32,
    pub tiles: &'a [u32], // index into a tileset palette/atlas (kept simple for MVP)
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct AgentInstance {
    pub position: [f32; 2],
    pub size: f32,
    pub color: [f32; 4],
}

pub struct WorldRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    size: (u32, u32),
    color: wgpu::Texture,
    color_view: wgpu::TextureView,
    format: wgpu::TextureFormat,
    readback: ReadbackRing,
    terrain: TerrainPipeline,
    agents: AgentPipeline,
    view: ViewUniforms,
}

pub struct RenderFrame {
    pub extent: (u32, u32),
}

impl WorldRenderer {
    pub async fn new(adapter: &wgpu::Adapter, size: (u32, u32)) -> Result<Self, String> {
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default(), None)
            .await
            .map_err(|e| format!("wgpu device request failed: {e}"))?;

        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let (color, color_view) = create_color(&device, format, size);
        let readback = ReadbackRing::new(&device, size, format)?;
        let view = ViewUniforms::new(&device, &queue, size);
        let terrain = TerrainPipeline::new(&device, format, &view);
        let agents = AgentPipeline::new(&device, format, &view);

        Ok(Self {
            device,
            queue,
            size,
            color,
            color_view,
            format,
            readback,
            terrain,
            agents,
            view,
        })
    }

    pub fn resize(&mut self, new_size: (u32, u32)) -> Result<(), String> {
        if new_size == self.size || new_size.0 == 0 || new_size.1 == 0 {
            return Ok(());
        }
        let (tex, view) = create_color(&self.device, self.format, new_size);
        self.color = tex;
        self.color_view = view;
        self.size = new_size;
        self.readback = ReadbackRing::new(&self.device, new_size, self.format)?;
        self.view.update(&self.queue, new_size);
        Ok(())
    }

    pub fn render(&mut self, snapshot: &WorldSnapshot) -> RenderFrame {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("world.render") });
        // Ensure view uniforms match current viewport
        self.view.update(&self.queue, self.size);
        // Background clear
        {
            let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("world.clear"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.color_view,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.03, g: 0.06, b: 0.12, a: 1.0 }), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
        }
        // Terrain + agents
        self.terrain.encode(&self.device, &self.queue, &mut encoder, &self.color_view, &self.view, snapshot);
        self.agents.encode(&self.device, &self.queue, &mut encoder, &self.color_view, &self.view, snapshot);
        self.queue.submit(Some(encoder.finish()));
        RenderFrame { extent: self.size }
    }

    pub fn copy_to_readback(&mut self, _frame: &RenderFrame) -> Result<(), String> {
        self.readback.copy(&self.device, &self.queue, &self.color)
    }

    pub fn mapped_rgba(&mut self) -> Option<ReadbackView<'_>> { self.readback.mapped() }
}

fn create_color(device: &wgpu::Device, format: wgpu::TextureFormat, size: (u32, u32)) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("world.color"),
        size: wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}

// ---------------- Readback ring (triple-buffered) ----------------

pub struct ReadbackRing {
    slots: [ReadbackSlot; 3],
    curr: usize,
    bytes_per_row: u32,
    extent: (u32, u32),
}

pub struct ReadbackSlot {
    buf: wgpu::Buffer,
    ready: bool,
}

pub struct ReadbackView<'a> {
    pub guard: wgpu::BufferView<'a>,
    pub bytes_per_row: u32,
    pub width: u32,
    pub height: u32,
}

impl<'a> ReadbackView<'a> {
    pub fn bytes(&self) -> &[u8] {
        &self.guard
    }
}

impl ReadbackRing {
    pub fn new(device: &wgpu::Device, extent: (u32, u32), format: wgpu::TextureFormat) -> Result<Self, String> {
        assert_eq!(format, wgpu::TextureFormat::Rgba8UnormSrgb, "only RGBA8 sRGB supported for readback");
        let bytes_per_row = align_256(extent.0 * 4);
        let size_bytes = bytes_per_row as u64 * extent.1 as u64;
        let mk = || device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("world.readback"),
            size: size_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let slots = [ReadbackSlot { buf: mk(), ready: false }, ReadbackSlot { buf: mk(), ready: false }, ReadbackSlot { buf: mk(), ready: false }];
        Ok(Self { slots, curr: 0, bytes_per_row, extent })
    }

    pub fn copy(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, color: &wgpu::Texture) -> Result<(), String> {
        let slot = &mut self.slots[self.curr];
        slot.ready = false;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("world.readback.copy") });
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: color,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &slot.buf,
                layout: wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(self.bytes_per_row), rows_per_image: Some(self.extent.1) },
            },
            wgpu::Extent3d { width: self.extent.0, height: self.extent.1, depth_or_array_layers: 1 },
        );
        queue.submit(Some(encoder.finish()));

        // Map asynchronously; mark ready upon success via polling.
        let slice = slot.buf.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = sender.send(res.is_ok());
        });
        // Non-blocking poll; the app can call `mapped` later to retrieve the previous ready slot.
        device.poll(wgpu::Maintain::Poll);
        if let Some(Ok(true)) = receiver.try_receive() { slot.ready = true; }
        // Advance ring pointer
        self.curr = (self.curr + 1) % self.slots.len();
        Ok(())
    }

    pub fn mapped(&mut self) -> Option<ReadbackView<'_>> {
        let prev = (self.curr + self.slots.len() - 1) % self.slots.len();
        let slot = &mut self.slots[prev];
        if !slot.ready { return None; }
        let slice = slot.buf.slice(..);
        let guard = slice.get_mapped_range();
        Some(ReadbackView { guard, bytes_per_row: self.bytes_per_row, width: self.extent.0, height: self.extent.1 })
    }
}

fn align_256(n: u32) -> u32 { ((n + 255) / 256) * 256 }

// ---------------- View uniforms (viewport size) ----------------

struct ViewUniforms {
    buf: wgpu::Buffer,
    layout: wgpu::BindGroupLayout,
    bg: wgpu::BindGroup,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ViewData { viewport: [f32; 2], padding: [f32; 2] }

impl ViewUniforms {
    fn new(device: &wgpu::Device, queue: &wgpu::Queue, size: (u32, u32)) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("view.bg_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: std::num::NonZeroU64::new(std::mem::size_of::<ViewData>() as u64),
                },
                count: None,
            }],
        });
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("view.ubuf"),
            size: std::mem::size_of::<ViewData>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("view.bg"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: buf.as_entire_binding() }],
        });
        let this = Self { buf, layout, bg };
        this.update(queue, size);
        this
    }

    fn update(&self, queue: &wgpu::Queue, size: (u32, u32)) {
        let data = ViewData { viewport: [size.0 as f32, size.1 as f32], padding: [0.0, 0.0] };
        queue.write_buffer(&self.buf, 0, bytemuck::bytes_of(&data));
    }
}

// ---------------- Terrain pipeline (instanced tiles with atlas) ----------------

struct TerrainPipeline {
    pipeline: wgpu::RenderPipeline,
    sampler: wgpu::Sampler,
    atlas: wgpu::Texture,
    atlas_view: wgpu::TextureView,
    bg_layout: wgpu::BindGroupLayout,
    bg: wgpu::BindGroup,
    tile_vbuf: wgpu::Buffer,
    tile_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TileInstance {
    pos: [f32; 2],
    size: [f32; 2],
    atlas_uv: [f32; 4],
}

impl TerrainPipeline {
    fn new(device: &wgpu::Device, color_format: wgpu::TextureFormat, view: &ViewUniforms) -> Self {
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("terrain.sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        // 1x1 white atlas placeholder; real atlas supplied later via update
        let atlas = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("terrain.atlas"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let atlas_view = atlas.create_view(&wgpu::TextureViewDescriptor::default());

        let bg_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("terrain.bg_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
                count: None,
            }, wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            }],
        });
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("terrain.bg"),
            layout: &bg_layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&atlas_view) }, wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&sampler) }],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("terrain.wgsl"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(TERRAIN_WGSL)),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("terrain.layout"), bind_group_layouts: &[&bg_layout, &view.layout], push_constant_ranges: &[] });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("terrain.pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState { module: &shader, entry_point: "vs_main", compilation_options: Default::default(), buffers: &[wgpu::VertexBufferLayout { array_stride: std::mem::size_of::<TileInstance>() as u64, step_mode: wgpu::VertexStepMode::Instance, attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2, 2 => Float32x4] }] },
            fragment: Some(wgpu::FragmentState { module: &shader, entry_point: "fs_main", compilation_options: Default::default(), targets: &[Some(wgpu::ColorTargetState { format: color_format, blend: Some(wgpu::BlendState::ALPHA_BLENDING), write_mask: wgpu::ColorWrites::ALL })] }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleStrip, strip_index_format: None, ..Default::default() },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        let tile_vbuf = device.create_buffer(&wgpu::BufferDescriptor { label: Some("terrain.instances"), size: 1024 * std::mem::size_of::<TileInstance>() as u64, usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        Self { pipeline, sampler, atlas, atlas_view, bg_layout, bg, tile_vbuf, tile_count: 0 }
    }

    fn encode(&self, _device: &wgpu::Device, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder, view_tex: &wgpu::TextureView, view_uniforms: &ViewUniforms, snapshot: &WorldSnapshot) {
        // Build tile instances for visible terrain (simple full mesh for now). In a follow‑up commit, add culling.
        let (tw, th) = snapshot.terrain.dims;
        let cell = snapshot.terrain.cell_size as f32;
        let mut staging: Vec<TileInstance> = Vec::with_capacity((tw as usize) * (th as usize));
        for y in 0..th {
            for x in 0..tw {
                let px = x as f32 * cell;
                let py = y as f32 * cell;
                staging.push(TileInstance { pos: [px, py], size: [cell, cell], atlas_uv: [0.0, 0.0, 1.0, 1.0] });
            }
        }
        if !staging.is_empty() {
            queue.write_buffer(&self.tile_vbuf, 0, bytemuck::cast_slice(&staging));
        }
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("terrain.pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: view_tex, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store } })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bg, &[]);
        pass.set_bind_group(1, &view_uniforms.bg, &[]);
        pass.set_vertex_buffer(0, self.tile_vbuf.slice(..));
        pass.draw(0..4, 0..staging.len() as u32);
    }
}

const TERRAIN_WGSL: &str = r#"
struct VsIn {
  @location(0) pos: vec2<f32>,
  @location(1) size: vec2<f32>,
  @location(2) uv: vec4<f32>,
};

struct VsOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

struct View { viewport: vec2<f32>, _pad: vec2<f32> };
@group(1) @binding(0) var<uniform> view: View;

@vertex
fn vs_main(inst: VsIn, @builtin(vertex_index) vid: u32) -> VsOut {
  var o: VsOut;
  var quad = array<vec2<f32>, 4>(vec2<f32>(0.0,0.0), vec2<f32>(1.0,0.0), vec2<f32>(0.0,1.0), vec2<f32>(1.0,1.0));
  let p = quad[vid];
  let xy = inst.pos + p * inst.size;
  let ndc = vec2<f32>(xy.x / view.viewport.x * 2.0 - 1.0, 1.0 - (xy.y / view.viewport.y * 2.0));
  o.pos = vec4<f32>(ndc, 0.0, 1.0);
  o.uv = mix(inst.uv.xy, inst.uv.zw, p);
  return o;
}

@group(0) @binding(0) var atlas_tex: texture_2d<f32>;
@group(0) @binding(1) var atlas_smp: sampler;

@fragment
fn fs_main(v: VsOut) -> @location(0) vec4<f32> {
  return textureSample(atlas_tex, atlas_smp, v.uv);
}
"#;

// ---------------- Agent pipeline (instanced sprites with effects) ----------------

struct AgentPipeline {
    pipeline: wgpu::RenderPipeline,
    vbuf: wgpu::Buffer,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AgentInstanceGpu {
    pos: [f32; 2],
    size: f32,
    _pad: f32,
    color: [f32; 4],
}

impl AgentPipeline {
    fn new(device: &wgpu::Device, color_format: wgpu::TextureFormat, view: &ViewUniforms) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("agents.wgsl"), source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(AGENTS_WGSL)) });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("agents.layout"), bind_group_layouts: &[&view.layout], push_constant_ranges: &[] });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("agents.pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState { module: &shader, entry_point: "vs_main", compilation_options: Default::default(), buffers: &[wgpu::VertexBufferLayout { array_stride: std::mem::size_of::<AgentInstanceGpu>() as u64, step_mode: wgpu::VertexStepMode::Instance, attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32, 2 => Float32, 3 => Float32x4] }] },
            fragment: Some(wgpu::FragmentState { module: &shader, entry_point: "fs_main", compilation_options: Default::default(), targets: &[Some(wgpu::ColorTargetState { format: color_format, blend: Some(wgpu::BlendState::ALPHA_BLENDING), write_mask: wgpu::ColorWrites::ALL })] }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleStrip, ..Default::default() },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });
        let vbuf = device.create_buffer(&wgpu::BufferDescriptor { label: Some("agents.instances"), size: 1024 * std::mem::size_of::<AgentInstanceGpu>() as u64, usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        Self { pipeline, vbuf }
    }

    fn encode(&self, _device: &wgpu::Device, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder, view_tex: &wgpu::TextureView, view_uniforms: &ViewUniforms, snapshot: &WorldSnapshot) {
        let mut staging: Vec<AgentInstanceGpu> = Vec::with_capacity(snapshot.agents.len());
        for a in snapshot.agents {
            staging.push(AgentInstanceGpu { pos: a.position, size: a.size, _pad: 0.0, color: a.color });
        }
        if !staging.is_empty() {
            queue.write_buffer(&self.vbuf, 0, bytemuck::cast_slice(&staging));
        }
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("agents.pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: view_tex, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store } })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &view_uniforms.bg, &[]);
        pass.set_vertex_buffer(0, self.vbuf.slice(..));
        pass.draw(0..4, 0..staging.len() as u32);
    }
}

const AGENTS_WGSL: &str = r#"
struct InInst {
  @location(0) pos: vec2<f32>,
  @location(1) size: f32,
  @location(2) _pad: f32,
  @location(3) color: vec4<f32>,
};

struct VsOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) color: vec4<f32>,
  @location(1) local: vec2<f32>,
  @location(2) radius: f32,
};

@vertex
fn vs_main(inst: InInst, @builtin(vertex_index) vid: u32) -> VsOut {
  var o: VsOut;
  var quad = array<vec2<f32>, 4>(vec2<f32>(-0.5,-0.5), vec2<f32>(0.5,-0.5), vec2<f32>(-0.5,0.5), vec2<f32>(0.5,0.5));
  let l = quad[vid];
  let world = inst.pos + l * inst.size;
  o.pos = vec4<f32>(world, 0.0, 1.0);
  o.color = inst.color;
  o.local = l;          // range [-0.5, 0.5]
  o.radius = inst.size; // pixel radius
  return o;
}

@fragment
fn fs_main(v: VsOut) -> @location(0) vec4<f32> {
  // Signed distance circle (soft edge) with thin rim highlight for a premium look
  let d = length(v.local * 2.0);         // 0 at center, ~1 at edge
  let edge = smoothstep(1.05, 0.95, 1.0 - d);
  let rim = smoothstep(1.02, 0.98, d);   // thin rim near edge
  var base = v.color;
  base.a = base.a * edge;
  let rim_col = vec4<f32>(1.0, 1.0, 1.0, 0.25) * rim;
  return clamp(base + rim_col, vec4<f32>(0.0), vec4<f32>(1.0));
}
"#;


