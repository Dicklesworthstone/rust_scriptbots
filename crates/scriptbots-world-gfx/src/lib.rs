#![forbid(unsafe_code)]

use bytemuck::{Pod, Zeroable};
use std::time::Instant;
use std::sync::atomic::{AtomicBool, Ordering};

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
    pub elevation: Option<&'a [f32]>, // optional elevation field for slope accents
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct AgentInstance {
    pub position: [f32; 2],
    pub size: f32,
    pub color: [f32; 4],
    pub selection: u32,    // 0=None, 1=Hovered, 2=Selected/Focused
    pub glow: f32,         // 0..1 extra glow (e.g., reproduction/spike)
    pub boost: f32,        // 0..1 boost intensity
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
    cam_scale: f32,
    cam_offset: (f32, f32),
    start_time: Instant,
    post: Option<PostFx>,
    #[cfg(feature = "perf_counters")]
    last_render_ms: f32,
    #[cfg(feature = "perf_counters")]
    last_readback_ms: f32,
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
        let mut terrain = TerrainPipeline::new(&device, format, &view);
        terrain.init_atlas(&device, &queue);
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
            cam_scale: 1.0,
            cam_offset: (0.0, 0.0),
            start_time: Instant::now(),
            post: None,
            #[cfg(feature = "perf_counters")]
            last_render_ms: 0.0,
            #[cfg(feature = "perf_counters")]
            last_readback_ms: 0.0,
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
        // keep time monotonic across resizes
        let elapsed = self.start_time.elapsed().as_secs_f32();
        self.view.update(&self.queue, new_size, elapsed, self.cam_scale, self.cam_offset);
        if let Some(post) = self.post.as_mut() {
            post.resize(&self.device, self.format, new_size);
        }
        Ok(())
    }

    pub fn set_camera(&mut self, scale: f32, offset: (f32, f32)) {
        self.cam_scale = scale;
        self.cam_offset = offset;
    }

    pub fn render(&mut self, snapshot: &WorldSnapshot) -> RenderFrame {
        #[cfg(feature = "perf_counters")]
        let t0 = Instant::now();
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("world.render") });
        // Ensure view uniforms match current viewport, time, and camera
        let elapsed = self.start_time.elapsed().as_secs_f32();
        self.view.update(&self.queue, self.size, elapsed, self.cam_scale, self.cam_offset);
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
        self.terrain.encode(&self.device, &self.queue, &mut encoder, &self.color_view, &self.view, snapshot, self.size, self.cam_scale, self.cam_offset);
        self.agents.encode(&self.device, &self.queue, &mut encoder, &self.color_view, &self.view, snapshot, self.size, self.cam_scale, self.cam_offset);
        // Post‑FX (ACES + vignette; FXAA stub): color_view → post.target
        if self.ensure_post() {
            if let Some(p) = self.post.as_mut() {
                p.run(&self.device, &self.queue, &mut encoder, &self.color_view, self.size);
            }
        }
        self.queue.submit(Some(encoder.finish()));
        #[cfg(feature = "perf_counters")]
        { self.last_render_ms = t0.elapsed().as_secs_f32() * 1000.0; }
        RenderFrame { extent: self.size }
    }

    pub fn copy_to_readback(&mut self, _frame: &RenderFrame) -> Result<(), String> {
        #[cfg(feature = "perf_counters")]
        let t0 = Instant::now();
        let src_tex: &wgpu::Texture = if let Some(post) = self.post.as_ref() { &post.target } else { &self.color };
        self.readback.copy(&self.device, &self.queue, src_tex)
            .map(|_| ())
            .and_then(|_| { #[cfg(feature = "perf_counters")] { self.last_readback_ms = t0.elapsed().as_secs_f32() * 1000.0; } Ok(()) })
    }

    pub fn mapped_rgba(&mut self) -> Option<ReadbackView<'_>> { self.readback.mapped() }

    #[cfg(feature = "perf_counters")]
    pub fn last_timings_ms(&self) -> (f32, f32) { (self.last_render_ms, self.last_readback_ms) }

    fn ensure_post(&mut self) -> bool {
        let enable = wants_post();
        if !enable { return false; }
        if self.post.is_none() {
            self.post = Some(PostFx::new(&self.device, self.format, &self.color_view, self.size));
        }
        true
    }
}

fn create_color(device: &wgpu::Device, format: wgpu::TextureFormat, size: (u32, u32)) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("world.color"),
        size: wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::TEXTURE_BINDING,
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
    mapped: std::sync::Arc<AtomicBool>,
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
        let mk_slot = || ReadbackSlot { buf: mk(), ready: false, mapped: std::sync::Arc::new(AtomicBool::new(false)) };
        let slots = [mk_slot(), mk_slot(), mk_slot()];
        Ok(Self { slots, curr: 0, bytes_per_row, extent })
    }

    pub fn copy(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, color: &wgpu::Texture) -> Result<(), String> {
        let slot = &mut self.slots[self.curr];
        slot.ready = false;
        if slot.mapped.load(Ordering::Relaxed) {
            slot.buf.unmap();
            slot.mapped.store(false, Ordering::Relaxed);
        }
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
                layout: wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(std::num::NonZeroU32::new(self.bytes_per_row).unwrap()), rows_per_image: Some(std::num::NonZeroU32::new(self.extent.1).unwrap()) },
            },
            wgpu::Extent3d { width: self.extent.0, height: self.extent.1, depth_or_array_layers: 1 },
        );
        queue.submit(Some(encoder.finish()));

        // Map asynchronously; mark ready upon success via polling.
        let slice = slot.buf.slice(..);
        let mapped_flag = std::sync::Arc::clone(&slot.mapped);
        slice.map_async(wgpu::MapMode::Read, move |res| {
            if res.is_ok() { mapped_flag.store(true, Ordering::Relaxed); }
        });
        // Non-blocking poll; readiness will be observed next frame via `mapped()`
        device.poll(wgpu::Maintain::Poll);
        // Advance ring pointer
        self.curr = (self.curr + 1) % self.slots.len();
        Ok(())
    }

    pub fn mapped(&mut self) -> Option<ReadbackView<'_>> {
        let prev = (self.curr + self.slots.len() - 1) % self.slots.len();
        let slot = &mut self.slots[prev];
        if !slot.ready && !slot.mapped.load(Ordering::Relaxed) { return None; }
        let slice = slot.buf.slice(..);
        let guard = slice.get_mapped_range();
        slot.ready = true; // latch until consumer takes a view at least once
        Some(ReadbackView { guard, bytes_per_row: self.bytes_per_row, width: self.extent.0, height: self.extent.1 })
    }
}

fn align_256(n: u32) -> u32 { ((n + 255) / 256) * 256 }

// ---------------- View uniforms (viewport size) ----------------

#[cfg(test)]
mod tests {
    use super::align_256;

    #[test]
    fn stride_alignment_is_multiple_of_256() {
        let widths = [1u32, 2, 63, 64, 65, 257, 1023, 1920, 2560, 3840];
        for w in widths {
            let raw = w * 4; // RGBA8 bytes per row without alignment
            let aligned = align_256(raw);
            assert_eq!(aligned % 256, 0, "aligned stride must be a multiple of 256 for width {w}");
            assert!(aligned >= raw, "aligned stride must be >= raw stride");
            assert!(aligned <= raw + 255, "aligned stride must not exceed raw+255");
        }
    }

    #[test]
    fn agent_boost_tint_within_bounds() {
        // approximate the WGSL tint effect: base.rgb + (boost*0.35)*vec3(0.6,0.2,0.0)
        for &boost in &[0.0f32, 0.25, 0.5, 0.75, 1.0] {
            let tint = boost * 0.35;
            let add = [tint * 0.6, tint * 0.2, 0.0];
            let mut base = [0.4f32, 0.5, 0.6];
            for i in 0..3 {
                base[i] = (base[i] + add[i]).clamp(0.0, 1.0);
            }
            assert!((0.0..=1.0).contains(&base[0]));
            assert!((0.0..=1.0).contains(&base[1]));
            assert!((0.0..=1.0).contains(&base[2]));
        }
    }
}

struct ViewUniforms {
    buf: wgpu::Buffer,
    layout: wgpu::BindGroupLayout,
    bg: wgpu::BindGroup,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ViewData { v0: [f32; 4], v1: [f32; 4] }

impl ViewUniforms {
    fn new(device: &wgpu::Device, queue: &wgpu::Queue, size: (u32, u32)) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("view.bg_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
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
        this.update(queue, size, 0.0, 1.0, (0.0, 0.0));
        this
    }

    fn update(&self, queue: &wgpu::Queue, size: (u32, u32), time: f32, scale: f32, offset: (f32, f32)) {
        let v0 = [size.0 as f32, size.1 as f32, time, scale];
        let v1 = [offset.0, offset.1, 0.0, 0.0];
        let data = ViewData { v0, v1 };
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
    grid_cols: u32,
    grid_rows: u32,
    tile_w: u32,
    tile_h: u32,
    atlas_w: u32,
    atlas_h: u32,
    vbuf_capacity_bytes: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TileInstance {
    pos: [f32; 2],
    size: [f32; 2],
    atlas_uv: [f32; 4],
    kind: u32,
    slope: f32,
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
            vertex: wgpu::VertexState { module: &shader, entry_point: "vs_main", compilation_options: Default::default(), buffers: &[wgpu::VertexBufferLayout { array_stride: std::mem::size_of::<TileInstance>() as u64, step_mode: wgpu::VertexStepMode::Instance, attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2, 2 => Float32x4, 3 => Uint32, 4 => Float32] }] },
            fragment: Some(wgpu::FragmentState { module: &shader, entry_point: "fs_main", compilation_options: Default::default(), targets: &[Some(wgpu::ColorTargetState { format: color_format, blend: Some(wgpu::BlendState::ALPHA_BLENDING), write_mask: wgpu::ColorWrites::ALL })] }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleStrip, strip_index_format: None, ..Default::default() },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });
        let vbuf_capacity_bytes = (1024 * std::mem::size_of::<TileInstance>()) as u64;
        let tile_vbuf = device.create_buffer(&wgpu::BufferDescriptor { label: Some("terrain.instances"), size: vbuf_capacity_bytes, usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        // default atlas grid config (3x2 tiles of 64px)
        let grid_cols = 3;
        let grid_rows = 2;
        let tile_w = 64;
        let tile_h = 64;
        let atlas_w = grid_cols * tile_w;
        let atlas_h = grid_rows * tile_h;
        Self { pipeline, sampler, atlas, atlas_view, bg_layout, bg, tile_vbuf, tile_count: 0, grid_cols, grid_rows, tile_w, tile_h, atlas_w, atlas_h, vbuf_capacity_bytes }
    }
    fn init_atlas(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        // Generate a simple 3x2 atlas (DeepWater, ShallowWater, Sand, Grass, Bloom, Rock)
        self.atlas = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("terrain.atlas.real"),
            size: wgpu::Extent3d { width: self.atlas_w, height: self.atlas_h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.atlas_view = self.atlas.create_view(&wgpu::TextureViewDescriptor::default());
        // fill tiles with curated colors
        let mut pixels = vec![0u8; (self.atlas_w * self.atlas_h * 4) as usize];
        let colors: [[u8; 4]; 6] = [
            [18, 98, 189, 255],  // DeepWater
            [38, 140, 220, 255], // ShallowWater
            [219, 180, 117, 255],// Sand
            [90, 140, 64, 255],  // Grass
            [159, 201, 84, 255], // Bloom
            [125, 125, 125, 255],// Rock
        ];
        for row in 0..self.grid_rows {
            for col in 0..self.grid_cols {
                let idx = (row * self.grid_cols + col) as usize;
                let color = colors.get(idx).copied().unwrap_or([255, 255, 255, 255]);
                for y in 0..self.tile_h {
                    for x in 0..self.tile_w {
                        let px = col * self.tile_w + x;
                        let py = row * self.tile_h + y;
                        let offset = ((py * self.atlas_w + px) * 4) as usize;
                        let mut rgba = color;
                        // add gentle vignette/variation for non-water tiles
                        if idx >= 2 {
                            let fx = (x as f32 / self.tile_w as f32 - 0.5).abs();
                            let fy = (y as f32 / self.tile_h as f32 - 0.5).abs();
                            let vignette = (fx.max(fy) * 0.12) as f32;
                            let dim = (1.0 - vignette).clamp(0.85, 1.0);
                            rgba[0] = ((rgba[0] as f32) * dim) as u8;
                            rgba[1] = ((rgba[1] as f32) * dim) as u8;
                            rgba[2] = ((rgba[2] as f32) * dim) as u8;
                        }
                        pixels[offset..offset + 4].copy_from_slice(&rgba);
                    }
                }
            }
        }
        queue.write_texture(
            wgpu::ImageCopyTexture { texture: &self.atlas, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            &pixels,
            wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(std::num::NonZeroU32::new(self.atlas_w * 4).unwrap()), rows_per_image: Some(std::num::NonZeroU32::new(self.atlas_h).unwrap()) },
            wgpu::Extent3d { width: self.atlas_w, height: self.atlas_h, depth_or_array_layers: 1 },
        );
        // refresh bind group to point to the new view
        self.bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("terrain.bg.rebind"),
            layout: &self.bg_layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&self.atlas_view) }, wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) }],
        });
    }

    fn atlas_uv_for(&self, tile_index: u32) -> [f32; 4] {
        // Map palette indices 0..5 to our 3x2 grid
        let idx = (tile_index as usize).min(5);
        let col = (idx % (self.grid_cols as usize)) as u32;
        let row = (idx / (self.grid_cols as usize)) as u32;
        let u0 = (col as f32 * self.tile_w as f32) / self.atlas_w as f32;
        let v0 = (row as f32 * self.tile_h as f32) / self.atlas_h as f32;
        let u1 = ((col + 1) as f32 * self.tile_w as f32) / self.atlas_w as f32;
        let v1 = ((row + 1) as f32 * self.tile_h as f32) / self.atlas_h as f32;
        [u0, v0, u1, v1]
    }

    fn ensure_vbuf_capacity(&mut self, device: &wgpu::Device, needed_bytes: u64) {
        if needed_bytes <= self.vbuf_capacity_bytes { return; }
        let mut cap = self.vbuf_capacity_bytes.max(1024);
        while cap < needed_bytes { cap *= 2; }
        self.tile_vbuf = device.create_buffer(&wgpu::BufferDescriptor { label: Some("terrain.instances.realloc"), size: cap, usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        self.vbuf_capacity_bytes = cap;
    }

    fn encode(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder, view_tex: &wgpu::TextureView, view_uniforms: &ViewUniforms, snapshot: &WorldSnapshot, viewport: (u32, u32), scale: f32, offset: (f32, f32)) {
        // Build tile instances for visible terrain with simple CPU frustum culling.
        let (tw, th) = snapshot.terrain.dims;
        let cell = snapshot.terrain.cell_size as f32;
        let mut staging: Vec<TileInstance> = Vec::with_capacity((tw as usize) * (th as usize));
        let (vp_w, vp_h) = (viewport.0 as f32, viewport.1 as f32);
        let elev_opt = snapshot.terrain.elevation;
        let get_elev = |x: i32, y: i32| -> f32 {
            if let Some(elev) = elev_opt {
                let xi = x.clamp(0, (tw as i32) - 1) as usize;
                let yi = y.clamp(0, (th as i32) - 1) as usize;
                let idx = yi * (tw as usize) + xi;
                elev.get(idx).copied().unwrap_or(0.5)
            } else { 0.0 }
        };
        for y in 0..th as i32 {
            for x in 0..tw as i32 {
                let px = x as f32 * cell;
                let py = y as f32 * cell;
                // Convert to pixel-space for culling using camera
                let min_x_px = px * scale + offset.0;
                let min_y_px = py * scale + offset.1;
                let max_x_px = min_x_px + cell * scale;
                let max_y_px = min_y_px + cell * scale;
                if max_x_px < 0.0 || max_y_px < 0.0 || min_x_px > vp_w || min_y_px > vp_h { continue; }
                let idx = (y as usize) * (tw as usize) + (x as usize);
                let tile_id = snapshot.terrain.tiles.get(idx).copied().unwrap_or(3);
                let uv = self.atlas_uv_for(tile_id);
                // slope via central differences if elevation present
                let slope = if elev_opt.is_some() {
                    let dx = (get_elev(x + 1, y) - get_elev(x - 1, y)) * 0.5;
                    let dy = (get_elev(x, y + 1) - get_elev(x, y - 1)) * 0.5;
                    (dx * dx + dy * dy).sqrt().clamp(0.0, 1.0)
                } else { 0.0 };
                staging.push(TileInstance { pos: [px, py], size: [cell, cell], atlas_uv: uv, kind: tile_id, slope });
            }
        }
        if !staging.is_empty() {
            let needed = (staging.len() * std::mem::size_of::<TileInstance>()) as u64;
            self.ensure_vbuf_capacity(device, needed);
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
  @location(3) kind: u32,
  @location(4) slope: f32,
};

struct VsOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
  @location(1) kind: u32,
  @location(2) slope: f32,
  @location(3) world: vec2<f32>,
};

struct View { v0: vec4<f32>, v1: vec4<f32> }; // v0=(viewport.x,viewport.y,time,scale) v1=(offset_x,offset_y,_,_)
@group(1) @binding(0) var<uniform> view: View;

@vertex
fn vs_main(inst: VsIn, @builtin(vertex_index) vid: u32) -> VsOut {
  var o: VsOut;
  var quad = array<vec2<f32>, 4>(vec2<f32>(0.0,0.0), vec2<f32>(1.0,0.0), vec2<f32>(0.0,1.0), vec2<f32>(1.0,1.0));
  let p = quad[vid];
  let xy = inst.pos + p * inst.size;
  let viewport = view.v0.xy;
  let scale = view.v0.w;
  let offset = view.v1.xy;
  let pos = (inst.pos + p * inst.size) * scale + offset;
  let ndc = vec2<f32>(pos.x / viewport.x * 2.0 - 1.0, 1.0 - (pos.y / viewport.y * 2.0));
  o.pos = vec4<f32>(ndc, 0.0, 1.0);
  o.uv = mix(inst.uv.xy, inst.uv.zw, p);
  o.kind = inst.kind;
  o.slope = inst.slope;
  o.world = inst.pos;
  return o;
}

@group(0) @binding(0) var atlas_tex: texture_2d<f32>;
@group(0) @binding(1) var atlas_smp: sampler;

@fragment
fn fs_main(v: VsOut) -> @location(0) vec4<f32> {
  var base = textureSample(atlas_tex, atlas_smp, v.uv);
  // water shimmer for Deep/Shallow water kinds (0,1)
  if (v.kind <= 1u) {
    let time = view.v0.z;
    let wave = sin((v.uv.x * 40.0 + v.uv.y * 28.0) + time * 2.2);
    let shimmer = 0.04 + 0.06 * wave;
    // tuned caustics: stronger on shallow (kind==1), very subtle on deep (kind==0)
    let ca_s = (sin(v.uv.x * 160.0 + time * 1.7) * sin(v.uv.y * 140.0 + time * 1.5));
    let ca_amp = select(0.01, 0.05, v.kind == 1u);
    let ca = ca_s * ca_amp;
    base.rgb = clamp(base.rgb + vec3<f32>(shimmer + ca), vec3<f32>(0.0), vec3<f32>(1.0));
  } else {
    // slope accents (darken proportionally)
    let darken = clamp(1.0 - v.slope * 0.35, 0.0, 1.0);
    base.rgb *= vec3<f32>(darken);
  }
  // subtle biome variation for grass/bloom/rock (kinds 3,4,5)
  if (v.kind >= 3u && v.kind <= 5u) {
    // stable hash from world coords -> [-1,1] noise
    let h = fract(sin(dot(v.world, vec2<f32>(12.9898, 78.233))) * 43758.5453);
    let n = (h * 2.0 - 1.0) * 0.06; // +/-6% brightness tweak
    base.rgb = clamp(base.rgb * (1.0 + n), vec3<f32>(0.0), vec3<f32>(1.0));
  }
  return base;
}
"#;

// ---------------- Agent pipeline (instanced sprites with effects) ----------------

struct AgentPipeline {
    pipeline: wgpu::RenderPipeline,
    vbuf: wgpu::Buffer,
    vbuf_capacity_bytes: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AgentInstanceGpu {
    pos: [f32; 2],
    size: f32,
    _pad: f32,
    color: [f32; 4],
    selection: u32,
    glow: f32,
    boost: f32,
    _pad2: [f32; 1],
}

impl AgentPipeline {
    fn new(device: &wgpu::Device, color_format: wgpu::TextureFormat, view: &ViewUniforms) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("agents.wgsl"), source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(AGENTS_WGSL)) });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("agents.layout"), bind_group_layouts: &[&view.layout], push_constant_ranges: &[] });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("agents.pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState { module: &shader, entry_point: "vs_main", compilation_options: Default::default(), buffers: &[wgpu::VertexBufferLayout { array_stride: std::mem::size_of::<AgentInstanceGpu>() as u64, step_mode: wgpu::VertexStepMode::Instance, attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32, 2 => Float32, 3 => Float32x4, 4 => Uint32, 5 => Float32, 6 => Float32] }] },
            fragment: Some(wgpu::FragmentState { module: &shader, entry_point: "fs_main", compilation_options: Default::default(), targets: &[Some(wgpu::ColorTargetState { format: color_format, blend: Some(wgpu::BlendState::ALPHA_BLENDING), write_mask: wgpu::ColorWrites::ALL })] }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleStrip, ..Default::default() },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });
        let vbuf_capacity_bytes = (1024 * std::mem::size_of::<AgentInstanceGpu>()) as u64;
        let vbuf = device.create_buffer(&wgpu::BufferDescriptor { label: Some("agents.instances"), size: vbuf_capacity_bytes, usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        Self { pipeline, vbuf, vbuf_capacity_bytes }
    }

    fn ensure_vbuf_capacity(&mut self, device: &wgpu::Device, needed_bytes: u64) {
        if needed_bytes <= self.vbuf_capacity_bytes { return; }
        let mut cap = self.vbuf_capacity_bytes.max(1024);
        while cap < needed_bytes { cap *= 2; }
        self.vbuf = device.create_buffer(&wgpu::BufferDescriptor { label: Some("agents.instances.realloc"), size: cap, usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        self.vbuf_capacity_bytes = cap;
    }

    fn encode(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder, view_tex: &wgpu::TextureView, view_uniforms: &ViewUniforms, snapshot: &WorldSnapshot, viewport: (u32, u32), scale: f32, offset: (f32, f32)) {
        let mut staging: Vec<AgentInstanceGpu> = Vec::with_capacity(snapshot.agents.len());
        let (vp_w, vp_h) = (viewport.0 as f32, viewport.1 as f32);
        for a in snapshot.agents {
            // CPU frustum culling (pixel-space); assumes positions/sizes are pixels in this pass
            let half = a.size * 0.5;
            let cx = a.position[0] * scale + offset.0;
            let cy = a.position[1] * scale + offset.1;
            let radius = half * scale;
            if cx + radius < 0.0 || cx - radius > vp_w || cy + radius < 0.0 || cy - radius > vp_h {
                continue;
            }
            staging.push(AgentInstanceGpu { pos: a.position, size: a.size, _pad: 0.0, color: a.color, selection: a.selection, glow: a.glow, boost: a.boost, _pad2: [0.0] });
        }
        if !staging.is_empty() {
            let needed = (staging.len() * std::mem::size_of::<AgentInstanceGpu>()) as u64;
            self.ensure_vbuf_capacity(device, needed);
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
  @location(4) selection: u32,
  @location(5) glow: f32,
  @location(6) boost: f32,
};

struct VsOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) color: vec4<f32>,
  @location(1) local: vec2<f32>,
  @location(2) radius: f32,
  @location(3) selection: u32,
  @location(4) glow: f32,
  @location(5) boost: f32,
};

struct View { v0: vec4<f32>, v1: vec4<f32> }; // v0=(viewport.x,viewport.y,time,scale) v1=(offset_x,offset_y,_,_)
@group(0) @binding(0) var<uniform> view: View;

@vertex
fn vs_main(inst: InInst, @builtin(vertex_index) vid: u32) -> VsOut {
  var o: VsOut;
  var quad = array<vec2<f32>, 4>(vec2<f32>(-0.5,-0.5), vec2<f32>(0.5,-0.5), vec2<f32>(-0.5,0.5), vec2<f32>(0.5,0.5));
  let l = quad[vid];
  let viewport = view.v0.xy;
  let scale = view.v0.w;
  let offset = view.v1.xy;
  let world = (inst.pos + l * inst.size) * scale + offset;
  let ndc = vec2<f32>(world.x / viewport.x * 2.0 - 1.0, 1.0 - (world.y / viewport.y * 2.0));
  o.pos = vec4<f32>(ndc, 0.0, 1.0);
  o.color = inst.color;
  o.local = l;          // range [-0.5, 0.5]
  o.radius = inst.size; // pixel radius
  o.selection = inst.selection;
  o.glow = inst.glow;
  o.boost = inst.boost;
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
  let sel_glow = select(0.0, 0.25, v.selection == 1u) + select(0.0, 0.45, v.selection == 2u);
  let boost_tint = v.boost * 0.35;
  // apply subtle boost tint towards warm color when boosting
  base.rgb = clamp(base.rgb + vec3<f32>(boost_tint * 0.6, boost_tint * 0.2, 0.0), vec3<f32>(0.0), vec3<f32>(1.0));
  let glow = max(sel_glow, v.glow);
  let rim_col = vec4<f32>(1.0, 1.0, 1.0, glow + v.boost * 0.2) * rim;
  return clamp(base + rim_col, vec4<f32>(0.0), vec4<f32>(1.0));
}
"#;

struct PostFx {
    // Final composite (tonemap + vignette + fog + bloom composite)
    pipeline: wgpu::RenderPipeline,
    sampler: wgpu::Sampler,
    src_layout: wgpu::BindGroupLayout, // src color + sampler + bloom texture
    src_bg: wgpu::BindGroup,
    params_layout: wgpu::BindGroupLayout,
    params_bg: wgpu::BindGroup,
    params_buf: wgpu::Buffer,
    target: wgpu::Texture,
    target_view: wgpu::TextureView,
    color_format: wgpu::TextureFormat,

    // Bloom: extract brights to half-res, separable blur (ping-pong)
    bloom_extract_pipeline: wgpu::RenderPipeline,
    bloom_blur_pipeline: wgpu::RenderPipeline,
    bloom_src_layout: wgpu::BindGroupLayout, // single texture + sampler
    bloom_src_bg: wgpu::BindGroup,
    blur_params_layout: wgpu::BindGroupLayout,
    blur_params_bg: wgpu::BindGroup,
    blur_params_buf: wgpu::Buffer,
    bloom_a: Option<wgpu::Texture>,
    bloom_a_view: Option<wgpu::TextureView>,
    bloom_b: Option<wgpu::Texture>,
    bloom_b_view: Option<wgpu::TextureView>,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PostParams {
    exposure: f32,
    vignette: f32,
    tonemap: u32,
    fxaa: u32,
    bloom_thresh: f32,
    bloom_intensity: f32,
    fog_density: f32,
    fog_enabled: u32,
    fog_color: [f32; 3],
    _pad0: f32,
}

impl PostFx {
    fn new(device: &wgpu::Device, format: wgpu::TextureFormat, src_view: &wgpu::TextureView, size: (u32, u32)) -> Self {
        // Final composite shader
        let post_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("postfx.wgsl"), source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(POST_WGSL)) });
        // Final pass samples: src color (binding 0), sampler (1), bloom blurred (2)
        let src_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("post.src_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None },
            ],
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor { label: Some("post.sampler"), mag_filter: wgpu::FilterMode::Linear, min_filter: wgpu::FilterMode::Linear, ..Default::default() });
        // Placeholder bloom view: use source for now; real bloom bound during run()
        let src_bg = device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("post.src_bg"), layout: &src_layout, entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(src_view) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&sampler) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(src_view) },
        ] });
        let params_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("post.params_layout"),
            entries: &[wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: std::num::NonZeroU64::new(std::mem::size_of::<PostParams>() as u64) }, count: None }],
        });
        let params_buf = device.create_buffer(&wgpu::BufferDescriptor { label: Some("post.params_buf"), size: std::mem::size_of::<PostParams>() as u64, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let params_bg = device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("post.params_bg"), layout: &params_layout, entries: &[wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() }] });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("post.layout"), bind_group_layouts: &[&src_layout, &params_layout], push_constant_ranges: &[] });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("post.pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState { module: &post_shader, entry_point: "vs_fullscreen", compilation_options: Default::default(), buffers: &[] },
            fragment: Some(wgpu::FragmentState { module: &post_shader, entry_point: "fs_post", compilation_options: Default::default(), targets: &[Some(wgpu::ColorTargetState { format, blend: Some(wgpu::BlendState::REPLACE), write_mask: wgpu::ColorWrites::ALL })] }),
            primitive: wgpu::PrimitiveState::default(), depth_stencil: None, multisample: wgpu::MultisampleState::default(), multiview: None,
        });
    let (target, target_view) = create_color(device, format, size);

        // Bloom shaders and layouts
        let bloom_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("bloom.wgsl"), source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(BLOOM_WGSL)) });
        let bloom_src_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bloom.src_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
            ],
        });
        let bloom_src_bg = device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("bloom.src_bg"), layout: &bloom_src_layout, entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(src_view) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&sampler) },
        ] });
        let blur_params_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bloom.blur_params_layout"),
            entries: &[wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: std::num::NonZeroU64::new(16) }, count: None }],
        });
        let blur_params_buf = device.create_buffer(&wgpu::BufferDescriptor { label: Some("bloom.blur_params_buf"), size: 16, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let blur_params_bg = device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("bloom.blur_params_bg"), layout: &blur_params_layout, entries: &[wgpu::BindGroupEntry { binding: 0, resource: blur_params_buf.as_entire_binding() }] });
        // Extract pipeline: src -> bloom_a (half res)
        let bloom_extract_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("bloom.extract.layout"), bind_group_layouts: &[&bloom_src_layout, &params_layout], push_constant_ranges: &[] });
        let bloom_extract_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("bloom.extract.pipeline"),
            layout: Some(&bloom_extract_layout),
            vertex: wgpu::VertexState { module: &bloom_shader, entry_point: "vs_fullscreen", compilation_options: Default::default(), buffers: &[] },
            fragment: Some(wgpu::FragmentState { module: &bloom_shader, entry_point: "fs_extract", compilation_options: Default::default(), targets: &[Some(wgpu::ColorTargetState { format, blend: Some(wgpu::BlendState::REPLACE), write_mask: wgpu::ColorWrites::ALL })] }),
            primitive: wgpu::PrimitiveState::default(), depth_stencil: None, multisample: wgpu::MultisampleState::default(), multiview: None,
        });
        // Blur pipeline: bloom_a <-> bloom_b
        let bloom_blur_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("bloom.blur.layout"), bind_group_layouts: &[&bloom_src_layout, &blur_params_layout], push_constant_ranges: &[] });
        let bloom_blur_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("bloom.blur.pipeline"),
            layout: Some(&bloom_blur_layout),
            vertex: wgpu::VertexState { module: &bloom_shader, entry_point: "vs_fullscreen", compilation_options: Default::default(), buffers: &[] },
            fragment: Some(wgpu::FragmentState { module: &bloom_shader, entry_point: "fs_blur", compilation_options: Default::default(), targets: &[Some(wgpu::ColorTargetState { format, blend: Some(wgpu::BlendState::REPLACE), write_mask: wgpu::ColorWrites::ALL })] }),
            primitive: wgpu::PrimitiveState::default(), depth_stencil: None, multisample: wgpu::MultisampleState::default(), multiview: None,
        });

        Self {
            pipeline,
            sampler,
            src_layout,
            src_bg,
            params_layout,
            params_bg,
            params_buf,
            target,
            target_view,
            color_format: format,
            bloom_extract_pipeline,
            bloom_blur_pipeline,
            bloom_src_layout,
            bloom_src_bg,
            blur_params_layout,
            blur_params_bg,
            blur_params_buf,
            bloom_a: None,
            bloom_a_view: None,
            bloom_b: None,
            bloom_b_view: None,
        }
    }

    fn resize(&mut self, device: &wgpu::Device, format: wgpu::TextureFormat, size: (u32, u32)) {
        let (t, v) = create_color(device, format, size);
        self.target = t;
        self.target_view = v;
        self.color_format = format;
        // Drop bloom targets; will be recreated lazily on next run
        self.bloom_a = None;
        self.bloom_a_view = None;
        self.bloom_b = None;
        self.bloom_b_view = None;
    }

    fn rebind(&mut self, device: &wgpu::Device, src_view: &wgpu::TextureView, bloom_view: &wgpu::TextureView) {
        self.src_bg = device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("post.src_bg.rebind"), layout: &self.src_layout, entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(src_view) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(bloom_view) },
        ] });
    }

    fn ensure_bloom_targets(&mut self, device: &wgpu::Device, format: wgpu::TextureFormat, full: (u32, u32)) {
        if self.bloom_a.is_some() && self.bloom_b.is_some() { return; }
        let half = (full.0.max(1) / 2).max(1);
        let half_h = (full.1.max(1) / 2).max(1);
        let make = |label: &str| device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: half, height: half_h, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
            format, usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING, view_formats: &[],
        });
        let a = make("post.bloom.a");
        let b = make("post.bloom.b");
        let av = a.create_view(&wgpu::TextureViewDescriptor::default());
        let bv = b.create_view(&wgpu::TextureViewDescriptor::default());
        self.bloom_a = Some(a);
        self.bloom_a_view = Some(av);
        self.bloom_b = Some(b);
        self.bloom_b_view = Some(bv);
    }

    fn bind_bloom_src(&mut self, device: &wgpu::Device, src_view: &wgpu::TextureView) {
        self.bloom_src_bg = device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("bloom.src_bg.rebind"), layout: &self.bloom_src_layout, entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(src_view) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
        ] });
    }

    fn run(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder, src: &wgpu::TextureView, size: (u32, u32)) {
        // Env knobs
        let exposure = std::env::var("SB_WGPU_EXPOSURE").ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(1.0);
        let vignette = std::env::var("SB_WGPU_VIGNETTE").ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(0.08);
        let tonemap = match std::env::var("SB_WGPU_TONEMAP").ok().as_deref() { Some("filmic") => 1u32, Some("reinhard") => 2u32, _ => 0u32 };
        let fxaa = std::env::var("SB_WGPU_FXAA").ok().and_then(|v| v.parse::<u32>().ok()).unwrap_or(0);
        let bloom_on = std::env::var("SB_WGPU_BLOOM").ok().and_then(|v| v.parse::<u32>().ok()).unwrap_or(0) != 0;
        let bloom_thresh = std::env::var("SB_WGPU_BLOOM_THRESH").ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(0.8);
        let bloom_intensity = std::env::var("SB_WGPU_BLOOM_INTENSITY").ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(0.65);
        let fog_mode = std::env::var("SB_WGPU_FOG").ok().unwrap_or_else(|| "off".to_string());
        let fog_enabled = match fog_mode.as_str() { "low" | "med" | "high" => 1u32, _ => 0u32 };
        let fog_density = match fog_mode.as_str() { "low" => 0.6, "med" => 1.0, "high" => 1.6, _ => 0.0 };
        let fog_color = std::env::var("SB_WGPU_FOG_COLOR").ok().and_then(|v| {
            let parts: Vec<_> = v.split(',').collect();
            if parts.len() == 3 { Some([
                parts[0].trim().parse::<f32>().unwrap_or(0.6),
                parts[1].trim().parse::<f32>().unwrap_or(0.7),
                parts[2].trim().parse::<f32>().unwrap_or(0.8),
            ]) } else { None }
        }).unwrap_or([0.6, 0.7, 0.8]);

        let params = PostParams { exposure, vignette, tonemap, fxaa, bloom_thresh, bloom_intensity, fog_density, fog_enabled, fog_color, _pad0: 0.0 };
        queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));

        // Bloom pass chain if enabled
        let mut bloom_view_opt: Option<wgpu::TextureView> = None;
        if bloom_on {
            self.ensure_bloom_targets(device, self.color_format, size);
            // Create fresh local views to avoid borrowing self across mutable calls
            let a_view_local = self.bloom_a.as_ref().unwrap().create_view(&wgpu::TextureViewDescriptor::default());
            let b_view_local = self.bloom_b.as_ref().unwrap().create_view(&wgpu::TextureViewDescriptor::default());
            // Extract brights from src -> A
            self.bind_bloom_src(device, src);
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("bloom.extract"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: &a_view_local, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store } })],
                    depth_stencil_attachment: None, occlusion_query_set: None, timestamp_writes: None,
                });
                pass.set_pipeline(&self.bloom_extract_pipeline);
                pass.set_bind_group(0, &self.bloom_src_bg, &[]);
                pass.set_bind_group(1, &self.params_bg, &[]);
                pass.draw(0..3, 0..1);
            }
            // Blur A -> B (horizontal)
            let dir_h: [f32; 4] = [1.0 / (size.0.max(1) as f32 * 0.5), 0.0, 0.0, 0.0];
            queue.write_buffer(&self.blur_params_buf, 0, bytemuck::bytes_of(&dir_h));
            self.bind_bloom_src(device, &a_view_local);
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("bloom.blur.h"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: &b_view_local, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store } })],
                    depth_stencil_attachment: None, occlusion_query_set: None, timestamp_writes: None,
                });
                pass.set_pipeline(&self.bloom_blur_pipeline);
                pass.set_bind_group(0, &self.bloom_src_bg, &[]);
                pass.set_bind_group(1, &self.blur_params_bg, &[]);
                pass.draw(0..3, 0..1);
            }
            // Blur B -> A (vertical)
            let dir_v: [f32; 4] = [0.0, 1.0 / (size.1.max(1) as f32 * 0.5), 0.0, 0.0];
            queue.write_buffer(&self.blur_params_buf, 0, bytemuck::bytes_of(&dir_v));
            self.bind_bloom_src(device, &b_view_local);
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("bloom.blur.v"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: &a_view_local, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store } })],
                    depth_stencil_attachment: None, occlusion_query_set: None, timestamp_writes: None,
                });
                pass.set_pipeline(&self.bloom_blur_pipeline);
                pass.set_bind_group(0, &self.bloom_src_bg, &[]);
                pass.set_bind_group(1, &self.blur_params_bg, &[]);
                pass.draw(0..3, 0..1);
            }
            bloom_view_opt = Some(a_view_local);
        }

        // Final composite: bind src + bloom (or src placeholder) and draw
        let bloom_view_ref = bloom_view_opt.as_ref().unwrap_or(src);
        self.rebind(device, src, bloom_view_ref);
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("post.pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: &self.target_view, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store } })],
            depth_stencil_attachment: None, occlusion_query_set: None, timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.src_bg, &[]);
        pass.set_bind_group(1, &self.params_bg, &[]);
        pass.draw(0..3, 0..1);
    }
}

fn wants_post() -> bool {
    std::env::var("SB_WGPU_FXAA").ok().and_then(|v| v.parse::<u32>().ok()).unwrap_or(1) != 0
        || std::env::var("SB_WGPU_TONEMAP").ok().map(|v| !v.is_empty()).unwrap_or(true)
        || std::env::var("SB_WGPU_BLOOM").ok().and_then(|v| v.parse::<u32>().ok()).unwrap_or(0) != 0
        || matches!(std::env::var("SB_WGPU_FOG").ok().as_deref(), Some("low") | Some("med") | Some("high"))
}

const POST_WGSL: &str = r#"
@vertex
fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
  // Fullscreen triangle
  let x = f32((vid << 1u) & 2u);
  let y = f32(vid & 2u);
  return vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
}

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var src_smp: sampler;
@group(0) @binding(2) var bloom_tex: texture_2d<f32>;
struct Params {
  exposure: f32, vignette: f32, tonemap: u32, fxaa: u32,
  bloom_thresh: f32, bloom_intensity: f32, fog_density: f32, fog_enabled: u32,
  fog_color: vec3<f32>, _pad0: f32,
};
@group(1) @binding(0) var<uniform> params: Params;

fn aces_tonemap(c: vec3<f32>) -> vec3<f32> {
  // Fitted ACES curve
  let a = 2.51; let b = 0.03; let d = 0.59; let e = 0.14;
  let numerator = c * (a * c + b);
  let denom = c * ( (a - 1.0) * c + d ) + e;
  return clamp(numerator / denom, vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fs_post(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  let tex_size = vec2<f32>(textureDimensions(src_tex));
  // Pixel coords from builtin position -> normalize to [0,1]
  let uv01 = pos.xy / tex_size;
  var col = textureSampleLevel(src_tex, src_smp, uv01, 0.0);
  // FXAA stub: none (placeholder for later)
  // ACES tonemap + mild vignette
  var rgb = aces_tonemap(col.rgb * params.exposure);
  let p = uv01 * 2.0 - 1.0;
  let vign = clamp(1.0 - dot(p, p) * params.vignette, 0.85, 1.0);
  rgb *= vign;
  // Height-fog (screen-space Y proxy)
  if (params.fog_enabled != 0u) {
    let h = uv01.y; // bottom-heavy fog
    let fog_f = clamp(1.0 - exp(-params.fog_density * h), 0.0, 1.0);
    rgb = mix(rgb, params.fog_color, fog_f);
  }
  // Bloom composite (additive)
  if (params.bloom_intensity > 0.0) {
    let b = textureSampleLevel(bloom_tex, src_smp, uv01, 0.0).rgb;
    rgb = clamp(rgb + b * params.bloom_intensity, vec3<f32>(0.0), vec3<f32>(1.0));
  }
  return vec4<f32>(rgb, 1.0);
}
"#;

// Bloom helpers (extract + separable blur)
const BLOOM_WGSL: &str = r#"
@vertex
fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
  let x = f32((vid << 1u) & 2u);
  let y = f32(vid & 2u);
  return vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
}

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var src_smp: sampler;
struct Params { exposure: f32, vignette: f32, tonemap: u32, fxaa: u32, bloom_thresh: f32, bloom_intensity: f32, fog_density: f32, fog_enabled: u32, fog_color: vec3<f32>, _pad0: f32 };
@group(1) @binding(0) var<uniform> params: Params;

@fragment
fn fs_extract(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  let tex_size = vec2<f32>(textureDimensions(src_tex));
  let uv = pos.xy / tex_size;
  let c = textureSampleLevel(src_tex, src_smp, uv, 0.0).rgb;
  let luma = max(c.r, max(c.g, c.b));
  let m = smoothstep(params.bloom_thresh, params.bloom_thresh + 0.1, luma);
  return vec4<f32>(c * m, 1.0);
}

struct BlurParams { dir: vec2<f32>, _pad: vec2<f32> };
@group(1) @binding(0) var<uniform> blur: BlurParams;

@fragment
fn fs_blur(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  let tex_size = vec2<f32>(textureDimensions(src_tex));
  let uv = pos.xy / tex_size;
  // 5-tap gaussian (weights approximate)
  let w0 = 0.227027;
  let w1 = 0.316216;
  let w2 = 0.070270;
  let off1 = blur.dir * 1.384615;
  let off2 = blur.dir * 3.230769;
  var c = textureSampleLevel(src_tex, src_smp, uv, 0.0).rgb * w0;
  c += textureSampleLevel(src_tex, src_smp, uv + off1, 0.0).rgb * w1;
  c += textureSampleLevel(src_tex, src_smp, uv - off1, 0.0).rgb * w1;
  c += textureSampleLevel(src_tex, src_smp, uv + off2, 0.0).rgb * w2;
  c += textureSampleLevel(src_tex, src_smp, uv - off2, 0.0).rgb * w2;
  return vec4<f32>(c, 1.0);
}
"#;


