#![forbid(unsafe_code)]

use bytemuck::{Pod, Zeroable};
use scriptbots_core::NUM_EYES;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

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
    pub quad_extent: [f32; 2],
    pub heading: [f32; 2],
    pub body_radius: f32,
    pub body_half_length: f32,
    pub wheel_offset: f32,
    pub wheel_radius: f32,
    pub mouth_open: f32,
    pub herbivore_tendency: f32,
    pub temperature_preference: f32,
    pub food_delta: f32,
    pub sound_level: f32,
    pub sound_output: f32,
    pub wheel_left: f32,
    pub wheel_right: f32,
    pub spike_length: f32,
    pub trait_smell: f32,
    pub trait_sound: f32,
    pub trait_hearing: f32,
    pub trait_eye: f32,
    pub trait_blood: f32,
    pub selection: f32, // 0=None, 1=Hovered, 2=Selected/Focused
    pub color: [f32; 4],
    pub glow: f32,  // 0..1 extra glow (e.g., reproduction/spike)
    pub boost: f32, // 0..1 boost intensity
    pub spiked: f32,
    pub eye_dirs: [f32; NUM_EYES],
    pub eye_fov: [f32; NUM_EYES],
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
        // Guard against zero-sized viewports (can happen during early window init on some platforms)
        let size = (size.0.max(1), size.1.max(1));
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
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
        self.view.update(
            &self.queue,
            new_size,
            elapsed,
            self.cam_scale,
            self.cam_offset,
        );
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
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("world.render"),
            });
        // Ensure view uniforms match current viewport, time, and camera
        let elapsed = self.start_time.elapsed().as_secs_f32();
        self.view.update(
            &self.queue,
            self.size,
            elapsed,
            self.cam_scale,
            self.cam_offset,
        );
        // Background clear
        {
            let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("world.clear"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.color_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(if env_flag("SB_WGPU_DEBUG_BRIGHT_BG") {
                            wgpu::Color {
                                r: 0.10,
                                g: 0.10,
                                b: 0.25,
                                a: 1.0,
                            }
                        } else {
                            wgpu::Color {
                                r: 0.03,
                                g: 0.06,
                                b: 0.12,
                                a: 1.0,
                            }
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
        }
        // Terrain + agents
        let vis_tiles = self.terrain.encode(
            &self.device,
            &self.queue,
            &mut encoder,
            &self.color_view,
            &self.view,
            snapshot,
            self.size,
            self.cam_scale,
            self.cam_offset,
        );
        let vis_agents = self.agents.encode(
            &self.device,
            &self.queue,
            &mut encoder,
            &self.color_view,
            &self.view,
            snapshot,
            self.size,
            self.cam_scale,
            self.cam_offset,
        );
        tracing::info!(
            tiles = vis_tiles,
            agents = vis_agents,
            "wgpu visible instances"
        );
        // Post‑FX (ACES + vignette; FXAA stub): color_view → post.target
        if self.ensure_post()
            && let Some(p) = self.post.as_mut()
        {
            p.run(
                &self.device,
                &self.queue,
                &mut encoder,
                &self.color_view,
                self.size,
            );
        }
        self.queue.submit(Some(encoder.finish()));
        #[cfg(feature = "perf_counters")]
        {
            self.last_render_ms = t0.elapsed().as_secs_f32() * 1000.0;
        }
        RenderFrame { extent: self.size }
    }

    pub fn copy_to_readback(&mut self, _frame: &RenderFrame) -> Result<(), String> {
        #[cfg(feature = "perf_counters")]
        let t0 = Instant::now();
        let src_tex: &wgpu::Texture = if let Some(post) = self.post.as_ref() {
            &post.target
        } else {
            &self.color
        };
        self.readback
            .copy(&self.device, &self.queue, src_tex)
            .map(|_| {
                #[cfg(feature = "perf_counters")]
                {
                    self.last_readback_ms = t0.elapsed().as_secs_f32() * 1000.0;
                }
            })
    }

    pub fn mapped_rgba(&mut self) -> Option<ReadbackView> {
        self.readback.mapped()
    }

    #[cfg(feature = "perf_counters")]
    pub fn last_timings_ms(&self) -> (f32, f32) {
        (self.last_render_ms, self.last_readback_ms)
    }

    fn ensure_post(&mut self) -> bool {
        let enable = wants_post();
        if !enable {
            return false;
        }
        if self.post.is_none() {
            self.post = Some(PostFx::new(
                &self.device,
                self.format,
                &self.color_view,
                self.size,
            ));
        }
        true
    }
}

#[cfg(test)]
mod capture_smoke_test {
    use super::*;

    // Not a real unit test; handy local harness to write one frame PNG for diagnosis.
    // Run: `cargo test -p scriptbots-world-gfx capture_smoke -- --nocapture`
    #[test]
    fn capture_smoke() {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .expect("adapter");
        let size = (640, 360);
        let mut renderer =
            pollster::block_on(WorldRenderer::new(&adapter, size)).expect("renderer");
        let dims = (120u32, 60u32);
        let tiles = vec![3u32; (dims.0 * dims.1) as usize];
        let snapshot = WorldSnapshot {
            world_size: (6000.0, 3000.0),
            terrain: TerrainView {
                dims,
                cell_size: 50,
                tiles: &tiles,
                elevation: None,
            },
            agents: &[],
        };
        let frame = renderer.render(&snapshot);
        renderer.copy_to_readback(&frame).unwrap();
        if let Some(view) = renderer.mapped_rgba() {
            let row_bytes = (view.width as usize) * 4;
            let mut tight = vec![0u8; row_bytes * (view.height as usize)];
            let src = view.bytes();
            for y in 0..(view.height as usize) {
                let s = y * (view.bytes_per_row as usize);
                let d = y * row_bytes;
                tight[d..d + row_bytes].copy_from_slice(&src[s..s + row_bytes]);
            }
            let _ = image::save_buffer_with_format(
                "wgpu_capture_smoke.png",
                &tight,
                view.width,
                view.height,
                image::ColorType::Rgba8,
                image::ImageFormat::Png,
            );
        }
    }
}

fn create_color(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    size: (u32, u32),
) -> (wgpu::Texture, wgpu::TextureView) {
    // Defensive clamp to ensure valid texture extent
    let size = (size.0.max(1), size.1.max(1));
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("world.color"),
        size: wgpu::Extent3d {
            width: size.0,
            height: size.1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::TEXTURE_BINDING,
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

pub struct ReadbackView {
    pub guard: wgpu::BufferView,
    pub bytes_per_row: u32,
    pub width: u32,
    pub height: u32,
}

impl ReadbackView {
    pub fn bytes(&self) -> &[u8] {
        &self.guard
    }
}

impl ReadbackRing {
    pub fn new(
        device: &wgpu::Device,
        extent: (u32, u32),
        format: wgpu::TextureFormat,
    ) -> Result<Self, String> {
        assert_eq!(
            format,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            "only RGBA8 sRGB supported for readback"
        );
        // Clamp extent to avoid zero-sized buffers which are invalid on some backends
        let extent = (extent.0.max(1), extent.1.max(1));
        let bytes_per_row = align_256(extent.0 * 4);
        let size_bytes = bytes_per_row as u64 * extent.1 as u64;
        let mk = || {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("world.readback"),
                size: size_bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            })
        };
        let mk_slot = || ReadbackSlot {
            buf: mk(),
            ready: false,
            mapped: std::sync::Arc::new(AtomicBool::new(false)),
        };
        let slots = [mk_slot(), mk_slot(), mk_slot()];
        Ok(Self {
            slots,
            curr: 0,
            bytes_per_row,
            extent,
        })
    }

    pub fn copy(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        color: &wgpu::Texture,
    ) -> Result<(), String> {
        let slot = &mut self.slots[self.curr];
        slot.ready = false;
        if slot.mapped.load(Ordering::Relaxed) {
            slot.buf.unmap();
            slot.mapped.store(false, Ordering::Relaxed);
        }
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("world.readback.copy"),
        });
        encoder.copy_texture_to_buffer(
            color.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &slot.buf,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(
                        std::num::NonZeroU32::new(self.bytes_per_row).unwrap().get(),
                    ),
                    rows_per_image: Some(std::num::NonZeroU32::new(self.extent.1).unwrap().get()),
                },
            },
            wgpu::Extent3d {
                width: self.extent.0,
                height: self.extent.1,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(Some(encoder.finish()));

        // Map asynchronously; mark ready upon success via polling.
        let slice = slot.buf.slice(..);
        let mapped_flag = std::sync::Arc::clone(&slot.mapped);
        slice.map_async(wgpu::MapMode::Read, move |res| {
            if res.is_ok() {
                mapped_flag.store(true, Ordering::Relaxed);
            }
        });
        // Ensure progress on mapping; non-blocking is sufficient for our readback ring
        // Non-blocking poll may be insufficient in tests; use indefinite wait
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        // Advance ring pointer
        self.curr = (self.curr + 1) % self.slots.len();
        Ok(())
    }

    pub fn mapped(&mut self) -> Option<ReadbackView> {
        // Prefer the most recently mapped slot (scan last -> older)
        for i in 0..self.slots.len() {
            let idx = (self.curr + self.slots.len() - 1 - i) % self.slots.len();
            let slot = &mut self.slots[idx];
            if !slot.ready && !slot.mapped.load(Ordering::Relaxed) {
                continue;
            }
            let slice = slot.buf.slice(..);
            let guard = slice.get_mapped_range();
            slot.ready = true; // latch until consumer takes a view at least once
            return Some(ReadbackView {
                guard,
                bytes_per_row: self.bytes_per_row,
                width: self.extent.0,
                height: self.extent.1,
            });
        }
        None
    }
}

fn align_256(n: u32) -> u32 {
    n.div_ceil(256) * 256
}

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
            assert_eq!(
                aligned % 256,
                0,
                "aligned stride must be a multiple of 256 for width {w}"
            );
            assert!(aligned >= raw, "aligned stride must be >= raw stride");
            assert!(
                aligned <= raw + 255,
                "aligned stride must not exceed raw+255"
            );
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
struct ViewData {
    v0: [f32; 4],
    v1: [f32; 4],
}

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
                    min_binding_size: std::num::NonZeroU64::new(
                        std::mem::size_of::<ViewData>() as u64
                    ),
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
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buf.as_entire_binding(),
            }],
        });
        let this = Self { buf, layout, bg };
        this.update(queue, size, 0.0, 1.0, (0.0, 0.0));
        this
    }

    fn update(
        &self,
        queue: &wgpu::Queue,
        size: (u32, u32),
        time: f32,
        scale: f32,
        offset: (f32, f32),
    ) {
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
    _tile_count: u32,
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
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
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
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("terrain.bg"),
            layout: &bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&atlas_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("terrain.wgsl"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(TERRAIN_WGSL)),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("terrain.layout"),
            bind_group_layouts: &[&bg_layout, &view.layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("terrain.pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs_main"), compilation_options: Default::default(), buffers: &[wgpu::VertexBufferLayout { array_stride: std::mem::size_of::<TileInstance>() as u64, step_mode: wgpu::VertexStepMode::Instance, attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2, 2 => Float32x4, 3 => Uint32, 4 => Float32] }] },
            fragment: Some(wgpu::FragmentState { module: &shader, entry_point: Some("fs_main"), compilation_options: Default::default(), targets: &[Some(wgpu::ColorTargetState { format: color_format, blend: Some(wgpu::BlendState::ALPHA_BLENDING), write_mask: wgpu::ColorWrites::ALL })] }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleStrip, strip_index_format: None, ..Default::default() },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });
        let vbuf_capacity_bytes = (1024 * std::mem::size_of::<TileInstance>()) as u64;
        let tile_vbuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("terrain.instances"),
            size: vbuf_capacity_bytes,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // default atlas grid config (3x2 tiles of 64px)
        let grid_cols = 3;
        let grid_rows = 2;
        let tile_w = 64;
        let tile_h = 64;
        let atlas_w = grid_cols * tile_w;
        let atlas_h = grid_rows * tile_h;
        Self {
            pipeline,
            sampler,
            atlas,
            atlas_view,
            bg_layout,
            bg,
            tile_vbuf,
            _tile_count: 0,
            grid_cols,
            grid_rows,
            tile_w,
            tile_h,
            atlas_w,
            atlas_h,
            vbuf_capacity_bytes,
        }
    }
    fn init_atlas(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        // Generate a simple 3x2 atlas (DeepWater, ShallowWater, Sand, Grass, Bloom, Rock)
        self.atlas = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("terrain.atlas.real"),
            size: wgpu::Extent3d {
                width: self.atlas_w,
                height: self.atlas_h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.atlas_view = self
            .atlas
            .create_view(&wgpu::TextureViewDescriptor::default());
        // fill tiles with curated colors
        let mut pixels = vec![0u8; (self.atlas_w * self.atlas_h * 4) as usize];
        let colors: [[u8; 4]; 6] = [
            [18, 98, 189, 255],   // DeepWater
            [38, 140, 220, 255],  // ShallowWater
            [219, 180, 117, 255], // Sand
            [90, 140, 64, 255],   // Grass
            [159, 201, 84, 255],  // Bloom
            [125, 125, 125, 255], // Rock
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
                            let vignette = fx.max(fy) * 0.12;
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
            self.atlas.as_image_copy(),
            &pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(std::num::NonZeroU32::new(self.atlas_w * 4).unwrap().get()),
                rows_per_image: Some(std::num::NonZeroU32::new(self.atlas_h).unwrap().get()),
            },
            wgpu::Extent3d {
                width: self.atlas_w,
                height: self.atlas_h,
                depth_or_array_layers: 1,
            },
        );
        // refresh bind group to point to the new view
        self.bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("terrain.bg.rebind"),
            layout: &self.bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.atlas_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
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
        if needed_bytes <= self.vbuf_capacity_bytes {
            return;
        }
        let mut cap = self.vbuf_capacity_bytes.max(1024);
        while cap < needed_bytes {
            cap *= 2;
        }
        self.tile_vbuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("terrain.instances.realloc"),
            size: cap,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.vbuf_capacity_bytes = cap;
    }

    #[allow(clippy::too_many_arguments)]
    fn encode(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        view_tex: &wgpu::TextureView,
        view_uniforms: &ViewUniforms,
        snapshot: &WorldSnapshot,
        viewport: (u32, u32),
        scale: f32,
        offset: (f32, f32),
    ) -> u32 {
        // Build tile instances for visible terrain with simple CPU frustum culling.
        let (tw, th) = snapshot.terrain.dims;
        let cell = snapshot.terrain.cell_size as f32;
        let mut staging: Vec<TileInstance> = Vec::with_capacity((tw as usize) * (th as usize));
        let (vp_w, vp_h) = (viewport.0 as f32, viewport.1 as f32);
        let disable_cull = matches!(std::env::var("SB_WGPU_DISABLE_CULL").ok().map(|s| s.to_ascii_lowercase()), Some(ref v) if v == "1" || v == "true" || v == "yes" || v == "on");
        let elev_opt = snapshot.terrain.elevation;
        let get_elev = |x: i32, y: i32| -> f32 {
            if let Some(elev) = elev_opt {
                let xi = x.clamp(0, (tw as i32) - 1) as usize;
                let yi = y.clamp(0, (th as i32) - 1) as usize;
                let idx = yi * (tw as usize) + xi;
                elev.get(idx).copied().unwrap_or(0.5)
            } else {
                0.0
            }
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
                if !disable_cull
                    && (max_x_px < 0.0 || max_y_px < 0.0 || min_x_px > vp_w || min_y_px > vp_h)
                {
                    continue;
                }
                let idx = (y as usize) * (tw as usize) + (x as usize);
                let tile_id = snapshot.terrain.tiles.get(idx).copied().unwrap_or(3);
                let uv = self.atlas_uv_for(tile_id);
                // slope via central differences if elevation present
                let slope = if elev_opt.is_some() {
                    let dx = (get_elev(x + 1, y) - get_elev(x - 1, y)) * 0.5;
                    let dy = (get_elev(x, y + 1) - get_elev(x, y - 1)) * 0.5;
                    (dx * dx + dy * dy).sqrt().clamp(0.0, 1.0)
                } else {
                    0.0
                };
                staging.push(TileInstance {
                    pos: [px, py],
                    size: [cell, cell],
                    atlas_uv: uv,
                    kind: tile_id,
                    slope,
                });
            }
        }
        if !staging.is_empty() {
            let needed = (staging.len() * std::mem::size_of::<TileInstance>()) as u64;
            self.ensure_vbuf_capacity(device, needed);
            queue.write_buffer(&self.tile_vbuf, 0, bytemuck::cast_slice(&staging));
            self._tile_count = staging.len() as u32;
        }
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("terrain.pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: view_tex,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bg, &[]);
        pass.set_bind_group(1, &view_uniforms.bg, &[]);
        pass.set_vertex_buffer(0, self.tile_vbuf.slice(..));
        pass.draw(0..4, 0..staging.len() as u32);
        staging.len() as u32
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
  var rgb = base.rgb;
  // water shimmer for Deep/Shallow water kinds (0,1)
  if (v.kind <= 1u) {
    let time = view.v0.z;
    let wave = sin((v.uv.x * 40.0 + v.uv.y * 28.0) + time * 2.2);
    let shimmer = 0.04 + 0.06 * wave;
    // tuned caustics: stronger on shallow (kind==1), very subtle on deep (kind==0)
    let ca_s = (sin(v.uv.x * 160.0 + time * 1.7) * sin(v.uv.y * 140.0 + time * 1.5));
    let ca_amp = select(0.01, 0.05, v.kind == 1u);
    let ca = ca_s * ca_amp;
    rgb = clamp(rgb + vec3<f32>(shimmer + ca), vec3<f32>(0.0), vec3<f32>(1.0));
  } else {
    // slope accents (darken proportionally)
    let darken = clamp(1.0 - v.slope * 0.35, 0.0, 1.0);
    rgb = rgb * vec3<f32>(darken);
  }
  // subtle biome variation for grass/bloom/rock (kinds 3,4,5)
  if (v.kind >= 3u && v.kind <= 5u) {
    // stable hash from world coords -> [-1,1] noise
    let h = fract(sin(dot(v.world, vec2<f32>(12.9898, 78.233))) * 43758.5453);
    let n = (h * 2.0 - 1.0) * 0.06; // +/-6% brightness tweak
    rgb = clamp(rgb * (1.0 + n), vec3<f32>(0.0), vec3<f32>(1.0));
  }
  return vec4<f32>(rgb, base.a);
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
    data0: [f32; 4],
    data1: [f32; 4],
    data2: [f32; 4],
    data3: [f32; 4],
    data4: [f32; 4],
    data5: [f32; 4],
    data6: [f32; 4],
    data7: [f32; 4],
    data8: [f32; 4],
    data9: [f32; 4],
}

impl AgentPipeline {
    fn new(device: &wgpu::Device, color_format: wgpu::TextureFormat, view: &ViewUniforms) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("agents.wgsl"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(AGENTS_WGSL)),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("agents.layout"),
            bind_group_layouts: &[&view.layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("agents.pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<AgentInstanceGpu>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &wgpu::vertex_attr_array![
                        0 => Float32x4,
                        1 => Float32x4,
                        2 => Float32x4,
                        3 => Float32x4,
                        4 => Float32x4,
                        5 => Float32x4,
                        6 => Float32x4,
                        7 => Float32x4,
                        8 => Float32x4,
                        9 => Float32x4
                    ],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });
        let vbuf_capacity_bytes = (1024 * std::mem::size_of::<AgentInstanceGpu>()) as u64;
        let vbuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("agents.instances"),
            size: vbuf_capacity_bytes,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            pipeline,
            vbuf,
            vbuf_capacity_bytes,
        }
    }

    fn ensure_vbuf_capacity(&mut self, device: &wgpu::Device, needed_bytes: u64) {
        if needed_bytes <= self.vbuf_capacity_bytes {
            return;
        }
        let mut cap = self.vbuf_capacity_bytes.max(1024);
        while cap < needed_bytes {
            cap *= 2;
        }
        self.vbuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("agents.instances.realloc"),
            size: cap,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.vbuf_capacity_bytes = cap;
    }

    #[allow(clippy::too_many_arguments)]
    fn encode(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        view_tex: &wgpu::TextureView,
        view_uniforms: &ViewUniforms,
        snapshot: &WorldSnapshot,
        viewport: (u32, u32),
        scale: f32,
        offset: (f32, f32),
    ) -> u32 {
        let mut staging: Vec<AgentInstanceGpu> = Vec::with_capacity(snapshot.agents.len());
        let (vp_w, vp_h) = (viewport.0 as f32, viewport.1 as f32);
        let disable_cull = matches!(std::env::var("SB_WGPU_DISABLE_CULL").ok().map(|s| s.to_ascii_lowercase()), Some(ref v) if v == "1" || v == "true" || v == "yes" || v == "on");
        for a in snapshot.agents {
            // CPU frustum culling (pixel-space); assumes positions/sizes are pixels in this pass
            let cx = a.position[0] * scale + offset.0;
            let cy = a.position[1] * scale + offset.1;
            let radius_x = a.quad_extent[0] * scale;
            let radius_y = a.quad_extent[1] * scale;
            if !disable_cull
                && (cx + radius_x < 0.0
                    || cx - radius_x > vp_w
                    || cy + radius_y < 0.0
                    || cy - radius_y > vp_h)
            {
                continue;
            }
            staging.push(AgentInstanceGpu {
                data0: [
                    a.position[0],
                    a.position[1],
                    a.quad_extent[0],
                    a.quad_extent[1],
                ],
                data1: [
                    a.heading[0],
                    a.heading[1],
                    a.body_radius,
                    a.body_half_length,
                ],
                data2: [
                    a.wheel_offset,
                    a.wheel_radius,
                    a.mouth_open,
                    a.herbivore_tendency,
                ],
                data3: [
                    a.temperature_preference,
                    a.food_delta,
                    a.sound_level,
                    a.sound_output,
                ],
                data4: [a.wheel_left, a.wheel_right, a.trait_smell, a.trait_sound],
                data5: [a.trait_hearing, a.trait_eye, a.trait_blood, a.selection],
                data6: a.color,
                data7: [a.glow, a.boost, a.spiked, a.spike_length],
                data8: a.eye_dirs,
                data9: a.eye_fov,
            });
        }
        if !staging.is_empty() {
            let needed = (staging.len() * std::mem::size_of::<AgentInstanceGpu>()) as u64;
            self.ensure_vbuf_capacity(device, needed);
            queue.write_buffer(&self.vbuf, 0, bytemuck::cast_slice(&staging));
        }
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("agents.pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: view_tex,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &view_uniforms.bg, &[]);
        pass.set_vertex_buffer(0, self.vbuf.slice(..));
        pass.draw(0..4, 0..staging.len() as u32);
        staging.len() as u32
    }
}

const AGENTS_WGSL: &str = r#"
struct InInst {
  @location(0) data0: vec4<f32>,
  @location(1) data1: vec4<f32>,
  @location(2) data2: vec4<f32>,
  @location(3) data3: vec4<f32>,
  @location(4) data4: vec4<f32>,
  @location(5) data5: vec4<f32>,
  @location(6) data6: vec4<f32>,
  @location(7) data7: vec4<f32>,
  @location(8) data8: vec4<f32>,
  @location(9) data9: vec4<f32>,
};

struct VsOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) local: vec2<f32>,
  @location(1) extent: vec2<f32>,
  @location(2) heading: vec2<f32>,
  @location(3) body_params: vec4<f32>,
  @location(4) behavior: vec4<f32>,
  @location(5) audio: vec4<f32>,
  @location(6) traits_a: vec4<f32>,
  @location(7) traits_b: vec4<f32>,
  @location(8) color: vec4<f32>,
  @location(9) eye_dirs: vec4<f32>,
  @location(10) eye_fov: vec4<f32>,
  @location(11) extras: vec2<f32>,
};

struct View { v0: vec4<f32>, v1: vec4<f32> }; // v0=(viewport.x,viewport.y,time,scale) v1=(offset_x,offset_y,_,_)
@group(0) @binding(0) var<uniform> view: View;

fn capsule_distance(p: vec2<f32>, half_length: f32, radius: f32) -> f32 {
  let clamped = clamp(p.y, -half_length + radius, half_length - radius);
  return length(vec2<f32>(p.x, p.y - clamped)) - radius;
}

fn circle_distance(p: vec2<f32>, radius: f32) -> f32 {
  return length(p) - radius;
}

fn smooth_mask(dist: f32) -> f32 {
  let aa = max(fwidth(dist), 1e-3);
  return smoothstep(aa, -aa, dist);
}

fn layer(base_rgb: ptr<function, vec3<f32>>, base_alpha: ptr<function, f32>, color: vec3<f32>, alpha: f32) {
  let a = clamp(alpha, 0.0, 1.0);
  if (a <= 0.0001) {
    return;
  }
  let current = *base_alpha;
  let new_alpha = current + a * (1.0 - current);
  if (new_alpha <= 0.0001) {
    *base_rgb = color;
    *base_alpha = a;
    return;
  }
  let weight = a * (1.0 - current) / new_alpha;
  *base_rgb = mix(*base_rgb, color, weight);
  *base_alpha = new_alpha;
}

@vertex
fn vs_main(inst: InInst, @builtin(vertex_index) vid: u32) -> VsOut {
  var o: VsOut;
  let quad = array<vec2<f32>, 4>(vec2<f32>(-0.5,-0.5), vec2<f32>(0.5,-0.5), vec2<f32>(-0.5,0.5), vec2<f32>(0.5,0.5));
  let l = quad[vid];
  let extent = inst.data0.zw;
  let local = vec2<f32>(l.x * extent.x * 2.0, l.y * extent.y * 2.0);
  let viewport = view.v0.xy;
  let scale = view.v0.w;
  let offset = view.v1.xy;
  let center = inst.data0.xy;
  let world = (center + local) * scale + offset;
  let ndc = vec2<f32>(world.x / viewport.x * 2.0 - 1.0, 1.0 - (world.y / viewport.y * 2.0));
  o.pos = vec4<f32>(ndc, 0.0, 1.0);
  o.local = local;
  o.extent = extent;
  o.heading = inst.data1.xy;
  o.body_params = vec4<f32>(inst.data1.z, inst.data1.w, inst.data2.x, inst.data2.y);
  o.behavior = vec4<f32>(inst.data2.z, inst.data2.w, inst.data3.x, inst.data3.y);
  o.audio = vec4<f32>(inst.data3.z, inst.data3.w, inst.data4.x, inst.data4.y);
  o.traits_a = vec4<f32>(inst.data4.z, inst.data4.w, inst.data5.x, inst.data5.y);
  o.traits_b = vec4<f32>(inst.data5.z, inst.data5.w, inst.data7.x, inst.data7.y);
  o.color = inst.data6;
  o.eye_dirs = inst.data8;
  o.eye_fov = inst.data9;
  o.extras = vec2<f32>(inst.data7.z, inst.data7.w);
  return o;
}

@fragment
fn fs_main(v: VsOut) -> @location(0) vec4<f32> {
  let heading = normalize(v.heading);
  let right = vec2<f32>(-heading.y, heading.x);
  let local = vec2<f32>(dot(v.local, right), dot(v.local, heading));

  let body_radius = max(v.body_params.x, 0.5);
  let body_half_length = max(v.body_params.y, body_radius);
  let wheel_offset = v.body_params.z;
  let wheel_radius = v.body_params.w;
  let mouth_open = v.behavior.x;
  let herbivore = clamp(v.behavior.y, 0.0, 1.0);
  let temperature = clamp(v.behavior.z, 0.0, 1.0);
  let food_delta = v.behavior.w;
  let sound_level = clamp(abs(v.audio.x), 0.0, 1.0);
  let sound_output = clamp(abs(v.audio.y), 0.0, 1.0);
  let wheel_left = clamp(v.audio.z, 0.0, 1.0);
  let wheel_right = clamp(v.audio.w, 0.0, 1.0);
  let trait_smell = v.traits_a.x;
  let trait_sound = v.traits_a.y;
  let trait_hearing = v.traits_a.z;
  let trait_eye = v.traits_a.w;
  let trait_blood = v.traits_b.x;
  let selection = v.traits_b.y;
  let glow = v.traits_b.z;
  let boost = v.traits_b.w;
  let spiked = v.extras.x;
  let spike_length = v.extras.y;

  let body_dist = capsule_distance(local, body_half_length, body_radius);
  let body_mask = smooth_mask(body_dist);

  var accum_rgb = vec3<f32>(0.0);
  var accum_alpha = 0.0;

  // Wheels
  let wheel_half_length = body_half_length * 0.96;
  let wheel_base_color = vec3<f32>(0.14, 0.16, 0.21);
  let wheel_high_color = vec3<f32>(0.38, 0.42, 0.48);
  let left_dist = capsule_distance(vec2<f32>(local.x + wheel_offset, local.y), wheel_half_length, wheel_radius);
  let right_dist = capsule_distance(vec2<f32>(local.x - wheel_offset, local.y), wheel_half_length, wheel_radius);
  let left_color = mix(wheel_base_color, wheel_high_color, wheel_left);
  let right_color = mix(wheel_base_color, wheel_high_color, wheel_right);
  layer(&accum_rgb, &accum_alpha, left_color, smooth_mask(left_dist));
  layer(&accum_rgb, &accum_alpha, right_color, smooth_mask(right_dist));

  // Body shell
  let body_color = clamp(v.color.rgb, vec3<f32>(0.0), vec3<f32>(1.0));
  layer(&accum_rgb, &accum_alpha, body_color, body_mask);

  // Diet stripe
  let herb_color = vec3<f32>(0.24, 0.78, 0.36);
  let carn_color = vec3<f32>(0.88, 0.26, 0.21);
  let mut stripe = mix(carn_color, herb_color, herbivore);
  let blood_tint = clamp(0.7 + trait_blood * 0.2, 0.8, 1.35);
  stripe = clamp(stripe * blood_tint, vec3<f32>(0.0), vec3<f32>(1.2));
  let stripe_dist = capsule_distance(local, body_half_length * 0.82, body_radius * 0.45);
  layer(&accum_rgb, &accum_alpha, stripe, smooth_mask(stripe_dist) * body_mask * 0.9);

  // Flame (boost)
  if (boost > 0.05) {
    let flame_half = body_radius * (0.45 + boost * 0.5);
    let flame_radius = body_radius * (0.18 + boost * 0.4);
    let flame_center = vec2<f32>(0.0, -body_half_length - flame_half * 0.6);
    let flame_dist = capsule_distance(local - flame_center, flame_half, flame_radius);
    let flame_color = mix(vec3<f32>(1.0, 0.62, 0.22), vec3<f32>(1.0, 0.8, 0.3), sound_output);
    layer(&accum_rgb, &accum_alpha, flame_color, smooth_mask(flame_dist) * 0.8);
  }

  // Spike approximation
  let spike_half = body_radius * 0.7 + spike_length * 0.6;
  let spike_radius = body_radius * 0.32;
  let spike_center = vec2<f32>(0.0, body_half_length + spike_radius);
  let spike_dist = capsule_distance(local - spike_center, spike_half, spike_radius);
  let spike_color = mix(vec3<f32>(0.96, 0.44, 0.24), vec3<f32>(0.98, 0.58, 0.32), clamp(spiked, 0.0, 1.0));
  layer(&accum_rgb, &accum_alpha, spike_color, smooth_mask(spike_dist));

  // Mouth
  let mouth_half_length = body_radius * 0.62;
  let mouth_radius = max(body_radius * 0.14, 1.2) * mouth_open;
  let mouth_center = vec2<f32>(0.0, body_half_length - body_radius * 0.35);
  let mouth_local = local - mouth_center;
  let mouth_swapped = vec2<f32>(mouth_local.y, mouth_local.x);
  let mouth_dist = capsule_distance(mouth_swapped, mouth_half_length, mouth_radius);
  let eat_level = clamp(abs(food_delta), 0.0, 1.5);
  let yell_level = max(sound_output, sound_level);
  let mut mouth_color = vec3<f32>(
      0.85 + eat_level * 0.08,
      0.28 + eat_level * 0.3,
      0.32 + yell_level * 0.12
  );
  layer(&accum_rgb, &accum_alpha, clamp(mouth_color, vec3<f32>(0.0), vec3<f32>(1.0)), smooth_mask(mouth_dist));

  // Nose
  let nose_radius = max(body_radius * 0.12, 1.0) * (0.6 + trait_smell * 0.8);
  let nose_center = vec2<f32>(0.0, body_half_length - body_radius * 0.2);
  let nose_dist = circle_distance(local - nose_center, nose_radius);
  layer(&accum_rgb, &accum_alpha, vec3<f32>(0.92, 0.6, 0.28), smooth_mask(nose_dist));

  // Ears (sound/hearing)
  let ear_scale = clamp(0.6 + trait_hearing * 0.45, 0.6, 1.6);
  let ear_radius = max(body_radius * 0.28, 1.5) * ear_scale;
  let ear_offset = body_half_length * 0.15;
  let ear_color_base = vec3<f32>(0.32, 0.62, 0.92) * (0.9 + trait_sound * 0.45);
  let ear_left_center = vec2<f32>(-(body_radius + ear_radius * 0.45), -ear_offset);
  let ear_right_center = vec2<f32>(body_radius + ear_radius * 0.45, -ear_offset);
  let ear_left_dist = circle_distance(local - ear_left_center, ear_radius);
  let ear_right_dist = circle_distance(local - ear_right_center, ear_radius);
  layer(&accum_rgb, &accum_alpha, clamp(ear_color_base, vec3<f32>(0.0), vec3<f32>(1.0)), smooth_mask(ear_left_dist) * 0.9);
  layer(&accum_rgb, &accum_alpha, clamp(ear_color_base, vec3<f32>(0.0), vec3<f32>(1.0)), smooth_mask(ear_right_dist) * 0.9);

  // Eyes
  let eye_dirs = vec4<f32>(v.eye_dirs.x, v.eye_dirs.y, v.eye_dirs.z, v.eye_dirs.w);
  let eye_fov = vec4<f32>(v.eye_fov.x, v.eye_fov.y, v.eye_fov.z, v.eye_fov.w);
  let base_eye_radius = max(body_radius * 0.14, 1.2);
  let sclera_color = vec3<f32>(0.97, 0.98, 1.0);
  let pupil_color = vec3<f32>(0.08, 0.11, 0.18);
  for (var i: i32 = 0; i < 4; i = i + 1) {
    let angle = eye_dirs[i];
    let dir = vec2<f32>(cos(angle), sin(angle));
    let distance = body_radius * (0.4 + 0.35 * f32(i) / 4.0 + 0.25);
    let eye_center = dir * distance;
    var eye_radius = base_eye_radius * (0.65 + trait_eye * 0.35);
    eye_radius = clamp(eye_radius, 1.6, body_radius * 0.38);
    let eye_dist = circle_distance(local - eye_center, eye_radius);
    let eye_mask = smooth_mask(eye_dist);
    layer(&accum_rgb, &accum_alpha, sclera_color, eye_mask);

    let pupil_radius = eye_radius * (0.35 + clamp(eye_fov[i], 0.3, 3.0) * 0.12);
    let pupil_dist = circle_distance(local - eye_center, pupil_radius);
    layer(&accum_rgb, &accum_alpha, pupil_color, smooth_mask(pupil_dist));
  }

  // Temperature marker
  let temp_color = mix(vec3<f32>(0.20, 0.52, 0.96), vec3<f32>(0.98, 0.42, 0.18), temperature);
  let temp_center = vec2<f32>(0.0, -body_half_length * 0.25);
  let temp_radius = body_radius * 0.22;
  let temp_ring_dist = circle_distance(local - temp_center, temp_radius);
  let temp_ring = smooth_mask(temp_ring_dist) * 0.6;
  layer(&accum_rgb, &accum_alpha, temp_color, temp_ring * 0.6);

  // Sound arcs
  let vocal = max(sound_output, sound_level);
  if (vocal > 0.12) {
    let arc_origin = vec2<f32>(0.0, body_half_length + body_radius * 0.4);
    let arc_r1 = body_radius * (0.55 + vocal * 0.6);
    let arc_r2 = arc_r1 + body_radius * 0.35;
    let arc_color = vec3<f32>(0.95, 0.68, 0.32) * (0.6 + vocal * 0.6);
    let arc1 = circle_distance(local - arc_origin, arc_r1);
    let arc2 = circle_distance(local - arc_origin, arc_r2);
    layer(&accum_rgb, &accum_alpha, arc_color, smooth_mask(arc1) * 0.35);
    layer(&accum_rgb, &accum_alpha, arc_color, smooth_mask(arc2) * 0.25);
  }

  // Selection + indicator rim
  let sel_hover = step(0.5, selection) * (1.0 - step(1.5, selection));
  let sel_selected = step(1.5, selection);
  let selection_glow = sel_hover * 0.25 + sel_selected * 0.45;
  let rim = smoothstep(0.0, -max(fwidth(body_dist), 0.001) * 1.5, body_dist + body_radius * 0.15);
  let rim_color = vec3<f32>(1.0, 1.0, 1.0) * (selection_glow + glow);
  layer(&accum_rgb, &accum_alpha, rim_color, rim);

  // Boost tint overlay
  if (boost > 0.05) {
    let boost_tint = vec3<f32>(0.98, 0.62, 0.32);
    layer(&accum_rgb, &accum_alpha, boost_tint, body_mask * boost * 0.2);
  }

  let alpha = clamp(accum_alpha, 0.0, 1.0);
  let rgb = clamp(accum_rgb, vec3<f32>(0.0), vec3<f32>(1.0));
  return vec4<f32>(rgb, alpha);
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

    // Cached env-driven parameters to avoid per-frame parsing
    last_env_update: Option<std::time::Instant>,
    env_exposure: f32,
    env_vignette: f32,
    env_tonemap: u32,
    env_fxaa: u32,
    env_bloom_on: u32,
    env_bloom_thresh: f32,
    env_bloom_intensity: f32,
    env_fog_enabled: u32,
    env_fog_density: f32,
    env_fog_color: [f32; 3],
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
    fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        src_view: &wgpu::TextureView,
        size: (u32, u32),
    ) -> Self {
        // Final composite shader
        let post_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("postfx.wgsl"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(POST_WGSL)),
        });
        // Final pass samples: src color (binding 0), sampler (1), bloom blurred (2)
        let src_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("post.src_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("post.sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        // Placeholder bloom view: use source for now; real bloom bound during run()
        let src_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("post.src_bg"),
            layout: &src_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(src_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(src_view),
                },
            ],
        });
        let params_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("post.params_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: std::num::NonZeroU64::new(
                        std::mem::size_of::<PostParams>() as u64
                    ),
                },
                count: None,
            }],
        });
        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("post.params_buf"),
            size: std::mem::size_of::<PostParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("post.params_bg"),
            layout: &params_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buf.as_entire_binding(),
            }],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("post.layout"),
            bind_group_layouts: &[&src_layout, &params_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("post.pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &post_shader,
                entry_point: Some("vs_fullscreen"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &post_shader,
                entry_point: Some("fs_post"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });
        let (target, target_view) = create_color(device, format, size);

        // Bloom shaders and layouts
        let bloom_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bloom.wgsl"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(BLOOM_WGSL)),
        });
        let bloom_src_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bloom.src_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let bloom_src_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bloom.src_bg"),
            layout: &bloom_src_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(src_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });
        let blur_params_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("bloom.blur_params_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(16),
                    },
                    count: None,
                }],
            });
        let blur_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bloom.blur_params_buf"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let blur_params_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bloom.blur_params_bg"),
            layout: &blur_params_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: blur_params_buf.as_entire_binding(),
            }],
        });
        // Extract pipeline: src -> bloom_a (half res)
        let bloom_extract_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("bloom.extract.layout"),
            bind_group_layouts: &[&bloom_src_layout, &params_layout],
            push_constant_ranges: &[],
        });
        let bloom_extract_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("bloom.extract.pipeline"),
                layout: Some(&bloom_extract_layout),
                vertex: wgpu::VertexState {
                    module: &bloom_shader,
                    entry_point: Some("vs_fullscreen"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &bloom_shader,
                    entry_point: Some("fs_extract"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });
        // Blur pipeline: bloom_a <-> bloom_b
        let bloom_blur_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("bloom.blur.layout"),
            bind_group_layouts: &[&bloom_src_layout, &blur_params_layout],
            push_constant_ranges: &[],
        });
        let bloom_blur_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("bloom.blur.pipeline"),
            layout: Some(&bloom_blur_layout),
            vertex: wgpu::VertexState {
                module: &bloom_shader,
                entry_point: Some("vs_fullscreen"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &bloom_shader,
                entry_point: Some("fs_blur"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
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
            last_env_update: None,
            env_exposure: 1.0,
            env_vignette: 0.08,
            env_tonemap: 1, // aces
            env_fxaa: 0,
            env_bloom_on: 1,
            env_bloom_thresh: 0.8,
            env_bloom_intensity: 0.65,
            env_fog_enabled: 1,
            env_fog_density: 0.6, // low
            env_fog_color: [0.6, 0.7, 0.8],
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

    fn rebind(
        &mut self,
        device: &wgpu::Device,
        src_view: &wgpu::TextureView,
        bloom_view: &wgpu::TextureView,
    ) {
        self.src_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("post.src_bg.rebind"),
            layout: &self.src_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(src_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(bloom_view),
                },
            ],
        });
    }

    fn ensure_bloom_targets(
        &mut self,
        device: &wgpu::Device,
        _format: wgpu::TextureFormat,
        full: (u32, u32),
    ) {
        if self.bloom_a.is_some() && self.bloom_b.is_some() {
            return;
        }
        let half = (full.0.max(1) / 2).max(1);
        let half_h = (full.1.max(1) / 2).max(1);
        let make = |label: &str| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: half,
                    height: half_h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm, // linear for correct blur/composite
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
        };
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
        self.bloom_src_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bloom.src_bg.rebind"),
            layout: &self.bloom_src_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(src_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
    }

    fn run(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        src: &wgpu::TextureView,
        size: (u32, u32),
    ) {
        // Refresh env-driven parameters at most every 250ms to avoid per-frame parsing overhead
        let needs_refresh = self
            .last_env_update
            .map(|t| t.elapsed().as_millis() > 250)
            .unwrap_or(true);
        if needs_refresh {
            self.env_exposure = std::env::var("SB_WGPU_EXPOSURE")
                .ok()
                .and_then(|v| v.parse::<f32>().ok())
                .unwrap_or(self.env_exposure);
            self.env_vignette = std::env::var("SB_WGPU_VIGNETTE")
                .ok()
                .and_then(|v| v.parse::<f32>().ok())
                .unwrap_or(self.env_vignette);
            self.env_tonemap = match std::env::var("SB_WGPU_TONEMAP").ok().as_deref() {
                Some("filmic") => 1u32,
                Some("reinhard") => 2u32,
                _ => 0u32,
            };
            self.env_fxaa = std::env::var("SB_WGPU_FXAA")
                .ok()
                .and_then(|v| v.parse::<u32>().ok())
                .unwrap_or(self.env_fxaa);
            self.env_bloom_on = std::env::var("SB_WGPU_BLOOM")
                .ok()
                .and_then(|v| v.parse::<u32>().ok())
                .unwrap_or(self.env_bloom_on);
            self.env_bloom_thresh = std::env::var("SB_WGPU_BLOOM_THRESH")
                .ok()
                .and_then(|v| v.parse::<f32>().ok())
                .unwrap_or(self.env_bloom_thresh);
            self.env_bloom_intensity = std::env::var("SB_WGPU_BLOOM_INTENSITY")
                .ok()
                .and_then(|v| v.parse::<f32>().ok())
                .unwrap_or(self.env_bloom_intensity);
            let fog_mode = std::env::var("SB_WGPU_FOG")
                .ok()
                .unwrap_or_else(|| "off".to_string());
            self.env_fog_enabled = match fog_mode.as_str() {
                "low" | "med" | "high" => 1u32,
                _ => 0u32,
            };
            self.env_fog_density = match fog_mode.as_str() {
                "low" => 0.6,
                "med" => 1.0,
                "high" => 1.6,
                _ => self.env_fog_density,
            };
            if let Some(c) = std::env::var("SB_WGPU_FOG_COLOR").ok().and_then(|v| {
                let parts: Vec<_> = v.split(',').collect();
                if parts.len() == 3 {
                    Some([
                        parts[0]
                            .trim()
                            .parse::<f32>()
                            .unwrap_or(self.env_fog_color[0]),
                        parts[1]
                            .trim()
                            .parse::<f32>()
                            .unwrap_or(self.env_fog_color[1]),
                        parts[2]
                            .trim()
                            .parse::<f32>()
                            .unwrap_or(self.env_fog_color[2]),
                    ])
                } else {
                    None
                }
            }) {
                self.env_fog_color = c;
            }
            self.last_env_update = Some(std::time::Instant::now());
        }

        let params = PostParams {
            exposure: self.env_exposure,
            vignette: self.env_vignette,
            tonemap: self.env_tonemap,
            fxaa: self.env_fxaa,
            bloom_thresh: self.env_bloom_thresh,
            bloom_intensity: self.env_bloom_intensity,
            fog_density: self.env_fog_density,
            fog_enabled: self.env_fog_enabled,
            fog_color: self.env_fog_color,
            _pad0: 0.0,
        };
        queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
        // Recreate params bind group (demonstrates use of params_layout; allows for dynamic layout changes in future)
        self.params_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("post.params_bg.rebind"),
            layout: &self.params_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: self.params_buf.as_entire_binding(),
            }],
        });

        // Bloom pass chain if enabled
        let mut bloom_view_opt: Option<wgpu::TextureView> = None;
        if self.env_bloom_on != 0 {
            self.ensure_bloom_targets(device, self.color_format, size);
            // Create fresh local views to avoid borrowing self across mutable calls
            let a_view_local = self
                .bloom_a
                .as_ref()
                .unwrap()
                .create_view(&wgpu::TextureViewDescriptor::default());
            let b_view_local = self
                .bloom_b
                .as_ref()
                .unwrap()
                .create_view(&wgpu::TextureViewDescriptor::default());
            // Extract brights from src -> A
            self.bind_bloom_src(device, src);
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("bloom.extract"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &a_view_local,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    occlusion_query_set: None,
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.bloom_extract_pipeline);
                pass.set_bind_group(0, &self.bloom_src_bg, &[]);
                pass.set_bind_group(1, &self.params_bg, &[]);
                pass.draw(0..3, 0..1);
            }
            // Blur A -> B (horizontal)
            let dir_h: [f32; 4] = [1.0 / (size.0.max(1) as f32 * 0.5), 0.0, 0.0, 0.0];
            queue.write_buffer(&self.blur_params_buf, 0, bytemuck::bytes_of(&dir_h));
            self.blur_params_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bloom.blur_params_bg.rebind"),
                layout: &self.blur_params_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.blur_params_buf.as_entire_binding(),
                }],
            });
            self.bind_bloom_src(device, &a_view_local);
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("bloom.blur.h"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &b_view_local,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    occlusion_query_set: None,
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.bloom_blur_pipeline);
                pass.set_bind_group(0, &self.bloom_src_bg, &[]);
                pass.set_bind_group(1, &self.blur_params_bg, &[]);
                pass.draw(0..3, 0..1);
            }
            // Blur B -> A (vertical)
            let dir_v: [f32; 4] = [0.0, 1.0 / (size.1.max(1) as f32 * 0.5), 0.0, 0.0];
            queue.write_buffer(&self.blur_params_buf, 0, bytemuck::bytes_of(&dir_v));
            self.blur_params_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bloom.blur_params_bg.rebind"),
                layout: &self.blur_params_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.blur_params_buf.as_entire_binding(),
                }],
            });
            self.bind_bloom_src(device, &b_view_local);
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("bloom.blur.v"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &a_view_local,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    occlusion_query_set: None,
                    timestamp_writes: None,
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
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.target_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.src_bg, &[]);
        pass.set_bind_group(1, &self.params_bg, &[]);
        pass.draw(0..3, 0..1);
    }
}

fn wants_post() -> bool {
    static CACHE: std::sync::OnceLock<std::sync::Mutex<(std::time::Instant, bool)>> =
        std::sync::OnceLock::new();
    let cache = CACHE.get_or_init(|| std::sync::Mutex::new((std::time::Instant::now(), true)));
    let mut guard = cache.lock().unwrap();
    let (last, val) = &mut *guard;
    if last.elapsed().as_millis() > 250 {
        let fxaa = std::env::var("SB_WGPU_FXAA")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(0)
            != 0;
        let tonemap = std::env::var("SB_WGPU_TONEMAP")
            .map(|v| !v.is_empty())
            .unwrap_or(true);
        let bloom = std::env::var("SB_WGPU_BLOOM")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(1)
            != 0;
        let fog = matches!(
            std::env::var("SB_WGPU_FOG").ok().as_deref(),
            Some("low") | Some("med") | Some("high")
        );
        *val = fxaa || tonemap || bloom || fog;
        *last = std::time::Instant::now();
    }
    *val
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
    let h = 1.0 - uv01.y; // bottom-heavy fog (denser near bottom)
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

fn env_flag(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .map(|v| {
            let s = v.trim().to_ascii_lowercase();
            matches!(s.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false)
}
