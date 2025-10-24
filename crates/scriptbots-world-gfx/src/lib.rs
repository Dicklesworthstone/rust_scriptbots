#![forbid(unsafe_code)]

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec2, Vec3};
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
    // Pipelines and buffers will be added in subsequent commits.
}

pub struct RenderFrame {
    pub color: wgpu::Texture,
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

        Ok(Self {
            device,
            queue,
            size,
            color,
            color_view,
            format,
            readback,
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
        Ok(())
    }

    pub fn render(&mut self, _snapshot: &WorldSnapshot) -> RenderFrame {
        // Placeholder clear; pipelines will be added in later steps.
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("world.render") });
        {
            let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("world.clear"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.03, g: 0.06, b: 0.12, a: 1.0 }),
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
        }
        self.queue.submit(Some(encoder.finish()));
        RenderFrame { color: self.color.clone(), extent: self.size }
    }

    pub fn copy_to_readback(&mut self, frame: &RenderFrame) -> Result<(), String> {
        self.readback.copy(&self.device, &self.queue, frame)
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
    pub bytes: &'a [u8],
    pub bytes_per_row: u32,
    pub width: u32,
    pub height: u32,
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

    pub fn copy(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, frame: &RenderFrame) -> Result<(), String> {
        let slot = &mut self.slots[self.curr];
        slot.ready = false;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("world.readback.copy") });
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &frame.color,
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
        let data = slice.get_mapped_range();
        Some(ReadbackView { bytes: &data, bytes_per_row: self.bytes_per_row, width: self.extent.0, height: self.extent.1 })
    }
}

fn align_256(n: u32) -> u32 { ((n + 255) / 256) * 256 }


