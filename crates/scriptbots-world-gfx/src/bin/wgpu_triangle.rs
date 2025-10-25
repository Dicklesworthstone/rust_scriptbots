//! Minimal wgpu triangle smoke test (no vertex buffers), using winit 0.30 app model.
//! Run: `cargo run -p scriptbots-world-gfx --bin wgpu_triangle`

use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowAttributes},
};

#[cfg(target_os = "macos")]
const BACKEND: wgpu::Backends = wgpu::Backends::METAL;
#[cfg(any(target_os = "linux", target_os = "windows"))]
const BACKEND: wgpu::Backends = wgpu::Backends::VULKAN;

struct TriangleApp {
    window: Option<&'static Window>,
    instance: wgpu::Instance,
    surface: Option<wgpu::Surface<'static>>,
    adapter: Option<wgpu::Adapter>,
    device: Option<wgpu::Device>,
    queue: Option<wgpu::Queue>,
    config: Option<wgpu::SurfaceConfiguration>,
    shader: Option<wgpu::ShaderModule>,
    time_buf: Option<wgpu::Buffer>,
    time_bg: Option<wgpu::BindGroup>,
    pipeline: Option<wgpu::RenderPipeline>,
    start: std::time::Instant,
}

impl TriangleApp {
    fn new() -> Self {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: BACKEND,
            ..Default::default()
        });
        Self {
            window: None,
            instance,
            surface: None,
            adapter: None,
            device: None,
            queue: None,
            config: None,
            shader: None,
            time_buf: None,
            time_bg: None,
            pipeline: None,
            start: std::time::Instant::now(),
        }
    }
}

impl ApplicationHandler for TriangleApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let attrs: WindowAttributes = Window::default_attributes()
            .with_title("wgpu triangle smoke test")
            .with_inner_size(winit::dpi::LogicalSize::new(800.0, 600.0));
        let window_box = Box::new(event_loop.create_window(attrs).expect("window"));
        let window: &'static Window = Box::leak(window_box);

        let surface = self.instance.create_surface(window).expect("surface");
        let adapter = pollster::block_on(self.instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .expect("adapter");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
            .expect("device");

        let size = window.inner_size();
        let format = surface
            .get_capabilities(&adapter)
            .formats
            .into_iter()
            .find(|f| f.is_srgb())
            .unwrap_or(wgpu::TextureFormat::Bgra8UnormSrgb);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::Fifo,
            desired_maximum_frame_latency: 1,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        let shader_src = r#"
@vertex
fn vs(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
  var pos = array<vec2<f32>,3>(
    vec2<f32>(-0.8, -0.7),
    vec2<f32>( 0.8, -0.7),
    vec2<f32>( 0.0,  0.7)
  );
  let p = pos[vid];
  return vec4<f32>(p, 0.0, 1.0);
}

@group(0) @binding(0) var<uniform> time: f32;

@fragment
fn fs() -> @location(0) vec4<f32> {
  let r = 0.2 + fract(time * 0.31) * 0.6;
  let g = 0.4 + fract(time * 0.17) * 0.5;
  let b = 0.7 + fract(time * 0.11) * 0.3;
  return vec4<f32>(r, g, b, 1.0);
}
"#;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("triangle.wgsl"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(shader_src)),
        });

        let time_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("time"),
            size: 4,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("time.layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: Some(std::num::NonZeroU64::new(4).unwrap()),
                },
                count: None,
            }],
        });
        let time_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("time.bg"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: time_buf.as_entire_binding() }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("triangle.layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("triangle.pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState { format, blend: Some(wgpu::BlendState::REPLACE), write_mask: wgpu::ColorWrites::ALL })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        self.window = Some(window);
        self.surface = Some(surface);
        self.adapter = Some(adapter);
        self.device = Some(device);
        self.queue = Some(queue);
        self.config = Some(config);
        self.shader = Some(shader);
        self.time_buf = Some(time_buf);
        self.time_bg = Some(time_bg);
        self.pipeline = Some(pipeline);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: winit::window::WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(new_size) => {
                if let (Some(surface), Some(device), Some(cfg)) =
                    (self.surface.as_ref(), self.device.as_ref(), self.config.as_mut())
                {
                    cfg.width = new_size.width.max(1);
                    cfg.height = new_size.height.max(1);
                    surface.configure(device, cfg);
                }
            }
            WindowEvent::RedrawRequested => {
                if let (Some(surface), Some(device), Some(queue), Some(cfg), Some(time_buf), Some(time_bg), Some(pipeline)) = (
                    self.surface.as_ref(),
                    self.device.as_ref(),
                    self.queue.as_ref(),
                    self.config.as_ref(),
                    self.time_buf.as_ref(),
                    self.time_bg.as_ref(),
                    self.pipeline.as_ref(),
                ) {
                    let secs = self.start.elapsed().as_secs_f32();
                    queue.write_buffer(time_buf, 0, bytemuck::bytes_of(&secs));

                    let frame = match surface.get_current_texture() {
                        Ok(f) => f,
                        Err(wgpu::SurfaceError::Lost) | Err(wgpu::SurfaceError::Outdated) => {
                            surface.configure(device, cfg);
                            return;
                        }
                        Err(wgpu::SurfaceError::Timeout) => return,
                        Err(wgpu::SurfaceError::OutOfMemory) | Err(wgpu::SurfaceError::Other) => {
                            event_loop.exit();
                            return;
                        }
                    };
                    let view = frame
                        .texture
                        .create_view(&wgpu::TextureViewDescriptor::default());
                    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("triangle") });
                    {
                        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("triangle.pass"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: &view,
                                depth_slice: None,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.06, g: 0.08, b: 0.12, a: 1.0 }),
                                    store: wgpu::StoreOp::Store,
                                },
                            })],
                            depth_stencil_attachment: None,
                            occlusion_query_set: None,
                            timestamp_writes: None,
                        });
                        pass.set_pipeline(pipeline);
                        pass.set_bind_group(0, time_bg, &[]);
                        pass.draw(0..3, 0..1);
                    }
                    queue.submit(Some(encoder.finish()));
                    frame.present();
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }
}

fn main() {
    let mut app = TriangleApp::new();
    let event_loop = EventLoop::new().expect("event loop");
    let _ = event_loop.run_app(&mut app);
}

