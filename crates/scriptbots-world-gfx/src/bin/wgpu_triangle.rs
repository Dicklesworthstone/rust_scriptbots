//! Minimal wgpu triangle smoke test (no vertex buffers).
//! Run: `cargo run -p scriptbots-world-gfx --bin wgpu_triangle`

use winit::{event::Event, event_loop::EventLoop};

#[cfg(target_os = "macos")]
const BACKEND: wgpu::Backends = wgpu::Backends::METAL;
#[cfg(any(target_os = "linux", target_os = "windows"))]
const BACKEND: wgpu::Backends = wgpu::Backends::VULKAN;

fn main() { pollster::block_on(run()); }

async fn run() {
    let event_loop = EventLoop::new().expect("event loop");
    let window = event_loop
        .create_window(winit::window::Window::default_attributes()
            .with_title("wgpu triangle smoke test")
            .with_inner_size(winit::dpi::LogicalSize::new(800.0, 600.0)))
        .expect("window");

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor { backends: BACKEND, ..Default::default() });
    let surface = instance.create_surface(&window).expect("surface");
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, compatible_surface: Some(&surface), force_fallback_adapter: false })
        .await
        .expect("adapter");
    let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor::default()).await.expect("device");

    let size = window.inner_size();
    let format = surface
        .get_capabilities(&adapter)
        .formats
        .into_iter()
        .find(|f| f.is_srgb())
        .unwrap_or(wgpu::TextureFormat::Bgra8UnormSrgb);
    let mut config = wgpu::SurfaceConfiguration {
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

struct Push { t: f32 }
@group(0) @binding(0) var<uniform> time: f32;

@fragment
fn fs() -> @location(0) vec4<f32> {
  // simple animated color
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

    // A tiny uniform buffer for time animation
    let time_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("time"),
        size: 4,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("time.layout"),
        entries: &[wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: Some(std::num::NonZeroU64::new(4).unwrap()) }, count: None }],
    });
    let time_bg = device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("time.bg"), layout: &layout, entries: &[wgpu::BindGroupEntry { binding: 0, resource: time_buf.as_entire_binding() }] });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("triangle.layout"), bind_group_layouts: &[&layout], push_constant_ranges: &[] });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("triangle.pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs"), compilation_options: Default::default(), buffers: &[] },
        fragment: Some(wgpu::FragmentState { module: &shader, entry_point: Some("fs"), compilation_options: Default::default(), targets: &[Some(wgpu::ColorTargetState { format, blend: Some(wgpu::BlendState::REPLACE), write_mask: wgpu::ColorWrites::ALL })] }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    });

    let start = std::time::Instant::now();

    let _ = event_loop.run(move |event, target: &winit::event_loop::ActiveEventLoop| {
        match event {
            Event::WindowEvent { event, window_id } if window_id == window.id() => match event {
                winit::event::WindowEvent::CloseRequested => target.exit(),
                winit::event::WindowEvent::Resized(new_size) => {
                    config.width = new_size.width.max(1);
                    config.height = new_size.height.max(1);
                    surface.configure(&device, &config);
                }
                _ => {}
            },
            Event::AboutToWait => {
                let secs = start.elapsed().as_secs_f32();
                queue.write_buffer(&time_buf, 0, bytemuck::bytes_of(&secs));

                let frame = match surface.get_current_texture() {
                    Ok(f) => f,
                    Err(wgpu::SurfaceError::Lost) | Err(wgpu::SurfaceError::Outdated) => { surface.configure(&device, &config); return; }
                    Err(wgpu::SurfaceError::Timeout) => return,
                    Err(wgpu::SurfaceError::OutOfMemory) => { target.exit(); return; }
                };
                let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("triangle") });
                {
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("triangle.pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: &view, depth_slice: None, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.06, g: 0.08, b: 0.12, a: 1.0 }), store: wgpu::StoreOp::Store } })],
                        depth_stencil_attachment: None,
                        occlusion_query_set: None,
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&pipeline);
                    pass.set_bind_group(0, &time_bg, &[]);
                    pass.draw(0..3, 0..1);
                }
                queue.submit(Some(encoder.finish()));
                frame.present();
            }
            _ => {}
        }
    });
}


