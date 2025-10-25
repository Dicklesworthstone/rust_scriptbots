//! Minimal wgpu window smoke test: open a window and clear to a solid color.
//! Run: `cargo run -p scriptbots-world-gfx --bin wgpu_clear`

use winit::{event::Event, event_loop::EventLoop, window::WindowBuilder};

#[cfg(target_os = "macos")]
const BACKEND: wgpu::Backends = wgpu::Backends::METAL;
#[cfg(any(target_os = "linux", target_os = "windows"))]
const BACKEND: wgpu::Backends = wgpu::Backends::VULKAN;

fn main() {
    pollster::block_on(run());
}

async fn run() {
    let event_loop = EventLoop::new().expect("event loop");
    let window = WindowBuilder::new()
        .with_title("wgpu clear smoke test")
        .with_inner_size(winit::dpi::LogicalSize::new(800.0, 600.0))
        .build(&event_loop)
        .expect("window");

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: BACKEND,
        ..Default::default()
    });
    let surface = instance.create_surface(&window).expect("surface");
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        })
        .await
        .expect("adapter");
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default(), None)
        .await
        .expect("device");

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

    let mut bg = 0.12f32;

    let _ = event_loop.run(move |event, target| {
        match event {
            Event::WindowEvent { event, window_id } if window_id == window.id() => {
                match event {
                    winit::event::WindowEvent::CloseRequested => target.exit(),
                    winit::event::WindowEvent::Resized(new_size) => {
                        config.width = new_size.width.max(1);
                        config.height = new_size.height.max(1);
                        surface.configure(&device, &config);
                    }
                    _ => {}
                }
            }
            Event::AboutToWait => {
                // simple animated background so we can visually confirm frames
                bg = (bg + 0.002).fract();
                let frame = match surface.get_current_texture() {
                    Ok(frame) => frame,
                    Err(wgpu::SurfaceError::Lost) | Err(wgpu::SurfaceError::Outdated) => {
                        surface.configure(&device, &config);
                        return;
                    }
                    Err(wgpu::SurfaceError::Timeout) => return,
                    Err(wgpu::SurfaceError::OutOfMemory) => { target.exit(); return; }
                };
                let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("clear") });
                {
                    let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("clear.pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            depth_slice: None,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color { r: (bg as f64), g: 0.08, b: 0.18, a: 1.0 }),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        occlusion_query_set: None,
                        timestamp_writes: None,
                    });
                }
                queue.submit(Some(encoder.finish()));
                frame.present();
            }
            _ => {}
        }
    });
}


