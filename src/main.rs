mod controls;
mod scene;

use controls::Controls;
use scene::Scene;

use iced_wgpu::graphics::Viewport;
use iced_wgpu::{Engine, Renderer, wgpu};
use iced_winit::Clipboard;
use iced_winit::conversion;
use iced_winit::core::mouse;
use iced_winit::core::renderer;
use iced_winit::core::{Font, Pixels, Size, Theme, Color};
use iced_winit::futures;
use iced_winit::runtime::program;
use iced_winit::runtime::Debug;
use iced_winit::winit;

use winit::{
    application::ApplicationHandler,
    keyboard::ModifiersState,
    event::{WindowEvent, MouseScrollDelta, ElementState, MouseButton},
    event_loop::{ControlFlow, EventLoop, ActiveEventLoop},
    window::{Window, WindowId, WindowAttributes},
};

use std::rc::Rc;
use std::cell::RefCell;
use std::sync::Arc;

#[allow(clippy::large_enum_variant)]
enum Runner {
    Loading,
    Ready {
        window: Arc<Window>,
        queue: wgpu::Queue,
        device: wgpu::Device,
        surface: wgpu::Surface<'static>,
        format: wgpu::TextureFormat,
        engine: Engine,
        renderer: Renderer,
        debug: Debug,
        scene: Rc<RefCell<Scene>>,
        state: program::State<Controls>,
        cursor_position: Option<winit::dpi::PhysicalPosition<f64>>,
        clipboard: Clipboard,
        viewport: Viewport,
        modifiers: ModifiersState,
        resized: bool,
        // For tracking mouse movment and shifting (offsetting) the fractal
        prev_pos: (f64, f64),
        mouse_state: ElementState,
    },
}

impl ApplicationHandler for Runner {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if let Self::Loading = self {
            let window = Arc::new(event_loop.create_window(
                            WindowAttributes::default())
                            .expect("Create window"));

            let physical_size = window.inner_size();
            let viewport = Viewport::with_physical_size(
                Size::new(physical_size.width, physical_size.height),
                window.scale_factor() as f64);

            let clipboard = Clipboard::connect(window.clone());
            let backend = wgpu::util::backend_bits_from_env().unwrap_or_default();

            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: backend,
                ..Default::default()});
            let surface = instance
                .create_surface(window.clone())
                .expect("Create window surface");

            let (format, adapter, device, queue) =
                futures::futures::executor::block_on(async {
                    let adapter = wgpu::util::initialize_adapter_from_env_or_default(
                                    &instance, Some(&surface))
                        .await
                        .expect("Create adapter");

                    let adapter_features = adapter.features();
                    let capabilities = surface.get_capabilities(&adapter);

                    let (device, queue) = adapter
                        .request_device(&wgpu::DeviceDescriptor {label: None,
                            required_features: adapter_features &wgpu::Features::default(),
                            required_limits: wgpu::Limits::default()}, None)
                        .await
                        .expect("Request device");

                    (capabilities.formats.iter().copied().find(wgpu::TextureFormat::is_srgb)
                            .or_else(|| {capabilities.formats.first().copied()})
                            .expect("Get preferred format"),
                        adapter, device, queue)
                });

            surface.configure(&device, &wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT, format,
                width: physical_size.width,
                height: physical_size.height,
                present_mode: wgpu::PresentMode::AutoVsync,
                alpha_mode: wgpu::CompositeAlphaMode::Auto,
                view_formats: vec![],
                desired_maximum_frame_latency: 2});

            // Initialize scene and GUI controls
            let scene = Rc::new(RefCell::new(Scene::new(&device, format)));
            let controls = Controls::new(Rc::clone(&scene));

            // Initialize iced
            let mut debug = Debug::new();
            let engine = Engine::new(&adapter, &device, &queue, format, None);
            let mut renderer = Renderer::new(&device, &engine, Font::default(), Pixels::from(16));
            let state = program::State::new(controls, viewport.logical_size(), &mut renderer, &mut debug);

            // Change this to render continuously
            event_loop.set_control_flow(ControlFlow::Wait);
            
            let prev_pos = (-1.0, -1.0);
            let mouse_state = ElementState::Released;

            *self = Self::Ready {window, device, queue, surface, format, engine, renderer, debug, 
                scene: Rc::clone(&scene), state, cursor_position: None, modifiers: ModifiersState::default(),
                clipboard, viewport, resized: false, prev_pos, mouse_state};
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop,
            _window_id: WindowId, event: WindowEvent) {
        let Self::Ready {window, device, queue, surface, format, engine, renderer, debug, scene, 
            state, viewport, cursor_position, modifiers, clipboard, resized, 
            mouse_state, prev_pos} = self
        else {
            return;
        };
        //println!("Runner.window_event - {:?} {:?}", _window_id, event);
        
        match event {
            WindowEvent::RedrawRequested => {
                if *resized {
                    let size = window.inner_size();
                    scene.borrow_mut().set_aspect_ratio(size.width as f32 / size.height as f32);

                    *viewport = Viewport::with_physical_size(
                        Size::new(size.width, size.height),
                        window.scale_factor() as f64);

                    surface.configure(device, &wgpu::SurfaceConfiguration {
                            format: *format,
                            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                            width: size.width,
                            height: size.height,
                            present_mode: wgpu::PresentMode::AutoVsync,
                            alpha_mode: wgpu::CompositeAlphaMode::Auto,
                            view_formats: vec![],
                            desired_maximum_frame_latency: 2});

                    *resized = false;
                }

                match surface.get_current_texture() {
                    Ok(frame) => {
                        let mut encoder = device.create_command_encoder(
                            &wgpu::CommandEncoderDescriptor { label: None });

                        let view = frame.texture.create_view(
                            &wgpu::TextureViewDescriptor::default());

                        {
                            let s = scene.borrow_mut();

                            // Clear the frame
                            let mut render_pass = Scene::clear(&view, &mut encoder);

                            // Draw the scene
                            s.draw(&queue, &mut render_pass);
                        }

                        // Draw Iced on top
                        renderer.present(engine, &device, &queue, &mut encoder, 
                            None, frame.texture.format(), &view, viewport, &debug.overlay());
                        engine.submit(queue, encoder);
                        frame.present();

                        // Update the mouse cursor
                        window.set_cursor(iced_winit::conversion::mouse_interaction(
                            state.mouse_interaction()));
                    }
                    Err(error) => match error {
                        wgpu::SurfaceError::OutOfMemory => {
                            panic!("Swapchain error: {error}. Rendering cannot continue.")
                        }
                        _ => {
                            // Try rendering again next frame.
                            window.request_redraw();
                        }
                    }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                *cursor_position = Some(position);

                match mouse_state {
                    ElementState::Pressed => {
                        if prev_pos.0 > 0.0 {
                            let h = window.inner_size().height;

                            let diff_x = (position.x - prev_pos.0) / h as f64;
                            let diff_y = (position.y - prev_pos.1) / h as f64;

                            let curr_offset = scene.borrow().offset();
                            let curr_scale = scene.borrow().scale();
                            scene.borrow_mut().set_offset(
                                curr_offset.0 - diff_x as f32 * curr_scale,
                                curr_offset.1 + diff_y as f32 * curr_scale
                            )
                        }

                        prev_pos.0 = position.x;
                        prev_pos.1 = position.y;
                    }
                    ElementState::Released => {
                        *prev_pos = (-1.0, -1.0);
                    }
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                if let MouseScrollDelta::LineDelta(_, h) = delta {
                    let curr_scale = scene.borrow().scale();
                    let new_scale = if h > 0.0 {
                        curr_scale / 1.05
                    }
                    else {
                        curr_scale * 1.05
                    };
                    scene.borrow_mut().set_scale(new_scale);
                }
            }
            WindowEvent::MouseInput { state, button, ..} => {
                if let MouseButton::Right = button {
                    *mouse_state = state;
                }
            }
            WindowEvent::ModifiersChanged(new_modifiers) => {
                *modifiers = new_modifiers.state();
            }
            WindowEvent::Resized(_) => {
                *resized = true;
            }
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            _ => {}
        }

        // Map window event to iced event
        if let Some(event) = conversion::window_event(event,
            window.scale_factor(), *modifiers) {
            state.queue_event(event);
        }

        // If there are events pending
        if !state.is_queue_empty() {
            // We update iced
            let _ = state.update(viewport.logical_size(), cursor_position
                    .map(|p| {conversion::cursor_position(p,viewport.scale_factor())})
                    .map(mouse::Cursor::Available)
                    .unwrap_or(mouse::Cursor::Unavailable),
                renderer, &Theme::Dark, &renderer::Style {text_color: Color::WHITE},
                clipboard, debug);

            // and request a redraw
            window.request_redraw();
        }
    }
}

pub fn main() -> Result<(), winit::error::EventLoopError> {
    // Initialize winit
    let event_loop = EventLoop::new().expect("Opening winit Event Loop");
    let mut runner = Runner::Loading;

    // Run the event loop forever
    event_loop.run_app(&mut runner)
}