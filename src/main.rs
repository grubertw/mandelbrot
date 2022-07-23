mod controls;
mod scene;

use controls::Controls;
use scene::Scene;

use iced_wgpu::{wgpu, Backend, Renderer, Settings, Viewport};
use iced_winit::{
    conversion, futures, program, renderer, winit, Clipboard, Color, Debug,
    Size,
};

use winit::{
    dpi::PhysicalPosition,
    event::{Event, ModifiersState, WindowEvent, MouseScrollDelta, ElementState, MouseButton},
    event_loop::{ControlFlow, EventLoop},
};

pub fn main() {
    // Initialize winit
    let event_loop = EventLoop::new();
    let window = winit::window::Window::new(&event_loop).unwrap();
    window.set_inner_size(winit::dpi::PhysicalSize::new(1200, 800));

    let physical_size = window.inner_size();
    let mut viewport = Viewport::with_physical_size(
        Size::new(physical_size.width, physical_size.height),
        window.scale_factor(),
    );
    let mut cursor_position = PhysicalPosition::new(-1.0, -1.0);
    let mut modifiers = ModifiersState::default();
    let mut clipboard = Clipboard::connect(&window);

    // Initialize wgpu
    let default_backend = wgpu::Backends::PRIMARY;
    let backend = wgpu::util::backend_bits_from_env().unwrap_or(default_backend);
    let instance = wgpu::Instance::new(backend);
    let surface = unsafe { instance.create_surface(&window) };

    let (format, (device, queue)) = futures::executor::block_on(async {
        let adapter = wgpu::util::initialize_adapter_from_env_or_default(
            &instance, backend, Some(&surface),
        )
        .await
        .expect("No suitable GPU adapters found on the system!");

        (
            surface
                .get_supported_formats(&adapter)
                .first()
                .copied()
                .expect("Get preferred format"),
            adapter
                .request_device(
                    &wgpu::DeviceDescriptor {
                        label: None,
                        features: adapter.features() & wgpu::Features::default(),
                        limits: wgpu::Limits::default(),
                    },
                    None,
                )
                .await
                .expect("Request device"),
        )
    });

    surface.configure(
        &device, &wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: physical_size.width, height: physical_size.height,
            present_mode: wgpu::PresentMode::AutoVsync,
        },
    );

    let mut resized = false;

    // Initialize staging belt
    let mut staging_belt = wgpu::util::StagingBelt::new(5 * 1024);

    // Initialize scene and GUI controls
    let mut scene = Scene::new(&device, format);
    let controls = Controls::new();

    // Initialize iced
    let mut debug = Debug::new();
    let mut renderer = Renderer::new(Backend::new(&device, Settings::default(), format));

    let mut gui_state = program::State::new(
        controls, viewport.logical_size(), &mut renderer, &mut debug,
    );

    // For tracking mouse movment and shifting (offsetting) the fractal
    let mut prev_pos: (f64, f64) = (-1.0, -1.0);
    let mut mouse_state = ElementState::Released;

    // Run event loop
    event_loop.run(move |event, _, control_flow| {
        // You should change this if you want to render continuosly
        *control_flow = ControlFlow::Wait;

        match event {
            Event::WindowEvent { event, .. } => {
                match event {
                    WindowEvent::CursorMoved { position, .. } => {
                        cursor_position = position;

                        match mouse_state {
                            ElementState::Pressed => {
                                if prev_pos.0 > 0.0 {
                                    let h = window.inner_size().height;

                                    let diff_x = (position.x - prev_pos.0) / h as f64;
                                    let diff_y = (position.y - prev_pos.1) / h as f64;

                                    let curr_offset = scene.offset();
                                    scene.set_offset(
                                        curr_offset.0 - diff_x as f32 * scene.scale(),
                                        curr_offset.1 + diff_y as f32 * scene.scale()
                                    )
                                }

                                prev_pos.0 = position.x;
                                prev_pos.1 = position.y;
                            }
                            ElementState::Released => {
                                prev_pos = (-1.0, -1.0);
                            }
                        }
                    }
                    WindowEvent::ModifiersChanged(new_modifiers) => {
                        modifiers = new_modifiers;
                    }
                    WindowEvent::Resized(new_size) => {
                        viewport = Viewport::with_physical_size(
                            Size::new(new_size.width, new_size.height),
                            window.scale_factor(),
                        );

                        resized = true;
                    }
                    WindowEvent::MouseWheel { delta, .. } => {
                        match delta {
                            MouseScrollDelta::LineDelta(_, h) => {
                                if h > 0.0 {
                                    scene.set_scale(scene.scale() / 1.05);
                                }
                                else if h < 0.0 {
                                    scene.set_scale(scene.scale() * 1.05);
                                }
                            }
                            _ => {}
                        }
                    }
                    WindowEvent::MouseInput { state, button, ..} => {
                        match button {
                            MouseButton::Right => {
                                mouse_state = state;
                            }
                            _ => {}
                        }
                    }
                    WindowEvent::CloseRequested => {
                        *control_flow = ControlFlow::Exit;
                    }
                    _ => {}
                }

                // Map window event to iced event
                if let Some(event) = iced_winit::conversion::window_event(
                    &event, window.scale_factor(),
                    modifiers) { gui_state.queue_event(event); }
            }
            Event::MainEventsCleared => {
                // If there are events pending
                if !gui_state.is_queue_empty() {
                    // We update iced
                    gui_state.update(
                        viewport.logical_size(),
                        conversion::cursor_position(
                            cursor_position, viewport.scale_factor(),
                        ),
                        &mut renderer, &iced_wgpu::Theme::Dark,
                        &renderer::Style { text_color: Color::WHITE },
                        &mut clipboard, &mut debug,
                    );

                    // and request a redraw
                    window.request_redraw();
                }
            }
            Event::RedrawRequested(_) => {
                if resized {
                    let size = window.inner_size();
                    scene.set_aspect_ratio(size.width as f32 / size.height as f32);

                    surface.configure(
                        &device, &wgpu::SurfaceConfiguration {
                            format, usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                            width: size.width, height: size.height,
                            present_mode: wgpu::PresentMode::AutoVsync,
                        },
                    );

                    resized = false;
                }

                match surface.get_current_texture() {
                    Ok(frame) => {
                        // Update the scene with values from the controls.
                        scene.set_max_iterations(gui_state.program().max_iterations());
                        
                        let c_freq = gui_state.program().rgb_freq();
                        scene.set_red_freq(c_freq.r);
                        scene.set_green_freq(c_freq.g);
                        scene.set_blue_freq(c_freq.b);

                        let c_phase = gui_state.program().rgb_phase();
                        scene.set_red_phase(c_phase.r);
                        scene.set_green_phase(c_phase.g);
                        scene.set_blue_phase(c_phase.b);

                        let mut encoder = device.create_command_encoder(
                            &wgpu::CommandEncoderDescriptor { label: None },
                        );

                        let view = frame.texture.create_view(
                            &wgpu::TextureViewDescriptor::default()
                        );

                       // Draw the scene
                       scene.draw(&queue, &view, &mut encoder);

                        // And then iced on top
                        renderer.with_primitives(|backend, primitive| {
                            backend.present(
                                &device, &mut staging_belt, &mut encoder,
                                &view, primitive, &viewport, &debug.overlay(),
                            );
                        });

                        // Then we submit the work
                        staging_belt.finish();
                        queue.submit(Some(encoder.finish()));
                        frame.present();

                        // Update the mouse cursor
                         window.set_cursor_icon(
                             iced_winit::conversion::mouse_interaction(
                                gui_state.mouse_interaction(),
                             ),
                         );

                        // And recall staging buffers
                        staging_belt.recall();
                    }
                    Err(error) => match error {
                        wgpu::SurfaceError::OutOfMemory => {
                            panic!("Swapchain error: {}. Rendering cannot continue.", error)
                        }
                        _ => {
                            // Try rendering again next frame.
                            window.request_redraw();
                        }
                    },
                }
            }
            _ => {}
        }
    })
}