#[macro_use] extern crate glium;
#[macro_use] extern crate log;

mod logger;

use glium::{DisplayBuild, Surface};
use glium::glutin::{Event, VirtualKeyCode, MouseScrollDelta, ElementState, MouseButton, WindowBuilder};
use MouseScrollDelta::LineDelta;

#[derive(Copy, Clone)]
struct Vertex {
    position: [f32; 2],
}

implement_vertex!(Vertex, position);

fn main() {
    let logger_rc = logger::init();
    assert_eq!(logger_rc.is_ok(), true);

    // Initial window size, which may be changed by the user.
    let mut window_size: (u32, u32) = (800, 800);
    // inital maxmimum number of iterations (orbits)
    let mut max_iterations: u32 = 100;
    // Interval to increase max_iterations
    let mut iteration_interval: u32 = 1;
    // inital scale
    let mut scale: f64 = 1.0;
    // Interval to increase the scale
    let mut scale_interval: f64 = 1.05;
    // start x and y offsets at 0. mouse drags will increase/
    // decrease these values.
    let mut offset: (f64,f64) = (0.0,0.0);
    let mut prev_pos: (i32,i32) = (-1,-1);
    // frequency adjustment for color sine waves
    let mut red_freq: f32 = 1.0;
    let mut blue_freq: f32 = 1.0;
    let mut green_freq: f32 = 1.0;
    // phase adjustment for color sine waves
    let mut red_phase: f32 = 0.0;
    let mut blue_phase: f32 = 0.0; // alt 2.0
    let mut green_phase: f32 = 0.0; // alt 4.0
    // for tracking mouse up/down state.
    let mut mouse_state = ElementState::Released;

    // Create the window
    let display = WindowBuilder::new()
        .with_title("Mandelbrot Set".to_string())
        .with_dimensions(window_size.0, window_size.1)
        .build_glium()
        .unwrap();

    // Compile the shaders
    let program = glium::Program::from_source(
        &display,
        include_str!("mandelbrot.glslv"),
        include_str!("mandelbrot.glslf"),
        None).unwrap();

    // Render 2 triangles covering the whole screen
    let vertices = [
        // Top-left corner
        Vertex{ position: [-1.0,  1.0] },
        Vertex{ position: [ 1.0,  1.0] },
        Vertex{ position: [-1.0, -1.0] },

        // Bottom-right corner
        Vertex { position: [-1.0, -1.0] },
        Vertex { position: [ 1.0,  1.0] },
        Vertex { position: [ 1.0, -1.0] },
    ];

    let vertex_buffer = glium::VertexBuffer::new(&display, &vertices).unwrap();

    loop {
        let mut target = display.draw();
        // Draw the vertices
        target.draw(&vertex_buffer,
                    &glium::index::NoIndices(glium::index::PrimitiveType::TrianglesList),
                    &program,
                    &uniform! {max_iterations: max_iterations, 
                               window_height: window_size.1,
                               scale: scale,
                               offset_x: offset.0,
                               offset_y: offset.1,
                               red_freq: red_freq,
                               blue_freq: blue_freq,
                               green_freq: green_freq,
                               red_phase: red_phase,
                               blue_phase: blue_phase,
                               green_phase: green_phase},
                    &Default::default()).unwrap();
        target.finish().unwrap();

        for event in display.poll_events() {
            match event {
                // the window has been closed by the user:
                Event::Closed => return,
                // Quit on Esc:
                Event::KeyboardInput(_ , _, Some(VirtualKeyCode::Escape)) => return,
                // Window was resized.
                Event::Resized(w,h) => {window_size.0 = w; window_size.1 = h;},

                // - key decrements max_iterations
                Event::KeyboardInput(_,_, Some(VirtualKeyCode::Minus)) => {
                    if max_iterations > iteration_interval {
                        max_iterations -= iteration_interval;
                    }
                    else if iteration_interval > 2 {
                        iteration_interval -= 1;
                    }
                    info!("max_iterations decremented to {}", max_iterations);
                },
                // = key increments max_iterations
                Event::KeyboardInput(_,_, Some(VirtualKeyCode::Equals)) => {
                    max_iterations += iteration_interval;
                    info!("max_iterations incremented to {}", max_iterations);
                },
                // // i key increments the iteration_interval
                // Event::KeyboardInput(_,_, Some(VirtualKeyCode::I)) => {
                //     iteration_interval += 1;
                //     info!("iteration_interval incremented to {}", iteration_interval);
                // },
                // // u key decrements the iteration_interval
                // Event::KeyboardInput(_,_, Some(VirtualKeyCode::U)) => {
                //     if iteration_interval > 1 {
                //         iteration_interval -= 1;
                //     }
                //     info!("iteration_interval decremented to {}", iteration_interval);
                // },
                // s key increments the scale_interval
                Event::KeyboardInput(_,_, Some(VirtualKeyCode::S)) => {
                    scale_interval += 0.01;
                    info!("scale_interval incremented to {}", scale_interval);
                },
                // a key decrements the scale_interval
                Event::KeyboardInput(_,_, Some(VirtualKeyCode::A)) => {
                    if scale_interval > 1.05 {
                        scale_interval -= 0.01;
                    }
                    info!("scale_interval decremented to {}", scale_interval);
                },
                // r key increments the red frequency
                Event::KeyboardInput(_,_, Some(VirtualKeyCode::R)) => {
                    red_freq += 0.01;
                    info!("red_freq incremented to {}", red_freq);
                },
                // e key decrements the red frequency 
                Event::KeyboardInput(_,_, Some(VirtualKeyCode::E)) => {
                    if red_freq > 0.0 {
                        red_freq -= 0.01;
                    }
                    info!("red_freq decremented to {}", red_freq);
                },
                // g key increments the green frequency
                Event::KeyboardInput(_,_, Some(VirtualKeyCode::G)) => {
                    green_freq += 0.01;
                    info!("green_freq incremented to {}", green_freq);
                },
                // f key decrements the green frequency 
                Event::KeyboardInput(_,_, Some(VirtualKeyCode::F)) => {
                    if green_freq > 0.0 {
                        green_freq -= 0.01;
                    }
                    info!("green_freq decremented to {}", green_freq);
                },
                // b key increments the blue frequency
                Event::KeyboardInput(_,_, Some(VirtualKeyCode::B)) => {
                    blue_freq += 0.01;
                    info!("red_freq incremented to {}", blue_freq);
                },
                // v key decrements the blue frequency 
                Event::KeyboardInput(_,_, Some(VirtualKeyCode::V)) => {
                    if blue_freq > 0.0 {
                        blue_freq -= 0.01;
                    }
                    info!("blue_freq decremented to {}", blue_freq);
                },
                // y key increments the red phase
                Event::KeyboardInput(_,_, Some(VirtualKeyCode::Y)) => {
                    red_phase += 0.1;
                    info!("red_phase incremented to {}", red_phase);
                },
                // t key decrements the red phase
                Event::KeyboardInput(_,_, Some(VirtualKeyCode::T)) => {
                    red_phase -= 0.1;
                    info!("red_phase decremented to {}", red_phase);
                },
                // j key increments the green phase
                Event::KeyboardInput(_,_, Some(VirtualKeyCode::J)) => {
                    green_phase += 0.01;
                    info!("green_phase incremented to {}", green_phase);
                },
                // h key decrements the green phase
                Event::KeyboardInput(_,_, Some(VirtualKeyCode::H)) => {
                    green_phase -= 0.1;
                    info!("green_phase decremented to {}", green_phase);
                },
                // m key increments the blue phase
                Event::KeyboardInput(_,_, Some(VirtualKeyCode::M)) => {
                    blue_phase += 0.1;
                    info!("blue_phase incremented to {}", blue_phase);
                },
                // n key decrements the blue phase
                Event::KeyboardInput(_,_, Some(VirtualKeyCode::N)) => {
                    blue_phase -= 0.1;
                    info!("blue_phase decremented to {}", blue_phase);
                },

                // Zoom in and out with the mouse wheel.
                Event::MouseWheel(delta, _) => {
                    match delta {
                        LineDelta(_, h) => {
                            if h > 0.0 {
                                scale /= scale_interval;
                                info!("scale decremented to {}", scale);
                            }
                            else if h < 0.0 {
                                scale *= scale_interval;
                                info!("scale incremented to {}", scale);
                            }
                        },
                        _ => ()
                    }
                },

                // shift the drawing by clicking with the left mouse
                // button and dragging.
                Event::MouseInput(m_state, m_button) => {
                    match m_button {
                        MouseButton::Left => {
                            mouse_state = m_state;
                        }
                        _ => ()
                    }
                },
                Event::MouseMoved(curr_x, curr_y) => {
                    match mouse_state {
                        ElementState::Pressed => {
                            if prev_pos.0 >= 0 {
                                let diff_x = curr_x - prev_pos.0;
                                let diff_y = curr_y - prev_pos.1;

                                offset.0 -= (diff_x as f64) * scale;
                                offset.1 += (diff_y as f64) * scale;

                                info!("offset shifted ({},{})", offset.0, offset.1);
                            }

                            prev_pos.0 = curr_x;
                            prev_pos.1 = curr_y;
                        },
                        ElementState::Released => {
                            prev_pos = (-1,-1);
                        }
                    }
                },

                // Keep going on all other events.
                _ => ()
            }
        }
    }
}