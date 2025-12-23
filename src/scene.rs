use bytemuck;

use iced_winit::futures::futures;
use iced_wgpu::wgpu;
use iced_wgpu::wgpu::BufferAsyncError;
use wgpu::util::DeviceExt;

use rug::{Float, Complex};
use log::{trace, debug};
use std::mem::size_of;

fn split_mpfr_to_df(x: &Float) -> (f32, f32) {
    let hi = x.to_f64();

    // residual = x - hi
    let mut residual = Float::with_val(200, x);
    residual -= hi;

    let lo = residual.to_f64();
    (hi as f32, lo as f32)
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SceneUniform {
    center_x_hi: f32,
    center_x_lo: f32,
    center_y_hi: f32,
    center_y_lo: f32,
    scale_hi:    f32,
    scale_lo:    f32,
    pix_dx_hi:   f32, 
    pix_dx_lo:   f32,
    pix_dy_hi:   f32, 
    pix_dy_lo:   f32,
    width:       f32,
    height:      f32,
    max_iter:    u32,
    red_freq:    f32,
    blue_freq:   f32,
    green_freq:  f32,
    red_phase:   f32,
    blue_phase:  f32,
    green_phase: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct DebugOut {
    zx_lo: f32,
    zx_hi: f32,
    zy_hi: f32,
    zy_lo: f32,
    cr_hi: f32,
    cr_lo: f32,
    ci_hi: f32,
    ci_lo: f32,
    pix_dx_hi: f32,
    pix_dx_lo: f32,
    pix_dy_hi: f32,
    pix_dy_lo: f32,
}
unsafe impl bytemuck::Pod for DebugOut {}
unsafe impl bytemuck::Zeroable for DebugOut {}

#[derive(Debug)]
pub struct Scene {
    scale: Float,
    scale_factor: Float,
    center: Complex, // scaled and shifted with mouse drag
    width: Float,
    height: Float,
    uniform: SceneUniform,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    debug_buffer: wgpu::Buffer,
    debug_readback: wgpu::Buffer,
    debug_bind_group: wgpu::BindGroup,
    pipeline: wgpu::RenderPipeline
}

impl Scene {
    pub fn new(device: &wgpu::Device, 
        texture_format: wgpu::TextureFormat,
        width: f32, height: f32,
    ) -> Scene {
        let c = Complex::with_val(80, (-0.75, 0.0));
        let cxdf = split_mpfr_to_df(&c.real());
        let cydf = split_mpfr_to_df(&c.imag());
        let w = Float::with_val(80, width);
        let h = Float::with_val(80, height);
        let scale = Float::with_val(80, 3.5);
        let scale_df = split_mpfr_to_df(&scale);

        let pix_dx = Float::with_val(200, &scale / &w);
        let pix_dy = Float::with_val(200, &scale / &h);
        let pix_dx_df = split_mpfr_to_df(&pix_dx);
        let pix_dy_df = split_mpfr_to_df(&pix_dy);

        let uniform = SceneUniform {
            center_x_hi: cxdf.0,
            center_x_lo: cxdf.1,
            center_y_hi: cydf.0,
            center_y_lo: cydf.1,
            scale_hi:    scale_df.0,
            scale_lo:    scale_df.1,
            pix_dx_hi:   pix_dx_df.0,
            pix_dx_lo:   pix_dx_df.1,
            pix_dy_hi:   pix_dy_df.0,
            pix_dy_lo:   pix_dy_df.1,
            width,
            height,
            max_iter:    500,
            red_freq:    1.0,
            blue_freq:   1.0,
            green_freq:  1.0,
            red_phase:   0.0,
            blue_phase:  0.0,
            green_phase: 0.0,
        };

        let (uniform_buffer, bind_group, debug_buffer, debug_readback, debug_bind_group, pipeline) = 
            build_pipeline(device, uniform, texture_format);

        Scene { 
            scale, scale_factor: Float::with_val(80, 1.04),
            center: c, width: w, height: h,
            uniform, 
            uniform_buffer, bind_group, debug_buffer, debug_readback, debug_bind_group, pipeline 
        }
    }

    pub fn clear<'a>(target: &'a wgpu::TextureView, encoder: &'a mut wgpu::CommandEncoder) 
            -> wgpu::RenderPass<'a> {
        trace!("Clear");
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        })
    }

    pub fn draw<'a>(&'a self, queue: &wgpu::Queue, render_pass: &mut wgpu::RenderPass<'a>) {
        trace!("Draw with uniform={:?} size={} bytes: {:?}", 
            self.uniform, size_of::<SceneUniform>(), &bytemuck::bytes_of(&self.uniform));
        // Uniforms must be updated on every draw operation.
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[self.uniform]));

        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        render_pass.set_bind_group(1, &self.debug_bind_group, &[]);
        render_pass.draw(0..3, 0..1);
    }

    pub fn read_debug<'a>(&'a self, device: &wgpu::Device, queue: &wgpu::Queue) {
        // 1) create encoder, copy storage -> readback
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("debug copy encoder"),
        });

        encoder.copy_buffer_to_buffer(
            &self.debug_buffer, // src
            0,
            &self.debug_readback, // dst
            0,
            std::mem::size_of::<DebugOut>() as u64,
        );

        // submit the copy
        queue.submit(Some(encoder.finish()));

        let buffer_slice = self.debug_readback.slice(..);

        let (sender, receiver) = futures::channel::oneshot::channel::<Result<(), BufferAsyncError>>();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| {
            sender.send(v).unwrap();
        });

        device.poll(wgpu::Maintain::Wait);

        futures::executor::block_on(async {
            if let Ok(Ok(_)) = receiver.await {
                let data = buffer_slice.get_mapped_range();
                let dbg = bytemuck::from_bytes::<DebugOut>(&data[..]).clone();

                debug!("GPU DEBUG:");
                debug!("  zx = ({}, {})", dbg.zx_hi, dbg.zx_lo);
                debug!("  zy = ({}, {})", dbg.zy_hi, dbg.zy_lo);
                debug!("  c  = ({}, {})  ({}, {})", dbg.cr_hi, dbg.cr_lo, dbg.ci_hi, dbg.ci_lo);
                debug!("  pix_dx = ({}, {})", dbg.pix_dx_hi, dbg.pix_dx_lo);
                debug!("  pix_dy = ({}, {})", dbg.pix_dy_hi, dbg.pix_dy_lo);

                drop(data);
                self.debug_readback.unmap();
            }
        });
    }


    pub fn set_max_iterations(&mut self, max_iterations: u32) {
        self.uniform.max_iter = max_iterations;
    }

    pub fn set_window_size(&mut self, width: f32, height: f32) {
        self.uniform.width = width;
        self.uniform.height = height;

        debug!("Window size changed w={} h={}", width, height);
    }

    pub fn change_scale(&mut self, increase: bool) -> String {
        if increase {
            self.scale *= &self.scale_factor;
        } else {
            self.scale /= &self.scale_factor;
        }

        let scale_df = split_mpfr_to_df(&self.scale);
        self.uniform.scale_hi = scale_df.0;
        self.uniform.scale_lo = scale_df.1;
        
        let pix_dx = Float::with_val(200, &self.scale / &self.width);
        let pix_dy = Float::with_val(200, &self.scale / &self.height);

        let pix_dx_df = split_mpfr_to_df(&pix_dx);
        let pix_dy_df = split_mpfr_to_df(&pix_dy);

        self.uniform.pix_dx_hi = pix_dx_df.0;
        self.uniform.pix_dx_lo = pix_dx_df.1;
        self.uniform.pix_dy_hi = pix_dy_df.0;
        self.uniform.pix_dy_lo = pix_dy_df.1;

        let s = self.scale.to_string_radix(10, None);
        debug!("Scale changed {}", s);
        s
    }

    pub fn set_center(&mut self, center_diff: (f64, f64)) -> String {
        let dx = Float::with_val(200, center_diff.0);
        let dy = Float::with_val(200, center_diff.1);

        let dx = dx * Float::with_val(200, &self.scale / &self.width);
        let dy = dy * Float::with_val(200, &self.scale / &self.height);

        let (real, imag) = self.center.as_mut_real_imag();
        *real -= &dx;
        *imag -= &dy;

        let center_x_df = split_mpfr_to_df(&real);
        let center_y_df = split_mpfr_to_df(&imag);

        self.uniform.center_x_hi = center_x_df.0;
        self.uniform.center_x_lo = center_x_df.1;
        self.uniform.center_y_hi = center_y_df.0;
        self.uniform.center_y_lo = center_y_df.1;

        let c = self.center.to_string_radix(10, None);
        debug!("Center changed {:?} ----- diff ({:?} {:?})", 
            c, dx, dy);

        c
    }

    pub fn set_rgb_freq(&mut self, rgb_val: (f32, f32, f32)) {
        let (r, g, b) = rgb_val;

        self.uniform.red_freq = r;
        self.uniform.green_freq = g;
        self.uniform.blue_freq = b;
    }

    pub fn set_rgb_phase(&mut self, rgb_val: (f32, f32, f32)) {
        let (r, g, b) = rgb_val;

        self.uniform.red_phase = r;
        self.uniform.green_phase = g;
        self.uniform.blue_phase = b;
    }
}

fn build_pipeline(
    device: &wgpu::Device,
    uniform: SceneUniform,
    texture_format: wgpu::TextureFormat,
) -> (wgpu::Buffer, wgpu::BindGroup, wgpu::Buffer, wgpu::Buffer, wgpu::BindGroup, wgpu::RenderPipeline) {
    // Compile the shader
    let shader = device.create_shader_module(wgpu::include_wgsl!("mandelbrot.wgsl"));

    // Setup the Uniform into a buffer so it can be shared and seen by the shader
    let uniform_buff = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::bytes_of(&uniform),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // Layout the Uniform in graphics memory
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None
                },
                count: None,
            }
        ],
        label: None
    });

    // Create bind group for the uniform
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buff.as_entire_binding()
            }
        ],
        label: None
    });

    let debug_size = std::mem::size_of::<DebugOut>() as u64;

    let debug_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("debug_buffer"),
        size: debug_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let debug_readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("debug_readback"),
        size: debug_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });


    let debug_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("debug bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }
            ],
        });

    let debug_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("debug bind group"),
        layout: &debug_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: debug_buffer.as_entire_binding(),
            }
        ]
    });
 
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        push_constant_ranges: &[],
        bind_group_layouts: &[&bind_group_layout, &debug_bind_group_layout],
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_main",
            targets: &[Some(wgpu::ColorTargetState {
                format: texture_format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None
    });

    (uniform_buff, bind_group, debug_buffer, debug_readback, debug_bind_group, pipeline)
}