use bytemuck;

use super::numerics::{Df, ComplexDf};

use futures::channel;
use futures::executor;
use iced_wgpu::wgpu;
use iced_wgpu::wgpu::BufferAsyncError;
use wgpu::util::DeviceExt;

use rug::{Float, Complex};
use log::{trace, debug};
use std::mem::size_of;


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
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct GpuFeedbackOut {
    max_lambda_hi:  f32,
    max_lambda_lo:  f32,
    max_delta_z_hi: f32,
    max_delta_z_lo: f32,
    escape_ratio:   f32,
}
unsafe impl bytemuck::Pod for GpuFeedbackOut {}
unsafe impl bytemuck::Zeroable for GpuFeedbackOut {}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct DebugOut {
    zx_hi: f32,
    zx_lo: f32,
    zy_hi: f32,
    zy_lo: f32,
    cr_hi: f32,
    cr_lo: f32,
    ci_hi: f32,
    ci_lo: f32,
    ax:    f32,
    ay:    f32,
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
    width: f64,
    height: f64,
    pix_dx: Float,
    pix_dy: Float,
    uniform: SceneUniform,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    gpu_feedback_buffer: wgpu::Buffer,
    gpu_feedback_readback: wgpu::Buffer,
    gpu_feedback_bind_group: wgpu::BindGroup,
    debug_buffer: wgpu::Buffer,
    debug_readback: wgpu::Buffer,
    debug_bind_group: wgpu::BindGroup,
    pipeline: wgpu::RenderPipeline
}

impl Scene {
    pub fn new(device: &wgpu::Device, 
        texture_format: wgpu::TextureFormat,
        width: f64, height: f64,
    ) -> Scene {
        let center = Complex::with_val(80, (-0.75, 0.0));
        let c_df = ComplexDf::from_complex(&center);

        let scale = Float::with_val(80, 3.5);
        let scale_df = Df::from_float(&scale);

        let pix_dx = Float::with_val(200, &scale / width);
        let pix_dy = Float::with_val(200, &scale / height);
        let pix_dx_df = Df::from_float(&pix_dx);
        let pix_dy_df = Df::from_float(&pix_dy);

        let uniform = SceneUniform {
            center_x_hi: c_df.re.hi, center_x_lo: c_df.re.lo,
            center_y_hi: c_df.im.hi, center_y_lo: c_df.im.lo,
            scale_hi:    scale_df.hi, scale_lo:    scale_df.lo,
            pix_dx_hi:   pix_dx_df.hi, pix_dx_lo:   pix_dx_df.lo,
            pix_dy_hi:   pix_dy_df.hi, pix_dy_lo:   pix_dy_df.lo,
            width: width as f32, 
            height: height as f32,
            max_iter:    500,
        };

        let (uniform_buffer, bind_group, 
            gpu_feedback_buffer, gpu_feedback_readback, gpu_feedback_bind_group,
            debug_buffer, debug_readback, debug_bind_group, pipeline) = 
                build_pipeline(device, uniform, texture_format);

        Scene { 
            scale, scale_factor: Float::with_val(80, 1.04),
            center, width, height, pix_dx, pix_dy,
            uniform, uniform_buffer, bind_group, 
            gpu_feedback_buffer, gpu_feedback_readback, gpu_feedback_bind_group,
            debug_buffer, debug_readback, debug_bind_group, pipeline 
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
        render_pass.set_bind_group(1, &self.gpu_feedback_bind_group, &[]);
        render_pass.set_bind_group(2, &self.debug_bind_group, &[]);
        render_pass.draw(0..3, 0..1);
    }

    pub fn read_gpu_feedback<'a>(&'a self, device: &wgpu::Device, queue: &wgpu::Queue) {
        // 1) create encoder, copy storage -> readback
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gpu feedback copy encoder"),
        });

        encoder.copy_buffer_to_buffer(
            &self.gpu_feedback_buffer, // src
            0,
            &self.gpu_feedback_readback, // dst
            0,
            std::mem::size_of::<GpuFeedbackOut>() as u64,
        );

        // submit the copy
        queue.submit(Some(encoder.finish()));

        let buffer_slice = self.gpu_feedback_readback.slice(..);

        let (sender, receiver) = channel::oneshot::channel::<Result<(), BufferAsyncError>>();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| {
            sender.send(v).unwrap();
        });

        device.poll(wgpu::Maintain::Wait);

        executor::block_on(async {
            if let Ok(Ok(_)) = receiver.await {
                let data = buffer_slice.get_mapped_range();
                let dbg = bytemuck::from_bytes::<GpuFeedbackOut>(&data[..]).clone();

                debug!("FROM GPU (via GPU feedback buffer):");
                debug!("  max_lambda = ({}, {})", dbg.max_lambda_hi, dbg.max_lambda_lo);
                debug!("  max_delta_z = ({}, {})", dbg.max_delta_z_hi, dbg.max_delta_z_lo);
                debug!("  escape_ratio = {}", dbg.escape_ratio);
                
                drop(data);
                self.gpu_feedback_readback.unmap();
            }
        });
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

        let (sender, receiver) = channel::oneshot::channel::<Result<(), BufferAsyncError>>();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| {
            sender.send(v).unwrap();
        });

        device.poll(wgpu::Maintain::Wait);

        executor::block_on(async {
            if let Ok(Ok(_)) = receiver.await {
                let data = buffer_slice.get_mapped_range();
                let dbg = bytemuck::from_bytes::<DebugOut>(&data[..]).clone();

                debug!("FROM CPU (scene uniform):");
                debug!("  c_ref = ({}, {}) ({}, {})", self.uniform.center_x_hi, self.uniform.center_x_lo, self.uniform.center_y_hi, self.uniform.center_y_lo);
                debug!("  pix_dx = ({}, {})", self.uniform.pix_dx_hi, self.uniform.pix_dx_lo);
                debug!("  pix_dy = ({}, {})", self.uniform.pix_dy_hi, self.uniform.pix_dy_lo);
                debug!("FROM GPU (via debug buffer):");
                debug!("  zx = ({}, {})", dbg.zx_hi, dbg.zx_lo);
                debug!("  zy = ({}, {})", dbg.zy_hi, dbg.zy_lo);
                debug!("  ax = {}", dbg.ax);
                debug!("  ay = {}", dbg.ay);
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

    pub fn set_window_size(&mut self, width: f64, height: f64) {
        self.width = width;
        self.height = height;
        self.uniform.width = width as f32;
        self.uniform.height = height as f32;

        debug!("Window size changed w={} h={}", width, height);
    }

    pub fn change_scale(&mut self, increase: bool) -> String {
        if increase {
            self.scale *= &self.scale_factor;
        } else {
            self.scale /= &self.scale_factor;
        }

        let scale_df = Df::from_float(&self.scale);
        self.uniform.scale_hi = scale_df.hi;
        self.uniform.scale_lo = scale_df.lo;
        
        self.pix_dx = self.scale.clone() / self.width;
        self.pix_dy = self.scale.clone() / self.height;

        let pix_dx_df = Df::from_float(&self.pix_dx);
        let pix_dy_df = Df::from_float(&self.pix_dy);

        self.uniform.pix_dx_hi = pix_dx_df.hi;
        self.uniform.pix_dx_lo = pix_dx_df.lo;
        self.uniform.pix_dy_hi = pix_dy_df.hi;
        self.uniform.pix_dy_lo = pix_dy_df.lo;

        let s = self.scale.to_string_radix(10, None);
        let s_pix_dx = self.pix_dx.to_string_radix(10, None);
        let s_pix_dy = self.pix_dy.to_string_radix(10, None);
        debug!("Scale changed {} --- pix_dx={} pix_dy={}", s, s_pix_dx, s_pix_dy);
        s
    }

    pub fn set_center(&mut self, center_diff: (f64, f64)) -> String {
        let dx = self.pix_dx.clone() * center_diff.0;
        let dy = self.pix_dy.clone() * center_diff.1;

        let (real, imag) = self.center.as_mut_real_imag();
        *real -= &dx;
        *imag -= &dy;

        let center_df = ComplexDf::from_complex(&self.center);
        self.uniform.center_x_hi = center_df.re.hi;
        self.uniform.center_x_lo = center_df.re.lo;
        self.uniform.center_y_hi = center_df.im.hi;
        self.uniform.center_y_lo = center_df.im.lo;

        let c = self.center.to_string_radix(10, None);
        debug!("Center changed {:?} ----- diff ({:?} {:?})", 
            c, dx, dy);

        c
    }

}

fn build_pipeline(
    device: &wgpu::Device,
    uniform: SceneUniform,
    texture_format: wgpu::TextureFormat,
) -> (wgpu::Buffer, wgpu::BindGroup, // Scene Unifom 
      wgpu::Buffer, wgpu::Buffer, wgpu::BindGroup, // Gpu Feedback Buffers
      wgpu::Buffer, wgpu::Buffer, wgpu::BindGroup, // Debug Buffers
      wgpu::RenderPipeline) {
    // Compile the shader
    let shader = device.create_shader_module(wgpu::include_wgsl!("mandelbrot.wgsl"));

    // Setup the Uniform into a buffer so it can be shared and seen by the shader
    let uniform_buff = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::bytes_of(&uniform),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    ///////////////////////////////////////////////////////
    // Scene Uniform Pipeline configuration
    ///////////////////////////////////////////////////////
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

    ///////////////////////////////////////////////////////
    // Gpu Feedback Pipeline configuration
    ///////////////////////////////////////////////////////
    let gpu_feedback_size = std::mem::size_of::<GpuFeedbackOut>() as u64;

    let gpu_feedback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gpu_feedback_buffer"),
        size: gpu_feedback_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let gpu_feedback_readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gpu_feedback_readback"),
        size: gpu_feedback_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let gpu_feedback_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gpu feedback bind group layout"),
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

    let gpu_feedback_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gpu feedback bind group"),
        layout: &gpu_feedback_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: gpu_feedback_buffer.as_entire_binding(),
            }
        ]
    });

    ///////////////////////////////////////////////////////
    // Debug Pipeline configuration
    ///////////////////////////////////////////////////////
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
 
    ///////////////////////////////////////////////////////
    // Combign into one uniform pipeline layout
    ///////////////////////////////////////////////////////
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        push_constant_ranges: &[],
        bind_group_layouts: 
            &[&bind_group_layout, 
              &gpu_feedback_bind_group_layout, 
              &debug_bind_group_layout],
    });

    ///////////////////////////////////////////////////////
    // Create Vertex and Fragment Shaders 
    ///////////////////////////////////////////////////////
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

    (uniform_buff, bind_group, 
        gpu_feedback_buffer, gpu_feedback_readback, gpu_feedback_bind_group,
        debug_buffer, debug_readback, debug_bind_group, 
        pipeline)
}