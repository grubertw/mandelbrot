use bytemuck;

use super::numerics::{Df, ComplexDf};
use super::signals;
use super::scout_engine::{ScoutConfig, HeuristicConfig, TileId, TileLevel, ScoutEngine};

use futures::channel;
use futures::executor;
use iced_wgpu::wgpu;
use iced_wgpu::wgpu::BufferAsyncError;
use wgpu::util::DeviceExt;
use iced_winit::winit::window::Window;

use rug::{Float, Complex};
use log::{trace, debug};
use std::collections::HashMap;
use std::sync::Arc;
use std::mem::size_of;
use std::time;

const MAX_REF_ORBIT: u32 = 8192;
const INIT_RUG_PRECISION: u32 = 128;
const K: f32 = 0.25; // Multiplied by scale, and often used as a radius of validity test
const PERTURB_THRESHOLD: f32 = 1e-5;

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
    ref_len:     u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct GpuFeedbackOut {
    max_lambda_re_hi:  f32,
    max_lambda_re_lo:  f32,
    max_lambda_im_hi:  f32,
    max_lambda_im_lo:  f32,
    max_delta_z_re_hi: f32,
    max_delta_z_re_lo: f32,
    max_delta_z_im_hi: f32,
    max_delta_z_im_lo: f32,
    escape_ratio:   f32,
}
unsafe impl bytemuck::Pod for GpuFeedbackOut {}
unsafe impl bytemuck::Zeroable for GpuFeedbackOut {}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct DebugOut {
    c_ref_re_hi: f32,
    c_ref_re_lo: f32,
    c_ref_im_hi: f32,
    c_ref_im_lo: f32,
    delta_c_re_hi: f32,
    delta_c_re_lo: f32,
    delta_c_im_hi: f32,
    delta_c_im_lo: f32,
    perturb_escape_seq: u32,
    last_valid_i: u32,
    abs_i: u32,
    last_valid_z_re_hi: f32,
    last_valid_z_re_lo: f32,
    last_valid_z_im_hi: f32,
    last_valid_z_im_lo: f32,
}
unsafe impl bytemuck::Pod for DebugOut {}
unsafe impl bytemuck::Zeroable for DebugOut {}

struct OrbitMeta {
    base_row: u32,
    orbit_len: u32,
}

#[derive(Debug)]
pub struct Scene {
    frame_id: u64,
    frame_timestamp: time::Instant,
    scale: Float,
    scale_factor: Float,
    center: Complex, // scaled and shifted with mouse drag
    width: f64,
    height: f64,
    pix_dx: Float,
    pix_dy: Float,
    scout_engine: ScoutEngine,
    active_tiles: HashMap<TileId, signals::TileOrbitViewDf>,
    uniform: SceneUniform,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    gpu_feedback_buffer: wgpu::Buffer,
    gpu_feedback_readback: wgpu::Buffer,
    gpu_feedback_bind_group: wgpu::BindGroup,
    debug_buffer: wgpu::Buffer,
    debug_readback: wgpu::Buffer,
    debug_bind_group: wgpu::BindGroup,
    ref_orbit_texture: wgpu::Texture,
    ref_orbit_bind_group: wgpu::BindGroup,
    pipeline: wgpu::RenderPipeline,
}

impl Scene {
    pub fn new(window: Arc<Window>, device: &wgpu::Device, 
        texture_format: wgpu::TextureFormat,
        width: f64, height: f64,
    ) -> Scene {
        let center = Complex::with_val(INIT_RUG_PRECISION, (-0.75, 0.0));
        let c_df = ComplexDf::from_complex(&center);

        let scale = Float::with_val(INIT_RUG_PRECISION, 3.5);
        let scale_df = Df::from_float(&scale);

        let pix_dx = Float::with_val(INIT_RUG_PRECISION, &scale / width);
        let pix_dy = Float::with_val(INIT_RUG_PRECISION, &scale / height);
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
            ref_len: 0,
        };

        let (uniform_buffer, bind_group, 
            gpu_feedback_buffer, gpu_feedback_readback, gpu_feedback_bind_group,
            debug_buffer, debug_readback, debug_bind_group, 
            ref_orbit_texture, ref_orbit_bind_group,
            pipeline) = 
                build_pipeline(device, uniform, texture_format);

        // Configure ScoutEngine (our single source of truth for reference orbits)
        let scout_config = ScoutConfig {
            max_orbits: MAX_REF_ORBIT,
            max_iterations_ref: uniform.max_iter,
            rug_precision: INIT_RUG_PRECISION,
            heuristic_config: HeuristicConfig {
                weight_1: 0.0
            },
            tile_levels: vec![
                //TileLevel {
                //    level: 0,
                //    tile_size: Float::with_val(INIT_RUG_PRECISION, K),
                //    max_orbits_per_tile: 3
                //},
                TileLevel {
                    level: 1,
                    tile_size: Float::with_val(INIT_RUG_PRECISION, 5.0 * PERTURB_THRESHOLD),
                    influence_radius_factor: 1.25,
                    max_orbits_per_tile: 6,
                },
            ],
            exploration_budget: 5.0,
        };

        let scout_engine = ScoutEngine::new(window, scout_config);
        let active_tiles = HashMap::<TileId, signals::TileOrbitViewDf>::new();

        Scene { 
            frame_id: 0, frame_timestamp: time::Instant::now(), scale, scale_factor: Float::with_val(80, 1.04),
            center, width, height, pix_dx, pix_dy, scout_engine, active_tiles,
            uniform, uniform_buffer, bind_group, 
            gpu_feedback_buffer, gpu_feedback_readback, gpu_feedback_bind_group,
            ref_orbit_texture, ref_orbit_bind_group,
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
        render_pass.set_bind_group(3, &self.ref_orbit_bind_group, &[]);
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
                debug!("  max_lambda = ({}, {}) ({}, {})", dbg.max_lambda_re_hi, dbg.max_lambda_re_lo, dbg.max_lambda_im_hi, dbg.max_lambda_im_lo);
                debug!("  max_delta_z = ({}, {}) ({}, {})", dbg.max_delta_z_re_hi, dbg.max_delta_z_re_lo, dbg.max_delta_z_im_hi, dbg.max_delta_z_im_lo);
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

                debug!("FROM CPU (Secne struct)");
                debug!("  center={:?}", self.center);
                
                debug!("FROM CPU (scene uniform):");
                debug!("  pix_dx = ({}, {})", self.uniform.pix_dx_hi, self.uniform.pix_dx_lo);
                debug!("  pix_dy = ({}, {})", self.uniform.pix_dy_hi, self.uniform.pix_dy_lo);
                debug!("  scale = ({}, {})", self.uniform.scale_hi, self.uniform.scale_lo);
                debug!("FROM GPU (via debug buffer at px=0 py=0):");
                debug!("  c_ref   = (({}, {}) ({}, {}))", dbg.c_ref_re_hi, dbg.c_ref_re_lo, dbg.c_ref_im_hi, dbg.c_ref_im_lo);
                debug!("  delta_c = (({}, {}) ({}, {}))", dbg.delta_c_re_hi, dbg.delta_c_re_lo, dbg.delta_c_im_hi, dbg.delta_c_im_lo);
                debug!("  perturb_escape_seq = {}", dbg.perturb_escape_seq);
                debug!("  last_valid_i = {}", dbg.last_valid_i);
                debug!("  abs_i = {}", dbg.abs_i);
                debug!("  last_valid_z = ({},{}) ({},{})", dbg.last_valid_z_re_hi, dbg.last_valid_z_re_lo, dbg.last_valid_z_im_hi,dbg.last_valid_z_im_lo);
    
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

    pub fn upload_reference_orbit(&mut self, ref_orb_df: signals::ReferenceOrbitDf, queue: &wgpu::Queue) {
        let ref_orb_len: u32 = ref_orb_df.orbit_re_hi.len() as u32;
        debug!("Uploading Reference Orbit {} of size {} to GPU. c_ref={:?}", 
            ref_orb_df.orbit_id, ref_orb_len, ref_orb_df.c_ref);

        let mut orbit_texture_data = Vec::<f32>::with_capacity(ref_orb_len as usize * 4);
        for re_hi in ref_orb_df.orbit_re_hi {
            orbit_texture_data.push(re_hi);
        }
        for re_lo in ref_orb_df.orbit_re_lo {
            orbit_texture_data.push(re_lo);
        }
        for im_hi in ref_orb_df.orbit_im_hi {
            orbit_texture_data.push(im_hi);
        }
        for im_lo in ref_orb_df.orbit_im_lo {
            orbit_texture_data.push(im_lo);
        }

        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.ref_orbit_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&orbit_texture_data),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(std::num::NonZeroU32::new(
                    MAX_REF_ORBIT as u32 * std::mem::size_of::<f32>() as u32
                ).unwrap().into()),
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width: MAX_REF_ORBIT as u32,
                height: 4,
                depth_or_array_layers: 1,
            },
        );

        self.uniform.ref_len = ref_orb_len;
    }

    pub fn stamp_frame(&mut self) {
        self.frame_id += 1;
        self.frame_timestamp = time::Instant::now();
    }

    pub fn take_camera_snapshot(&mut self) {
        let cam_snap = signals::CameraSnapshot {
            frame_stamp: signals::FrameStamp {
                frame_id: self.frame_id,
                timestamp: self.frame_timestamp
            },
            center: self.center.clone(),
            scale: self.scale.clone(),
        };

        self.scout_engine.submit_camera_snapshot(cam_snap);
    }

    pub fn query_tile_orbits(&mut self, queue: &wgpu::Queue) {        
        let tiles = self.scout_engine.query_tiles_under_viewport(
            &self.center, &self.scale, self.width / self.height);

        // Purely for testing only!
        if tiles.len() == 0 {
            debug!("Query Tile Orbits found NO tiles!");
            return;
        }
        let tile_opt = tiles.iter().find(|tile| tile.orbits.len() > 0);
        if let Some(tile) = tile_opt {
            debug!("Taking orbit from tile {:?}", tile.tile);
            let ref_orb_df = &tile.orbits[0];
            self.upload_reference_orbit(ref_orb_df.clone(), queue);
        }
    }
}

fn build_pipeline(
    device: &wgpu::Device,
    uniform: SceneUniform,
    texture_format: wgpu::TextureFormat,
) -> (wgpu::Buffer, wgpu::BindGroup, // Scene Unifom 
      wgpu::Buffer, wgpu::Buffer, wgpu::BindGroup, // Gpu Feedback Buffers
      wgpu::Buffer, wgpu::Buffer, wgpu::BindGroup, // Debug Buffers
      wgpu::Texture, wgpu::BindGroup, // Reference Orbit Texture
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
    // Reference Orbit Texture (single orbit, step 1)
    ///////////////////////////////////////////////////////
    let ref_orbit_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("ref_orbit_texture"),
        size: wgpu::Extent3d {
            width: MAX_REF_ORBIT as u32,
            height: 4, // re_hi, re_lo, im_hi, im_lo
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    let ref_orbit_texture_view =
        ref_orbit_texture.create_view(&wgpu::TextureViewDescriptor::default());

    let ref_orbit_bgl = device.create_bind_group_layout(
        &wgpu::BindGroupLayoutDescriptor {
            label: Some("ref_orbit_texture_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        }
    );

    let ref_orbit_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("ref_orbit_texture_bg"),
        layout: &ref_orbit_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&ref_orbit_texture_view),
            },
        ],
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
              &debug_bind_group_layout,
              &ref_orbit_bgl],
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
        ref_orbit_texture, ref_orbit_bg,
        pipeline)
}