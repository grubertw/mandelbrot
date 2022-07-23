use bytemuck;
use iced_wgpu::wgpu;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 2],
}

impl Vertex {
    const ATTRIBS: [wgpu::VertexAttribute; 1] =
        wgpu::vertex_attr_array![0 => Float32x2];

    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS
        }
    }
}

// Verticies are static and will never change, as the fractal is computed 
// via the fragment shader. More triangles (vert tripples) can be added, however
// to increase GPU parallisim/performance
const VERTICES: &[Vertex] = &[
    // Top-left corner
    Vertex{ position: [ 1.0,  1.0] },
    Vertex{ position: [-1.0,  1.0] },
    Vertex{ position: [-1.0, -1.0] },

    // Bottom-right corner
    Vertex { position: [ 1.0, -1.0] },
    Vertex { position: [-1.0, -1.0] },
    Vertex { position: [ 1.0,  1.0] },
];

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SceneUniform {
    max_iterations: u32,
    aspect_ratio: f32,
    scale: f32,
    offset_x: f32,
    offset_y: f32,
    red_freq: f32,
    blue_freq: f32,
    green_freq: f32,
    red_phase: f32,
    blue_phase: f32,
    green_phase: f32,
}

pub struct Scene {
    num_verticies: u32,
    vertex_buffer: wgpu::Buffer,
    uniform: SceneUniform,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    pipeline: wgpu::RenderPipeline
}

impl Scene {
    pub fn new(
        device: &wgpu::Device,
        texture_format: wgpu::TextureFormat,
    ) -> Self {
        let num_verticies = VERTICES.len() as u32;

        let uniform = SceneUniform {
            max_iterations: 100,
            aspect_ratio: 1.0, // height/width
            scale: 1.0, 
            offset_x: 0.0,
            offset_y: 0.0,
            red_freq: 1.0,
            blue_freq: 1.0,
            green_freq: 1.0,
            red_phase: 0.0,
            blue_phase: 0.0,
            green_phase: 0.0,
        };

        let (vertex_buffer, uniform_buffer, bind_group, pipeline) = 
            build_pipeline(device, uniform, texture_format);

        Self { num_verticies, vertex_buffer, uniform, uniform_buffer, bind_group, pipeline }
    }

    pub fn draw(
        &self, 
        queue: &wgpu::Queue, 
        target: &wgpu::TextureView, 
        encoder: &mut wgpu::CommandEncoder
    ) {
        // Uniforms must be updated on every draw operation.
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[self.uniform]));

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {r: 0.0, g: 0.0, b: 0.0, a: 0.0}),
                    store: true,
                },
            })],
            depth_stencil_attachment: None,
        });


        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        render_pass.draw(0..self.num_verticies, 0..1);
    }

    pub fn set_max_iterations(&mut self, max_iterations: u32) {
        self.uniform.max_iterations = max_iterations;
    }

    pub fn set_aspect_ratio(&mut self, aspect_ratio: f32) {
        self.uniform.aspect_ratio = aspect_ratio;
    }

    pub fn scale(&self) -> f32 {self.uniform.scale}
    pub fn set_scale(&mut self, scale: f32) {
        self.uniform.scale = scale;
    }

    pub fn offset(&self) -> (f32, f32) {(self.uniform.offset_x, self.uniform.offset_y)}
    pub fn set_offset(&mut self, x_offset: f32, y_offset: f32) {
        self.uniform.offset_x = x_offset;
        self.uniform.offset_y = y_offset;
    }

    pub fn set_red_freq(&mut self, r_freq: f32) {
        self.uniform.red_freq = r_freq;
    }

    pub fn set_green_freq(&mut self, g_freq: f32) {
        self.uniform.green_freq = g_freq;
    }

    pub fn set_blue_freq(&mut self, b_freq: f32) {
        self.uniform.blue_freq = b_freq;
    }

    pub fn set_red_phase(&mut self, r_phase: f32) {
        self.uniform.red_phase = r_phase;
    }

    pub fn set_green_phase(&mut self, g_phase: f32) {
        self.uniform.green_phase = g_phase;
    }

    pub fn set_blue_phase(&mut self, b_phase: f32) {
        self.uniform.blue_phase = b_phase;
    }
}

fn build_pipeline(
    device: &wgpu::Device,
    uniform: SceneUniform,
    texture_format: wgpu::TextureFormat,
) -> (wgpu::Buffer, wgpu::Buffer, wgpu::BindGroup, wgpu::RenderPipeline) {
    // Compile the shader
    let shader = device.create_shader_module(wgpu::include_wgsl!("mandelbrot.wgsl"));

    // Use a Vertex buffer (and not a shader)
    let vertex_buffer = device.create_buffer_init(
        &wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        }
    );

    // Setup the Uniform into a buffer so it can be shared and seen by the shader
    let uniform_buff = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&[uniform]),
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

    let pipeline_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            push_constant_ranges: &[],
            bind_group_layouts: &[
                &bind_group_layout
            ],
        });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vertex_main",
            buffers: &[
                Vertex::desc(),
            ],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fragment_main",
            targets: &[Some(wgpu::ColorTargetState {
                format: texture_format,
                blend: Some(wgpu::BlendState {
                    color: wgpu::BlendComponent::REPLACE,
                    alpha: wgpu::BlendComponent::REPLACE,
                }),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleStrip,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        multiview: None,
    });

    (vertex_buffer, uniform_buff, bind_group, pipeline)
}