
struct VertexInput {
	@location(0) position: vec2<f32>,
}

struct VertexOutput {
	@builtin(position) clip_position: vec4<f32>,
	
	@location(0) fragment_position: vec2<f32>,
};

struct Uniforms {
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
};

@group(0) @binding(0) 
var<uniform> uniforms: Uniforms;

// Verticies must be mapped to 4d space, but our fragment shader only cares about 2d,
// so we carry over position into fragment_position.
@vertex
fn vertex_main(model: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(model.position, 0.0, 1.0);
	out.fragment_position = model.position;
    return out;
}

@fragment
fn fragment_main(input: VertexOutput) -> @location(0) vec4<f32> {
	// Adjust the offset (focus) according to scale.
    var scaledOffset = vec2<f32>(
		uniforms.offset_x / uniforms.scale, 
		uniforms.offset_y / uniforms.scale
	);

    // Shift the fragment coordinets based on the offset.
    var shifted_coords = input.fragment_position + scaledOffset;

    // At the starting scale of 4, the fractal spans from -3 to 1 (x) and 2 to -2 (y),
    // But the fragment co-ordinates are -1 to 1. Some math is required to shift and honor
    // the aspect ratio of the window.
	var c = vec2<f32>(
		shifted_coords.x * uniforms.aspect_ratio * uniforms.scale,
		shifted_coords.y * uniforms.scale
	);

    // Start interating at c rather than 0 because we never break out
    // at 0 anyways.
    var z = c;
    var i: u32 = 0u;
    for (; i < uniforms.max_iterations; i++) {
        z = vec2<f32>((z.x * z.x) - (z.y * z.y), 2.0 * z.x * z.y) + c;
        if(length(z) > 2.0) {
            break;
        }
    }

	var color: vec4<f32>;

    // If i got to max, we are in the set.
    if (i == uniforms.max_iterations) {
        color = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    } 
    // otherwise color the set with i (the orbit)
    else {
        // By default, make frequency proportional to the max iteractions.
        // Since the default frequcy of a sin wave is 2pi radians, make 1/2 pi
        // (which only makes use of the first part of the upward curve)
        let freq = f32(i) / f32(uniforms.max_iterations) * 1.57;

        color = vec4<f32>(
            sin(freq * uniforms.red_freq + uniforms.red_phase),
            sin(freq * uniforms.blue_freq + uniforms.blue_phase),
            sin(freq * uniforms.green_freq + uniforms.green_phase),
            1.0
        );
    }
	
	return color;
}