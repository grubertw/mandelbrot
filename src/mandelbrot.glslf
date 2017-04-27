// file: mandelbrot.glslf
//
// Fragment (pixel) shader.  This is where mandelbrout set
// inclusion is calculated (per pixel).  Pixels represent numbers
// on the complex plane (i.e. polynomials).  After max_iterations,
// the absolute value of the complex number (z, represented by a 
// vector with two doubles) can be taken. If greater than 2, then
// z is not in the set.
//
// For purposes of making an intresting drawing, it is common to 
// track the iterations it takes to escape (i.e. the orbit of z)
// and use this infromation to pick a color (from a spectum,
// or possbilly a 1d texture image).
//
// Lots of other intresting things can be done to see even more 
// properties of the set, such as it's bifurcation points (for
// points z inside the set). 
#version 450 core

uniform uint max_iterations;
uniform uint window_height;

// should start at 4, since mandelbrot goes from -2 to 1
uniform double scale; 

// should start at 0, and then cumulativly increase/decrease
// as the mouse is moved (with left button held down)
uniform double offset_x;
uniform double offset_y;

// frequency adjustment for color sine waves
uniform float red_freq;
uniform float blue_freq;
uniform float green_freq;

// phase shift adjustment for color sine waves
uniform float red_phase;
uniform float blue_phase;
uniform float green_phase;

out vec4 color;

vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main() {
    // Adjust the offset (focus) according to scale.
    dvec2 scaledOffset = dvec2(offset_x / scale, offset_y / scale);

    // Shift the fragment coordinets based on the offset.
    dvec2 shifted_coords = gl_FragCoord.xy + scaledOffset;

    // first make pixle coodinates between 0 and 1. Then scale to all 4 quadrents,
    // then do final shift.
    dvec2 c = shifted_coords / double(window_height) * (scale*4) - (scale*2);

    // Start interating at c rather than 0 because we never break out
    // at 0 anyways.
    dvec2 z = c;
    uint i;
    for (i = 0; i < max_iterations; i++) {
        z = dvec2((z.x * z.x) - (z.y * z.y), 2 * z.x * z.y) + c;
        if(length(z) > 2.0) {
            break;
        }
    }

    // If i got to max, we are in the set.
    if (i == max_iterations) {
        color = vec4(0.0, 0.0, 0.0, 1.0);
    } 
    // otherwise color the set with i (the orbit)
    else {
        // Calculate a continuous index for smoother coloring.
        //float continuous_i = i + 1 - (log(2) / float(length(z))) / log(2);

        // By default, make frequency proportional to the max iteractions.
        // Since the default frequcy of a sin wave is 2pi radians, make 1/2 pi
        // (which only makes use of the first part of the upward curve)
        float freq = float(i) / float(max_iterations) * 1.57;

        color = vec4(
            sin(freq * red_freq + red_phase),
            sin(freq * blue_freq + blue_phase),
            sin(freq * green_freq + green_phase),
            1.0
        );

        // float val = i / float(max_iterations);
        // color = vec4(hsv2rgb(vec3(val, 1.0, 1.0)), 1.0);
    }
}