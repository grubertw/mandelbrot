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
    last_valid_z_re_hi: f32,
    last_valid_z_re_lo: f32,
    last_valid_z_im_hi: f32,
    last_valid_z_im_lo: f32,
};

@group(2) @binding(0)
var<storage, read_write> debug_out: DebugOut;

// -------------------------------
// Double-float structure
// -------------------------------
struct Df {
    hi: f32,
    lo: f32,
};

struct ComplexDf {
    r: Df,
    i: Df,
};

// -------------------------------
// Double-float arithmatic operations
// -------------------------------
fn df_add(a: Df, b: Df) -> Df {
    let s = a.hi + b.hi;
    let e = (a.hi - s) + b.hi + a.lo + b.lo;
    return Df(s, e);
}

fn df_sub(a: Df, b: Df) -> Df {
    let s = a.hi - b.hi;
    let e = (a.hi - s) - b.hi + a.lo - b.lo;
    return Df(s, e);
}

fn df_mul(a: Df, b: Df) -> Df {
    let p = a.hi * b.hi;
    let e = a.hi * b.lo + a.lo * b.hi;
    return Df(p, e);
}

fn df_div(a: Df, b: Df) -> Df {
    let p = a.hi / b.hi;
    let e = a.hi / b.lo + a.lo / b.hi;
    return Df(p, e);
}

fn df_neg(a: Df) -> Df {
    var out: Df;
    out.hi = -a.hi;
    out.lo = -a.lo;
    return out;
}

// -------------------------------
// Convert a regular f32 into a Df (lo = 0.0)
// -------------------------------
fn df_from_f32(x: f32) -> Df {
    return Df(x, 0.0);
}

fn df_from_i32(i: i32) -> Df {
    return Df(f32(i), 0.0);
}

// multiply Df by scalar f32
fn df_mul_f32(a: Df, b: f32) -> Df {
    return df_mul(a, Df(b, 0.0));
}


fn df_mag2(v: Df) -> f32 {
    let v_abs = abs(v.hi) + abs(v.lo);
    return v_abs * v_abs;
}

fn df_mag2_upper(zx: Df, zy: Df) -> f32 {
    let ax = abs(zx.hi) + abs(zx.lo);
    let ay = abs(zy.hi) + abs(zy.lo);
    return ax * ax + ay * ay;
}

// -------------------------------
// Complex double-float operations
// z = x + i*y
// -------------------------------
fn cdf_add(a: ComplexDf, b: ComplexDf) -> ComplexDf {
    var r: ComplexDf;
    r.r = df_add(a.r, b.r);
    r.i = df_add(a.i, b.i);
    return r;
}

fn cdf_sub(a: ComplexDf, b: ComplexDf) -> ComplexDf {
    var r: ComplexDf;
    r.r = df_sub(a.r, b.r);
    r.i = df_sub(a.i, b.i);
    return r;
}

// complex multiply: (ar + i ai) * (br + i bi) = (ar*br - ai*bi) + i(ar*bi + ai*br)
fn cdf_mul(a: ComplexDf, b: ComplexDf) -> ComplexDf {
    let rr = df_mul(a.r, b.r);
    let ii = df_mul(a.i, b.i);
    var real = df_sub(rr, ii);
    let rbi = df_mul(a.r, b.i);
    let ibr = df_mul(a.i, b.r);
    var imag = df_add(rbi, ibr);
    var out: ComplexDf;
    out.r = real;
    out.i = imag;
    return out;
}

// -------------------------------
// Uniforms
// -------------------------------
struct Uniforms {
    center_x_hi:    f32,
    center_x_lo:    f32,
    center_y_hi:    f32,
    center_y_lo:    f32,
    scale_hi:       f32,
    scale_lo:       f32,
    pix_dx_hi:      f32,
    pix_dx_lo:      f32,
    pix_dy_hi:      f32,
    pix_dy_lo:      f32,
    width:          f32,
    height:         f32,
    max_iter:       u32,
    ref_len:        u32,
};
@group(0) @binding(0) var<uniform> uni: Uniforms;

struct GpuFeedback {
    max_lambda_re_hi:  f32,
    max_lambda_re_lo:  f32,
    max_lambda_im_hi:  f32,
    max_lambda_im_lo:  f32,
    max_delta_z_re_hi: f32,
    max_delta_z_re_lo: f32,
    max_delta_z_im_hi: f32,
    max_delta_z_im_lo: f32,
    escape_ratio:   f32,
};
@group(1) @binding(0) var<storage, read_write> gpu_fb: GpuFeedback;

// ----------------------------
// Reference orbit from ScoutEngine
// ----------------------------
@group(3) @binding(0)
var<storage, read> ref_orbit_re_hi: array<f32>;
@group(3) @binding(1)
var<storage, read> ref_orbit_re_lo: array<f32>;
@group(3) @binding(2)
var<storage, read> ref_orbit_im_hi: array<f32>;
@group(3) @binding(3)
var<storage, read> ref_orbit_im_lo: array<f32>;

// ---------- Build c from integer pixel offsets using CPU-provided pix_dx/pix_dy ----------
fn build_c_from_frag(coords: vec4<f32>) -> ComplexDf {
    // integer pixel coordinates (truncate)
    let ix: i32 = i32(coords.x);
    let iy: i32 = i32(coords.y);

    // integer center (half window). Compute half in f32 then cast to i32 is OK here because width is pixel count.
    let half_w: i32 = i32(uni.width * 0.5);
    let half_h: i32 = i32(uni.height * 0.5);

    let dx_i: i32 = ix - half_w;
    let dy_i: i32 = iy - half_h;

    let dx_df = df_from_i32(dx_i);
    let dy_df = df_from_i32(dy_i);

    // load pix vectors supplied by CPU (each is a Df)
    let pix_dx = Df(uni.pix_dx_hi, uni.pix_dx_lo);
    let pix_dy = Df(uni.pix_dy_hi, uni.pix_dy_lo);

    // offset = dx*pix_dx + dy*pix_dy
    let off_x = df_mul(dx_df, pix_dx);
    let off_y = df_mul(dy_df, pix_dy);

    let center_x = Df(uni.center_x_hi, uni.center_x_lo);
    let center_y = Df(uni.center_y_hi, uni.center_y_lo);

    let c = ComplexDf(df_add(center_x, off_x), df_add(center_y, off_y)); 
    return c;
}


// -------------------------------
// Mandelbrot iteration using DF arithmetic.
// Returns iteration count (u32).
// -------------------------------
fn mandelbrot_df_from_z(z: ComplexDf, c: ComplexDf) -> u32 {
    var zx: Df = z.r;
    var zy: Df = z.i;
    var i: u32 = 0u;
    let max_i: u32 = uni.max_iter;

    loop {
        // compute squares and cross product (all double-float)
        let zx2 = df_mul(zx, zx);      // zx*zx
        let zy2 = df_mul(zy, zy);      // zy*zy
        let zxy = df_mul(zx, zy);      // zx*zy

        // real = zx2 - zy2 + c.r
        let real_part = df_add(df_sub(zx2, zy2), c.r);

        // imag = 2*zx*zy + c.i  -> 2*zxy + c.i
        let imag_part = df_add(df_add(zxy, zxy), c.i);

        // update z
        zx = real_part;
        zy = imag_part;

        // Bailout
        let mag2 = df_mag2_upper(zx, zy);
        if (mag2 > 16.0) {
            break;
        }

        i = i + 1u;
        if (i >= max_i) { break; }
    }

    return i;
}

const validity_radius2 = 0.01; // or 0.001 to be stricter
const PERTURB_SCALE_THRESHOLD = 1e-5;

fn mandelbrot_perturb(c: ComplexDf, delta_c: ComplexDf, coords: vec4<f32>) -> vec2<u32> {
    var dz = ComplexDf(df_from_f32(0.0), df_from_f32(0.0));
    var i: u32 = 0u;
    let max_i = uni.max_iter;
    var esc_seq: u32 = 0u;

    let scale = Df(uni.scale_hi, uni.scale_lo);

    // Track last iteration where perturbation was valid
    var last_valid_i: u32 = 0u;
    var last_valid_z = ComplexDf(df_from_f32(0.0), df_from_f32(0.0));

    loop {
        // Load reference orbit Z_n
        let Z = ComplexDf(
            Df(ref_orbit_re_hi[i], ref_orbit_re_lo[i]),
            Df(ref_orbit_im_hi[i], ref_orbit_im_lo[i])
        );

        // λ_n = 2 * Z_n
        let lambda = cdf_add(Z, Z);

        // dz_{n+1} = λ_n * dz_n + Δc
        dz = cdf_add(cdf_mul(lambda, dz), delta_c);

        // Absolute z for escape testing
        let z = cdf_add(Z, dz);

        // Standard bailout
        if (df_mag2_upper(z.r, z.i) > 16.0) {
            esc_seq = 1u;
            break;
        }

        // Perturbation validity collapse
        if (df_mag2_upper(dz.r, dz.i) > validity_radius2) {
            last_valid_i = i;
            last_valid_z = z;
            esc_seq = 2u;
            break;
        }

        i = i + 1u;
        if (i >= max_i) { 
            esc_seq = 3u;
            break; 
        }
    }

    if (esc_seq == 2) {
        // Continue ABSOLUTE from last valid
        i = i + mandelbrot_df_from_z(last_valid_z, c);
    }

    debug_out.perturb_escape_seq = esc_seq;
    debug_out.last_valid_i = last_valid_i;
    debug_out.last_valid_z_re_hi = last_valid_z.r.hi;
    debug_out.last_valid_z_re_lo = last_valid_z.r.lo;
    debug_out.last_valid_z_im_hi = last_valid_z.i.hi;
    debug_out.last_valid_z_im_lo = last_valid_z.i.lo;

    return vec2<u32>(i, last_valid_i);
}


// -------------------------------
// Fullscreen triangle VS
// -------------------------------
@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    var pos: vec2<f32>;
    switch (vid) {
        case 0u: { pos = vec2<f32>(-1.0, -1.0); }
        case 1u: { pos = vec2<f32>( 3.0, -1.0); }
        case 2u: { pos = vec2<f32>(-1.0,  3.0); }
        default: { pos = vec2<f32>(0.0, 0.0); }
    }
    return vec4<f32>(pos, 0.0, 1.0);
}

// -------------------------------
// Fragment shader
// -------------------------------
@fragment
fn fs_main(@builtin(position) coords: vec4<f32>) -> @location(0) vec4<f32> {
    // Build complex point c with DF precision using pixel offsets.
    let c = build_c_from_frag(coords);
    var it: u32 = 0u;

    let c_ref = ComplexDf(
        Df(uni.center_x_hi, uni.center_x_lo),
        Df(uni.center_y_hi, uni.center_y_lo)
    );
    let delta_c = cdf_sub(c, c_ref);

    debug_out.c_ref_re_hi = c_ref.r.hi;
    debug_out.c_ref_re_lo = c_ref.r.lo;
    debug_out.c_ref_im_hi = c_ref.i.hi;
    debug_out.c_ref_im_lo = c_ref.i.lo;
    debug_out.delta_c_re_hi = delta_c.r.hi;
    debug_out.delta_c_re_lo = delta_c.r.lo;
    debug_out.delta_c_im_hi = delta_c.i.hi;
    debug_out.delta_c_im_lo = delta_c.i.lo;

    var t: f32 = 0.0;
    var color = vec3<f32>(0.0, 0.0, 0.0);

    if (uni.scale_hi < PERTURB_SCALE_THRESHOLD &&
        df_mag2_upper(delta_c.r, delta_c.i) < validity_radius2) {
        let p_res = mandelbrot_perturb(c, delta_c, coords);
        it = p_res.x;

        t = f32(p_res.y) / f32(uni.max_iter);
        color = vec3<f32>(t, 0.0, 1.0 - t);
    }
    else {
        var z = ComplexDf(df_from_f32(0.0), df_from_f32(0.0));
        it = mandelbrot_df_from_z(z, c);
        
        t = f32(it) / f32(uni.max_iter);
        color = vec3<f32>(t, t*t, pow(t, 0.5));
    }

    if (it == uni.max_iter) {
        // inside set -> black
        color = vec3<f32>(0.0, 0.0, 0.0);
    }

    return vec4<f32>(color, 1.0);
}
