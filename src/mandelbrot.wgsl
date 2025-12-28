struct DebugOut {
    zx_hi: f32,
    zx_lo: f32,
    zy_hi: f32,
    zy_lo: f32,
    cr_hi: f32,
    cr_lo: f32,
    ci_hi: f32,
    ci_lo: f32,
    ax: f32,
    ay: f32,
    pix_dx_hi: f32,
    pix_dx_lo: f32,
    pix_dy_hi: f32,
    pix_dy_lo: f32,
};

@group(1) @binding(0)
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
// Utility: renormalize a double-double (hi, lo) pair
// ensure hi contains the high part, lo contains residual
// -------------------------------
fn renorm(a_hi: f32, a_lo: f32) -> Df {
    let s = a_hi + a_lo;
    var out: Df;
    out.hi = s;
    out.lo = a_lo - (s - a_hi);
    return out;
}

// -------------------------------
// Double-float addition: a + b
// -------------------------------
// Dekker/Knuth
//fn df_add(a: Df, b: Df) -> Df {
//    let s = a.hi + b.hi;
//    let v = s - a.hi;
//    let err = (a.hi - (s - v)) + (b.hi - v) + a.lo + b.lo;
//    return renorm(s, err);
//}
fn df_add(a: Df, b: Df) -> Df {
    let s = a.hi + b.hi;
    let e = (a.hi - s) + b.hi + a.lo + b.lo;
    return Df(s, e);
}

// -------------------------------
// Double-float subtraction: a - b
// -------------------------------
// Dekker/Knuth
//fn df_sub(a: Df, b: Df) -> Df {
//    let s = a.hi - b.hi;
//    let v = s - a.hi;
//    let err = (a.hi - (s - v)) - (b.hi + v) + a.lo - b.lo;
//    return renorm(s, err);
//}
fn df_sub(a: Df, b: Df) -> Df {
    let s = a.hi - b.hi;
    let e = (a.hi - s) - b.hi + a.lo - b.lo;
    return Df(s, e);
}

// Split a float into high and low parts for Dekker multiplication.
// For float32 the recommended splitter is 8193.0 (2^13 + 1) to split 24-bit mantissa.
fn split_f32(a: f32) -> vec2<f32> {
    let splitter: f32 = 8193.0;
    let c = splitter * a;
    let abig = c - (c - a);
    let ahi = abig;
    let alo = a - ahi;
    return vec2<f32>(ahi, alo);
}

// -------------------------------
// Double-float multiply: a * b
// Improved df_mul using Dekker splitting for the hi*hi product error compensation.
// -------------------------------
// Dekker/Knuth
//fn df_mul(a: Df, b: Df) -> Df {
//    let p = a.hi * b.hi;
//
//    // Split hi parts
//    let sa = split_f32(a.hi);
//    let sb = split_f32(b.hi);
//    let ahi = sa.x; let alo = sa.y;
//    let bhi = sb.x; let blo = sb.y;
//
//    // Dekker-style error terms
//    let err1 = ((ahi * bhi) - p) + (ahi * blo) + (alo * bhi);
//    let err2 = alo * blo;
//    let cross = (a.hi * b.lo) + (a.lo * b.hi);
//    let err = err1 + err2 + cross + (a.lo * b.lo);
//
//    return renorm(p, err);
//}
fn df_mul(a: Df, b: Df) -> Df {
    let p = a.hi * b.hi;
    let e = a.hi * b.lo + a.lo * b.hi;
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
    var out: Df;
    out.hi = x;
    out.lo = 0.0;
    return out;
}

fn df_from_i32(i: i32) -> Df {
    var out: Df;
    out.hi = f32(i);
    out.lo = 0.0;
    return out;
}

fn df_mul_f32(a: Df, b: f32) -> Df {
    // multiply Df by scalar f32
    var bdf: Df;
    bdf.hi = b;
    bdf.lo = 0.0;
    return df_mul(a, bdf);
}

//fn df_to_f32(a: Df) -> f32 {
//    return a.hi + a.lo;
//}

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
    red_freq:       f32,
    blue_freq:      f32,
    green_freq:     f32,
    red_phase:      f32,
    blue_phase:     f32,
    green_phase:    f32,
};

@group(0) @binding(0) var<uniform> uni: Uniforms;


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
fn mandelbrot_df(c: ComplexDf, coords: vec4<f32>) -> u32 {
    var zx: Df = df_from_f32(0.0);
    var zy: Df = df_from_f32(0.0);

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

    if (i32(coords.x) == 0 && i32(coords.y) == 0) {
        // write out z and c
        debug_out.zx_hi = zx.hi;
        debug_out.zx_lo = zx.lo;
        debug_out.zy_hi = zy.hi;
        debug_out.zy_lo = zy.lo;
        debug_out.cr_hi = c.r.hi;
        debug_out.cr_lo = c.r.lo;
        debug_out.ci_hi = c.i.hi;
        debug_out.ci_lo = c.i.lo;

        debug_out.ax = abs(zx.hi) + abs(zx.lo);
        debug_out.ay = abs(zy.hi) + abs(zy.lo);

        debug_out.pix_dx_hi = uni.pix_dx_hi;
        debug_out.pix_dx_lo = uni.pix_dx_lo;
        debug_out.pix_dy_hi = uni.pix_dy_hi;
        debug_out.pix_dy_lo = uni.pix_dy_lo;
    }

    return i;
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

    let it = mandelbrot_df(c, coords);

    if (it == uni.max_iter) {
        // inside set -> black
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }

    // simple coloring: normalized iteration
    let t = f32(it) / f32(uni.max_iter);
    // you can replace with your frequency / phase coloring using the rgb_* uniforms
    let color = vec3<f32>(t, t*t, pow(t, 0.5));
    return vec4<f32>(color, 1.0);
}
