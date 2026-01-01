
use rug::{Float, Complex};

pub const DEFAULT_BAILOUT: f64 = 4.0;
pub const MAX_SAFE_DF_MAG: f64 = 1e30;

// Double-float, which is our 'bypass' of WGSL's lack of f64
// On that note, using two floats in this way is far more robust 
// accross a wider set of GPUs. While not giving 53 bits of precision,
// this can theoreticly give us up to 48 bits - i.e. 24+24 as f32 
// has 24 bits - though in practice it will likely yeild only 40-44.
#[derive(Clone, Copy, Debug)]
pub struct Df {
    pub hi: f32,
    pub lo: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct ComplexDf {
    pub re: Df,
    pub im: Df,
}

impl Df {
    // Convert a Rug Float to double-float
    // First rounds the arbitary precision to fit into f64, and then seeks to 
    // preserve what's lost with the initial rounding, and perserves that in 
    // another f64. Ultilate, these high and low f64s are trunkated again 
    // before finally being returns as a Df. This strategy tries to preserve
    // 'the most meaningful' significant digets at the beginning and end of
    // the arbitraty precision value.
    pub fn from_float(x: &Float) -> Self {
        let hi = x.to_f64();

        // residual = x - hi
        let mut residual = Float::with_val(x.prec(), x);
        residual -= hi;

        let lo = residual.to_f64();
        
        Self {hi: hi as f32, lo: lo as f32}
    }
}

impl ComplexDf {
    pub fn from_complex(c: &Complex) -> Self {
        let real_df = Df::from_float(c.real());
        let imag_df = Df::from_float(c.imag());

        Self {re: real_df, im: imag_df}
    }
}
