use super::numerics::{Df, ComplexDf};

use std::time;

use rug::{Float, Complex};

///////////////////////////////////////////////////////////
// Consumed by Scout Engine
///////////////////////////////////////////////////////////
#[derive(Clone, Copy, Debug)]
pub struct FrameStamp {
    pub frame_id: u64,
    pub timestamp: time::Instant, // in nanoseconds
} 

#[derive(Clone, Debug)]
pub struct CameraSnapshot {
    pub frame_stamp: FrameStamp,
    pub center: Complex,
    pub scale: Float,
}

#[derive(Clone, Copy, Debug)]
pub struct GpuFeedback {
    pub frame_stamp: FrameStamp,
    pub max_lambda: Df,
    pub max_delta_z: Df,
    pub escape_ratio: f32,
}

#[derive(Clone, Debug)]
pub struct FrameDiagnostics {
    pub frame_stamp: FrameStamp,
    pub message: String,
}

///////////////////////////////////////////////////////////
// Produced by Scout Engine
///////////////////////////////////////////////////////////
#[derive(Clone, Debug)]
pub struct ReferenceSelection {
    pub c_ref: ComplexDf,
    pub orbit: Vec<ComplexDf>,
    pub max_iterations: u32,
}

#[derive(Clone, Debug)]
pub struct ScoutDiagnostics {
    pub timestamp: time::SystemTime,
    pub message: String,
}
