use super::numerics::{Df, ComplexDf};

use std::hash::{Hash, Hasher};
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
pub struct ReferenceOrbitDf {
    pub c_ref: ComplexDf,
    pub orbit: Vec<ComplexDf>,
    pub escape_index: Option<u32>,
    pub creation_time: time::Instant,
}

impl Hash for ReferenceOrbitDf {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.creation_time.hash(state);
    }
}

impl PartialEq for ReferenceOrbitDf {
    fn eq(&self, other: &Self) -> bool {
        self.creation_time == other.creation_time
    }
}

impl Eq for ReferenceOrbitDf {}

#[derive(Clone, Debug)]
pub struct ScoutDiagnostics {
    pub timestamp: time::SystemTime,
    pub message: String,
}
