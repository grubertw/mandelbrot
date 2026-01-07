use super::numerics::ComplexDf;
use super::signals;

use iced_winit::winit::window::Window;

use futures::channel;
use futures::task::SpawnExt;
use futures::StreamExt;
use futures::select;
use futures::future::{join_all, RemoteHandle};
use futures::executor::{ThreadPool};

use rug::{Float, Complex};
use log::{debug, info};

use std::time;
use std::sync::{Arc, Mutex};

// Maximum size the orbit pool can be before we begin to trim the lowest ranked ones
const MAX_ORBIT_POOL_SIZE: usize = 25;

// Common types used throughout the module - mainly between async functions and accross the
// sync/async barrier
type LivingOrbits           = Arc<Mutex<Vec<ReferenceOrbit>>>;
type CameraSnapshotSender   = channel::mpsc::Sender<signals::CameraSnapshot>;
type CameraSnapshotReceiver = channel::mpsc::Receiver<signals::CameraSnapshot>;
type GpuFeedbackSender      = channel::mpsc::Sender<signals::GpuFeedback>;
type GpuFeedbackReceiver    = channel::mpsc::Receiver<signals::GpuFeedback>;

#[derive(Clone, Copy, Debug)]
pub struct HeuristicConfig {
    pub weight_1: f64, // Just a placeholder
}

#[derive(Clone, Copy, Debug)]
pub struct ScoutConfig {
    pub max_orbits: u32,
    pub max_iterations_ref: u32,
    pub rug_precision: u32,
    pub heuristic_config: HeuristicConfig,
    pub exploration_budget: f64,
}

#[derive(Debug)]
pub struct ScoutEngine {
    // Our winit window 
    window: Arc<Window>,
    // Our startup configuration
    config: Arc<ScoutConfig>,
    // Internal thread-pool, used both for the primary event loop, and internal async tasks
    thread_pool: ThreadPool,
    // Our working pool of reference orbits
    living_orbits: LivingOrbits,
    // Async channels
    cameara_snapshot_tx: CameraSnapshotSender,
    gpu_feedback_tx: GpuFeedbackSender,
}

#[derive(Clone, Debug)]
struct HeuristicWeights {
    w_dist:     f64, // Distance from camera center
    w_depth:    f64, // Escape Index
    w_age:      f64, // num framce since last use
}

impl HeuristicWeights {
    fn vectorize(&self) -> Vec<f64> {
        vec![self.w_dist, self.w_depth, self.w_age]
    }
}

#[derive(Clone, Debug)]
struct ReferenceOrbit {
    c_ref: Complex,
    c_ref_df: ComplexDf,
    orbit: Vec<Complex>,
    orbit_df: Vec<ComplexDf>,
    escape_index: Option<u32>,
    max_lambda: Float,
    weights: HeuristicWeights,
    current_score: i64,
    creation_time: time::Instant,
    creation_frame_id: u64,
}

impl PartialEq for ReferenceOrbit {
    fn eq(&self, other: &Self) -> bool {
        self.creation_time == other.creation_time
    }
}

impl Eq for ReferenceOrbit {}

#[derive(Debug)]
struct OrbitResult {
    orbit: Vec<Complex>,
    escape_index: Option<u32>,
}

impl ScoutEngine {
    pub fn new(window: Arc<Window>, config: ScoutConfig) -> Self {
        let config = Arc::new(config);
        let thread_pool = ThreadPool::new().expect("Failed to build ThreadPool for ScoutEngine");
        let living_orbits = Arc::new(Mutex::new(Vec::new()));

        // Create asynch channels for communicating between the render loop/thread and 
        // the long-lived ScoutEngine async task
        let (cameara_snapshot_tx, cameara_snapshot_rx) = channel::mpsc::channel::<signals::CameraSnapshot>(10);
        let (gpu_feedback_tx, gpu_feedback_rx) = channel::mpsc::channel::<signals::GpuFeedback>(10);

        // Launch ScoutEngine's long-lived task, which 'blocks' on the above async channels
        thread_pool.spawn_ok(Self::scout_worker(window.clone(), config.clone(), thread_pool.clone(),
            living_orbits.clone(), cameara_snapshot_rx, gpu_feedback_rx));

        Self {
            window: window, config, thread_pool, living_orbits,
            cameara_snapshot_tx, gpu_feedback_tx,
        } 
    }

    pub fn submit_camera_snapshot(&mut self, snapshot: signals::CameraSnapshot) {
        debug!("Camera Snapshot received: {:?}", snapshot);
        self.cameara_snapshot_tx.try_send(snapshot).ok();
    }

    pub fn submit_gpu_feedback(&mut self, feedback: signals::GpuFeedback) {
        debug!("Gpu Feedback received: {:?}", feedback);
        self.gpu_feedback_tx.try_send(feedback).ok();
    }

    // Ranks the orbits in the living_orbits according to an internal ranking algorithm
    // which is driven by attributes with each ReferenceOrbit.
    pub fn query_best_orbits(&self, num_orbits: u32) -> Vec<signals::ReferenceOrbitDf> {
        let lo_cloned = self.living_orbits.clone();
        let mut best_orbits = lo_cloned.lock().unwrap();
        
        let (best_orbits_split, _) = best_orbits.split_at(num_orbits as usize); 

        best_orbits_split.iter().map(|orb| {
            signals::ReferenceOrbitDf {
                c_ref: orb.c_ref_df.clone(),
                orbit: orb.orbit_df.clone(),
                escape_index: orb.escape_index,
                creation_time: orb.creation_time
            }
        }).collect()
    }

    pub fn diagnostics(&self) -> signals::ScoutDiagnostics {
        signals::ScoutDiagnostics {
            timestamp: time::SystemTime::now(),
            message: String::new()
        }
    }

    // ScoutEngine's internal work loop, a long-lived future that uses select to poll the 
    // the camera snaphot & gpu feedback channels.
    async fn scout_worker(
            window: Arc<Window>, config: Arc<ScoutConfig>, tp: ThreadPool, living_orbits: LivingOrbits, 
            cameara_snapshot_rx: CameraSnapshotReceiver, gpu_feedback_rx: GpuFeedbackReceiver) {
        
        let mut snap_rx = cameara_snapshot_rx.fuse();
        let mut gpu_rx = gpu_feedback_rx.fuse();

        info!("Scout Worker started!");

        loop {
            select! {
                cs_res = snap_rx.next() => {
                    match cs_res {
                        Some(snapshot) => {
                            info!("Scout Worker received camera snapshot {:?}", snapshot);
                            // Create a new set of reference orbits asynchronously.
                            // This is a computationally heavy operation, so it is best to 
                            // invoke on the threadpool without waiting.
                            tp.spawn_ok(Self::create_new_reference_orbits_from_camera_snapshot(
                                window.clone(), config.clone(), tp.clone(), 
                                living_orbits.clone(), Arc::new(snapshot)));
                        }
                        None => {
                            break;
                        }
                    }
                },
                gpu_res = gpu_rx.next() => {
                    match gpu_res {
                        Some(feedback) => {
                            debug!("Scout Worker received GpuFeedback={:?}", feedback);
                        }
                        None => {
                            break;
                        }
                    }
                    
                },
            };
        }
    }

    async fn create_new_reference_orbits_from_camera_snapshot(window: Arc<Window>, config: Arc<ScoutConfig>, 
            tp: ThreadPool, living_orbits: LivingOrbits, snapshot: Arc<signals::CameraSnapshot>) {
        let seeds = Self::create_new_orbit_seeds_from_camera_snapshot(snapshot.clone()).await;
        debug!("Created orbit seeds from snapshot: {:?}", seeds);

        // Schedule thread-pool execution for the creation of reference orbits from seeds.
        let handles = seeds.iter().fold(Vec::<RemoteHandle<ReferenceOrbit>>::new(), |mut acc, seed| {
            let res = tp.spawn_with_handle(
                Self::create_new_reference_orbit(seed.clone(), config.max_orbits, config.rug_precision,
                    snapshot.frame_stamp.frame_id));
            if let Ok(h) = res {
                acc.push(h);
            };
            acc
        });

        // Wait for all the orbit computations to complete on the thread-pool
        let results: Vec<ReferenceOrbit> = join_all(handles).await;
        let r_len = &results.len();

        // Insert new orbits into living orbits
        for orb in results {
            let mut lo_mut = living_orbits.lock().unwrap();
            lo_mut.push(orb);
        }
        debug!("New reference orbit results collected! results.len={} living_orbits.len()={}", 
            r_len, living_orbits.lock().unwrap().len());

        // Recalculate weights for all entires in the pool after addition
        // And with the snapshot provided on creation of these new orbits
        Self::weigh_living_orbits(living_orbits.clone(), Some(snapshot), None).await;

        // Score the orbits
        Self::score_living_orbits(living_orbits.clone()).await;

        // Rank/order the orbits
        Self::rank_living_orbits(living_orbits.clone()).await;

        // Trim the pool
        Self::trim_living_orbits(living_orbits.clone()).await;

        // Send a signal to the winit window to wake up the render loop and redraw the viewport
        // NOTE: the render loop may chose not to honor any information made available by 
        // ScoutEngine, and that is expected. This simpally exists to wake up the UI if more 
        // drawing can be done via a freshly ranked set of reference orbits.
        window.request_redraw();
    }

    // Fast operation, can be awaited.
    async fn create_new_orbit_seeds_from_camera_snapshot(snapshot: Arc<signals::CameraSnapshot>) -> Vec<Complex> {
        let mut seeds = Vec::<Complex>::new();
        let center = snapshot.center.clone();
        let scale = snapshot.scale.clone();

        // Stay very simple now, with a 2x2 pattern now, i.e. top-bottom-left-right-middle of the viewport
        seeds.push(center.clone());

        let mut top = center.clone();
        *top.mut_imag() += &scale;
        seeds.push(top);

        let mut bottom = center.clone();
        *bottom.mut_imag() -= &scale;
        seeds.push(bottom);

        let mut left = center.clone();
        *left.mut_real() += &scale;
        seeds.push(left);

        let mut right = center.clone();
        *right.mut_real() -= &scale;
        seeds.push(right);

        seeds
    }

    async fn create_new_reference_orbit(c_ref: Complex, max_iter: u32, prec: u32, snap_frame: u64) 
            -> ReferenceOrbit {
        let orbit_result = Self::compute_reference_orbit(&c_ref, max_iter, prec).await;
        let c_ref_df = ComplexDf::from_complex(&c_ref);
        let orbit_df = orbit_result.orbit.iter().map(|c| ComplexDf::from_complex(c)).collect();
        let creation_time = time::Instant::now();
        
        info!("Creating new ReferenceOrbit at time={:?} and snapshot frame_id={}", creation_time, snap_frame);
        ReferenceOrbit{c_ref: c_ref.clone(), c_ref_df,
            orbit: orbit_result.orbit, orbit_df,
            escape_index: orbit_result.escape_index,
            max_lambda: Float::with_val(prec, 0.0),
            weights: HeuristicWeights{ w_dist: 0.0, w_depth: 0.0, w_age: 0.0 },
            current_score: 0,
            creation_time,
            creation_frame_id: snap_frame
        }
    }

    async fn compute_reference_orbit(c_ref: &Complex, max_iter: u32, prec: u32) 
            -> OrbitResult {
        let mut orbit = Vec::<Complex>::with_capacity(max_iter as usize);
        let mut escape_index: Option<u32> = None;

        let mut z = Complex::with_val(prec, (0.0, 0.0));

        for i in 0..max_iter {
            orbit.push(z.clone());
            z = z.clone() * &z + c_ref;

            if z.clone().abs().real().to_f64() >= 2.0 && escape_index == None {
                escape_index = Some(i);
            }
        }

        debug!("New Reference Orbit computed for c_ref={} orbit.len={} escape_index={:?}", 
            &c_ref, orbit.len(), escape_index);

        OrbitResult {orbit, escape_index}
    }

    // Calculate weights that contribute to the score
    // Weight 'fairness' works on a decreasing scale
    // Score scale is linear, so logs/exponents are used to linerize the number
    // Negative weights are good, and large positives are bad
    // Intrinsic vector ording likes to go from smallest to largest, which needs 
    // to be kept in mind as the score will be used to rank/sort
    async fn weigh_living_orbits(
            living_orbits: LivingOrbits, 
            snapshot: Option<Arc<signals::CameraSnapshot>>,
            _feedback: Option<Arc<signals::GpuFeedback>>) {
        let mut orb_pool_l = living_orbits.lock().unwrap();

        for orb in orb_pool_l.iter_mut() {
            if let Some(ref cam_snap) = snapshot {
                let delta = orb.c_ref.clone() - &cam_snap.center;
                orb.weights.w_dist = delta.abs().real().to_f64() / &cam_snap.scale.to_f64();

                orb.weights.w_depth = if let Some(i) = orb.escape_index {i.into()} else {orb.orbit.len() as f64};
                orb.weights.w_depth *= -1.0;

                // Frame id's should always be increasing, otherwize this breaks
                orb.weights.w_age = (cam_snap.frame_stamp.frame_id as f32 - orb.creation_frame_id as f32) as f64;
            }
        };
    }

    // Simply adds all the weights in HeuristicWeights by first vectorizing the strcture,
    // which is possible because all values are the same type, and taking a sum of all 
    // elements in the vector, using an iterator.
    async fn score_living_orbits(living_orbits: LivingOrbits) {
        let mut orb_pool_l = living_orbits.lock().unwrap();
        for orb in orb_pool_l.iter_mut() {
            orb.current_score = orb.weights.vectorize().iter().sum::<f64>() as i64;
            debug!("Orbit has new score {}\tFrom weights: w_dist={} w_depth={} w_age={}\tAt c_ref_df={:?}", 
                orb.current_score, orb.weights.w_dist, orb.weights.w_depth, orb.weights.w_age, orb.c_ref_df);
        };
    }

    async fn rank_living_orbits(living_orbits: LivingOrbits) {
        // convert the orbit pool into a vector for sorting
        let mut lom = living_orbits.lock().unwrap();

        // Sort the vector if orbit's by it's score.
        lom.sort_by_key(|orb| orb.current_score);
    }

    async fn trim_living_orbits(living_orbits: LivingOrbits) {
        let mut lom = living_orbits.lock().unwrap(); 

        lom.truncate(MAX_ORBIT_POOL_SIZE);
    }
}
