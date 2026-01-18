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

use std::collections::HashMap;
use std::time;
use std::sync::{Arc, Mutex, Weak};

// Maximum size the orbit pool can be before we begin to trim the lowest ranked ones
const MAX_ORBIT_POOL_SIZE: usize = 25;

// Common types used throughout the module - mainly between async functions and accross the
// sync/async barrier
type LiveOrbit              = Arc<Mutex<ReferenceOrbit>>;
type WeakOrbit              = Weak<Mutex<ReferenceOrbit>>;
type LivingOrbits           = Arc<Mutex<Vec<LiveOrbit>>>;
type TileRegistry           = Arc<Mutex<HashMap<(u32, TileId), TileOrbitView>>>;
type CameraSnapshotSender   = channel::mpsc::Sender<signals::CameraSnapshot>;
type CameraSnapshotReceiver = channel::mpsc::Receiver<signals::CameraSnapshot>;
type GpuFeedbackSender      = channel::mpsc::Sender<signals::GpuFeedback>;
type GpuFeedbackReceiver    = channel::mpsc::Receiver<signals::GpuFeedback>;
type OrbitId                = u64;
type OrbitIdFact            = Arc<Mutex<OrbitIdFactory>>;

#[derive(Clone, Debug)]
pub struct HeuristicConfig {
    pub weight_1: f64, // Just a placeholder
}

#[derive(Clone, Debug)]
pub struct ScoutConfig {
    pub max_orbits: u32,
    pub max_iterations_ref: u32,
    pub rug_precision: u32,
    pub heuristic_config: HeuristicConfig,
    pub tile_levels: Vec<TileLevel>,
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
    // Where our (logical) tiles live
    tile_registry: TileRegistry,
    // Async channels
    cameara_snapshot_tx: CameraSnapshotSender,
    gpu_feedback_tx: GpuFeedbackSender,
}

struct OrbitIdFactory {
    next_id: OrbitId,
}

impl OrbitIdFactory {
    fn next_id(&mut self) -> OrbitId {
        self.next_id += 1;
        self.next_id
    }
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
    orbit_id: OrbitId,
    c_ref: Complex,
    c_ref_df: ComplexDf,
    orbit: Vec<Complex>,
    gpu_payload: OrbitGpuPayload,
    escape_index: Option<u32>,
    max_lambda: Float,
    weights: HeuristicWeights,
    current_score: i64,
    creation_time: time::Instant,
    creation_frame_id: u64,
}

impl PartialEq for ReferenceOrbit {
    fn eq(&self, other: &Self) -> bool {
        self.orbit_id == other.orbit_id
    }
}

impl Eq for ReferenceOrbit {}

#[derive(Debug)]
struct OrbitResult {
    orbit: Vec<Complex>,
    escape_index: Option<u32>,
}

#[derive(Clone, Debug)]
struct OrbitGpuPayload {
    re_hi: Vec<f32>,
    re_lo: Vec<f32>,
    im_hi: Vec<f32>,
    im_lo: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct TileLevel {
    /// Logical level index (0 = coarse, 1 = fine, etc.)
    pub level: u32,
    /// Size of one tile in complex-plane units
    /// (e.g. 0.25, 2.5e-6, ...)
    pub tile_size: Float,
    pub influence_radius_factor: f64, // between 1 & 2 
    /// Optional soft limit on how many orbits a tile should keep
    pub max_orbits_per_tile: usize,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct TileId {
    pub tx: i64, // floor(re / tile_size)
    pub ty: i64, // floor(im / tile_size)
}

impl TileId {
    fn from_point(c: &Complex, tile_size: &Float) -> Self {
        let mut re = c.real().clone();
        let mut im = c.imag().clone();
        re /= tile_size;
        im /= tile_size;

        let tx = re.floor().to_f64() as i64;
        let ty = im.floor().to_f64() as i64;
        Self { tx, ty }
    }
}

#[derive(Clone, Debug)]
pub struct TileGeometry {
    /// Center of the tile in the complex plane
    pub center: Complex,
    pub radius: Float,
    pub level: u32,
}

impl TileGeometry {
    fn from_id(tile_id: &TileId, tile_size: &Float, level: u32) -> Self {
        let mut center_re = tile_size.clone() * tile_id.tx;
        let mut center_im = tile_size.clone() * tile_id.ty;

        center_re += tile_size.clone() / 2;
        center_im += tile_size.clone() / 2;

        Self {
            center: Complex::with_val(
                tile_size.prec(),
                (center_re, center_im)
            ),
            radius: tile_size.clone() / 2,
            level,
        }
    }

    pub fn center(&self) -> &Complex {
        &self.center
    }

    pub fn half_diagonal(&self) -> Float {
        let mut rad = self.radius.clone();
        rad *= &self.radius;
        rad *= 2;
        Float::sqrt(rad)
    }
}

#[derive(Clone, Debug)]
struct TileOrbitView {
    /// Tile identity (stable)
    tile: TileId,
    /// Geometry of this tile
    geometry: TileGeometry,
    /// Weak refs into the global LivingOrbits pool
    weak_orbits: Vec<WeakOrbit>,
    local_score: f64,
    max_orbits_per_tile: usize,
    /// Timestamp for decay / aging heuristics
    last_updated: time::Instant
}

impl TileOrbitView {
    fn new(tile_id: TileId, level: &TileLevel) -> Self {
        Self {
            tile: tile_id.clone(), 
            geometry: TileGeometry::from_id(&tile_id, &level.tile_size, level.level),
            weak_orbits: Vec::<WeakOrbit>::new(),
            local_score: 0.0,
            max_orbits_per_tile: level.max_orbits_per_tile,
            last_updated: time::Instant::now(),
        }
    }
}

impl ScoutEngine {
    pub fn new(window: Arc<Window>, config: ScoutConfig) -> Self {
        let config = Arc::new(config);
        let thread_pool = ThreadPool::new().expect("Failed to build ThreadPool for ScoutEngine");
        let living_orbits = Arc::new(Mutex::new(Vec::new()));
        let tile_registry = Arc::new(Mutex::new(HashMap::new()));

        // Create asynch channels for communicating between the render loop/thread and 
        // the long-lived ScoutEngine async task
        let (cameara_snapshot_tx, cameara_snapshot_rx) = channel::mpsc::channel::<signals::CameraSnapshot>(10);
        let (gpu_feedback_tx, gpu_feedback_rx) = channel::mpsc::channel::<signals::GpuFeedback>(10);

        // Launch ScoutEngine's long-lived task, which 'blocks' on the above async channels
        thread_pool.spawn_ok(Self::scout_worker(window.clone(), config.clone(), thread_pool.clone(),
            living_orbits.clone(), tile_registry.clone(), cameara_snapshot_rx, gpu_feedback_rx));

        Self {
            window: window, config, thread_pool, living_orbits, tile_registry,
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

    // Determine which tiles lie 'underneath' the current viewport, then return
    // a list of tiles that has been translated/packed for consumption via the GPU.
    // NOTE: GPU packing of orbits is paid up-front during orbit creation, which means 
    // this call should be relativly cheap to make, where it's only job is to map pixel
    // space to the complex plane and retriving the tiles anchored to these complex 
    // cordinates. 
    // NOTE: There is no guarantee that tile views have any orbits in lists at the 
    // moment of this query. ScoutEngine itself will request redraws however, which 
    // should trigger the Scene to invoke this method again.
    pub fn query_tiles_under_viewport(&self, center: &Complex, scale: &Float, 
            viewport_aspect: f64) -> Vec<signals::TileOrbitViewDf> {
        let mut vp_w = scale.clone() * viewport_aspect;
        let mut vp_h = scale.clone();

        let mut tile_size = self.config.tile_levels[0].tile_size.clone();
        tile_size /= 2;
        vp_w += &tile_size;
        vp_h += &tile_size;

        let mut bot_left = center.clone();
        let (bl_re, bl_im) = bot_left.as_mut_real_imag();
        *bl_re -= &vp_w;
        *bl_im -= &vp_h;

        let mut top_right = center.clone();
        let (tr_re, tr_im) = top_right.as_mut_real_imag();
        *tr_re += &vp_w;
        *tr_im += &vp_h;

        debug!("Query tiles under viewport. bounding box is \n\t\t\t\t\t[({:?} {:?})\n\t\t\t\t\t ({:?} {:?})]", 
                bl_re, tr_re, bl_im, tr_im);

        let mut level_0_tile_count: u32 = 0;
        let mut level_1_tile_count: u32 = 0;
        let mut num_tiles_in_bounds: u32 = 0;

        let mut df_tiles = Vec::<signals::TileOrbitViewDf>::new();
        let reg = self.tile_registry.lock().unwrap();
        for ((t_level, tile_id), view) in reg.iter() {
            let tile_re = view.geometry.center.real();
            let tile_im = view.geometry.center.imag();

            if   tile_re > bl_re && tile_re < tr_re  
              && tile_im > bl_im && tile_im < tr_im && *t_level > 0 {
                let mut live_orbits = Vec::<LiveOrbit>::new();
                for weak_orb in &view.weak_orbits {
                    if let Some(orb) = weak_orb.upgrade() {
                        live_orbits.push(orb);
                    }
                }
                let df_orbits = live_orbits.iter().map(|s_orb| {
                    let orb = s_orb.lock().unwrap();
                    signals::ReferenceOrbitDf {
                        orbit_id: orb.orbit_id,
                        c_ref: orb.c_ref_df,
                        orbit_re_hi: orb.gpu_payload.re_hi.clone(),
                        orbit_re_lo: orb.gpu_payload.re_lo.clone(),
                        orbit_im_hi: orb.gpu_payload.im_hi.clone(),
                        orbit_im_lo: orb.gpu_payload.im_lo.clone(),
                        escape_index: orb.escape_index,
                        creation_time: orb.creation_time,
                    }
                }).collect();

                df_tiles.push(signals::TileOrbitViewDf {
                    tile: tile_id.clone(),
                    geometry: view.geometry.clone(),
                    orbits: df_orbits,
                });

                debug!("Found tile {:?} with {} orbits",
                    tile_id, view.weak_orbits.len());
                num_tiles_in_bounds += 1;
            }

            if *t_level == 0 {level_0_tile_count += 1;}
            if *t_level == 1 {level_1_tile_count += 1;} 
        }

        debug!("Level 0 tile count is {}.\tLevel 1 tile count is {}. Total number of tiles found in-bounds {}", 
            level_0_tile_count, level_1_tile_count, num_tiles_in_bounds);
        df_tiles
    }

    pub fn diagnostics(&self) -> signals::ScoutDiagnostics {
        signals::ScoutDiagnostics {
            timestamp: time::SystemTime::now(),
            message: String::new()
        }
    }

    // ScoutEngine's internal work loop, a long-lived future that uses select to poll the 
    // the camera snaphot & gpu feedback channels.
    async fn scout_worker(window: Arc<Window>, config: Arc<ScoutConfig>, 
            tp: ThreadPool, living_orbits: LivingOrbits, tile_registry: TileRegistry,
            cameara_snapshot_rx: CameraSnapshotReceiver, gpu_feedback_rx: GpuFeedbackReceiver) {
        
        let id_fac = Arc::new(Mutex::new(OrbitIdFactory {next_id: 0}));

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
                                window.clone(), config.clone(), tp.clone(), id_fac.clone(),
                                living_orbits.clone(), tile_registry.clone(), Arc::new(snapshot)));
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
            tp: ThreadPool, id_fac: OrbitIdFact, 
            living_orbits: LivingOrbits, tile_registry: TileRegistry, 
            snapshot: Arc<signals::CameraSnapshot>) {
        let seeds = Self::create_new_orbit_seeds_from_camera_snapshot(snapshot.clone()).await;
        debug!("Created orbit seeds from snapshot: {:?}", seeds);

        // Schedule thread-pool execution for the creation of reference orbits from seeds.
        let handles = seeds.iter().fold(Vec::<RemoteHandle<LiveOrbit>>::new(), |mut acc, seed| {
            let res = tp.spawn_with_handle(
                Self::create_new_reference_orbit(seed.clone(), id_fac.clone(), config.max_orbits, config.rug_precision,
                    snapshot.frame_stamp.frame_id));
            if let Ok(h) = res {
                acc.push(h);
            };
            acc
        });

        // Wait for all the orbit computations to complete on the thread-pool
        let results: Vec<LiveOrbit> = join_all(handles).await;
        let r_len = &results.len();

        // Insert new orbits into living orbits
        for orb in &results {
            let mut lo_mut = living_orbits.lock().unwrap();
            lo_mut.push(orb.clone());
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

        // Insert into our complex-anchored tiles.
        Self::insert_new_orbits_into_tile_registry(config.clone(), &results, tile_registry.clone()).await;

        // Rank orbits in each TileOrbitView of the TileRegistry
        Self::rank_tile_registry_orbits(tile_registry.clone()).await;

        // Trim the pool
        Self::trim_living_orbits(living_orbits.clone()).await;

        let orb_pool_g = living_orbits.lock().unwrap();
        for live_orb in orb_pool_g.iter() {
            let orb = live_orb.lock().unwrap();
            debug!("Orbit {} has score {}\tFrom weights: w_dist={} w_depth={} w_age={}\tAt c_ref_df={:?}", 
                orb.orbit_id, orb.current_score, orb.weights.w_dist, orb.weights.w_depth, orb.weights.w_age, orb.c_ref_df);
        };
        drop(orb_pool_g);

        // Send a signal to the winit window to wake up the render loop and redraw the viewport
        // This mechinism is largely in place so that the render loop need not run continually 
        // and communications to/from the GPU can be tightly controlled. 
        window.request_redraw();
    }

    // Fast operation, can be awaited.
    async fn create_new_orbit_seeds_from_camera_snapshot(snapshot: Arc<signals::CameraSnapshot>) -> Vec<Complex> {
        let mut seeds = Vec::<Complex>::new();
        let center = snapshot.center.clone();

        // Stay very simple now, with a 2x2 pattern now, i.e. top-bottom-left-right-middle of the viewport
        seeds.push(center.clone());

        seeds
    }

    async fn create_new_reference_orbit(c_ref: Complex, id_fac: OrbitIdFact, 
        max_iter: u32, prec: u32, snap_frame: u64) 
            -> LiveOrbit {
        let orbit_id = id_fac.lock().unwrap().next_id();
        let orbit_result = Self::compute_reference_orbit(&c_ref, max_iter, prec).await;
        let c_ref_df = ComplexDf::from_complex(&c_ref);
        let cdf_orbit: Vec<ComplexDf> = orbit_result.orbit.iter().map(|c| 
            ComplexDf::from_complex(c)).collect();
        let creation_time = time::Instant::now();
        
        let mut gpu_payload = OrbitGpuPayload{re_hi: Vec::<f32>::new(), re_lo: Vec::<f32>::new(), im_hi: Vec::<f32>::new(), im_lo: Vec::<f32>::new()};
        for cdf in cdf_orbit {
            gpu_payload.re_hi.push(cdf.re.hi);
            gpu_payload.re_lo.push(cdf.re.lo);
            gpu_payload.im_hi.push(cdf.im.hi);
            gpu_payload.im_lo.push(cdf.im.lo);
        }

        info!("Creating new ReferenceOrbit orbit_id={} at time={:?} and snapshot frame_id={}", 
            orbit_id, creation_time, snap_frame);
        Arc::new(Mutex::new(ReferenceOrbit{ orbit_id,
            c_ref: c_ref.clone(), c_ref_df,
            orbit: orbit_result.orbit, gpu_payload,
            escape_index: orbit_result.escape_index,
            max_lambda: Float::with_val(prec, 0.0),
            weights: HeuristicWeights{ w_dist: 0.0, w_depth: 0.0, w_age: 0.0 },
            current_score: 0,
            creation_time,
            creation_frame_id: snap_frame
        }))
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

        for s_orb in orb_pool_l.iter_mut() {
            if let Some(ref cam_snap) = snapshot {
                let mut orb = s_orb.lock().unwrap();
                let delta = orb.c_ref.clone() - &cam_snap.center;
                orb.weights.w_dist = delta.abs().real().to_f64() / &cam_snap.scale.to_f64();
                orb.weights.w_dist *= 100.0;

                orb.weights.w_depth = if let Some(i) = orb.escape_index {i.into()} else {orb.orbit.len() as f64};
                orb.weights.w_depth *= -0.1;

                // Frame id's should always be increasing, otherwize this breaks
                orb.weights.w_age = (cam_snap.frame_stamp.frame_id as f32 - orb.creation_frame_id as f32) as f64;
                orb.weights.w_age /= 5.0;
            }
        };
    }

    // Simply adds all the weights in HeuristicWeights by first vectorizing the strcture,
    // which is possible because all values are the same type, and taking a sum of all 
    // elements in the vector, using an iterator.
    async fn score_living_orbits(living_orbits: LivingOrbits) {
        let mut orb_pool_l = living_orbits.lock().unwrap();
        for s_orb in orb_pool_l.iter_mut() {
            let mut orb = s_orb.lock().unwrap();
            orb.current_score = orb.weights.vectorize().iter().sum::<f64>() as i64;
        };
    }

    async fn rank_living_orbits(living_orbits: LivingOrbits) {
        // convert the orbit pool into a vector for sorting
        let mut lom = living_orbits.lock().unwrap();

        // Sort the vector if orbit's by it's score.
        lom.sort_by_key(|orb| orb.lock().unwrap().current_score);
    }

    // Ultimatly, the TileRegistry is the best place to assess which ref-orbit will provide
    // the best perturbation results for any given pixel in the viewport. It is critical to 
    // understand that these tiles are not locked to the viewport itself, but instead they 
    // are anchored to the complex plane. Once a Tile is created, it will remain within the 
    // registry for the duration of the program, which is why the list of orbits found within
    // TileOrbitView contains Weak references, as living_orbits is regularly trimmed.
    //
    // It is also important to understand that incomming orbits may not survive long insde the
    // TileOrbitView lists, as these lists are meant to be kept very short, for the purposes 
    // of per-pixel rebasing on the GPU. Maintining a healthy per-pixel perturbation is the 
    // entire purpose for tiles, and also why these tiles divide complex geometry, and not 
    // pixel-space. Maintining a low DeltaC from the reference orbit is the first step, but 
    // because of the nature of Dynamical Systems, fractal geometry must be thought of in 
    // (x,y,i) terms, hence the need for a list of orbits, with it's own ranking, and with
    // feedback from per-pixel perturbation on the GPU, giving us insight into the behavior 
    // of i, which can can encounter rapid and unexpected change. 
    async fn insert_new_orbits_into_tile_registry(config: Arc<ScoutConfig>,
            new_orbits: &Vec<LiveOrbit>, tile_registry: TileRegistry) {
        let mut registry = tile_registry.lock().unwrap();
        for level in &config.tile_levels {
            let tile_size = level.tile_size.clone();
            let influence_radius = tile_size.clone() * level.influence_radius_factor;

            for orbit in new_orbits {
                let orbit_guard = orbit.lock().unwrap();
                let c_ref = orbit_guard.c_ref.clone();
                let orbit_id = orbit_guard.orbit_id;
                drop(orbit_guard);

                // Enumerate candidate tiles
                let candidates =
                    tiles_influenced_by_orbit(&c_ref, &tile_size, &influence_radius);

                for tile_id in candidates {
                    let view = registry.entry((level.level, tile_id.clone())).or_insert_with(|| {
                        TileOrbitView::new(tile_id, &level)
                    });

                    if tile_overlaps_orbit(&view.geometry, &c_ref, &influence_radius) {
                        // Insert weak reference if not already present
                        let weak = Arc::downgrade(&orbit);

                        let already_present = view.weak_orbits
                            .iter()
                            .any(|w| w.ptr_eq(&weak));

                        if !already_present {
                            view.weak_orbits.push(weak);
                            view.last_updated = time::Instant::now();
                        }
                    }
                }
            }
        }
    }

    // Rank current orbits within the Tile Registory. 
    // This also includes trunkating each TileOrbitView orbit list to it's max allowed 
    // orbits, depending on the tile's level. Also, if the weak reference to the orbit
    // is no longer valid, it will be removed.
    async fn rank_tile_registry_orbits(tile_registry: TileRegistry) {
        let mut reg = tile_registry.lock().unwrap();
        for ((_t_level, _tile_id), view) in reg.iter_mut() {
            let mut live_orbits = Vec::<LiveOrbit>::new();
            for weak_orb in view.weak_orbits.iter() {
                if let Some(orb) = weak_orb.upgrade() {
                    live_orbits.push(orb);
                }
            }
            view.weak_orbits.clear();
            live_orbits.sort_by_key(|orb| score_tile_view_orbit(orb.clone()));
            //live_orbits.truncate(view.max_orbits_per_tile);
            // Ensure only live orbits are kept, even if we revert them
            // back to weak references
            for orb in live_orbits {
                view.weak_orbits.push(Arc::downgrade(&orb));
            }
        }
    }

    async fn trim_living_orbits(living_orbits: LivingOrbits) {
        let mut lom = living_orbits.lock().unwrap(); 

        lom.truncate(MAX_ORBIT_POOL_SIZE);
    }
}

fn tiles_influenced_by_orbit(
    c_ref: &Complex,
    tile_size: &Float,
    influence_radius: &Float,
) -> Vec<TileId> {
    let base = TileId::from_point(c_ref, tile_size);

    let mut inf_per_rad = influence_radius.clone();
    inf_per_rad /= tile_size;
    let tiles_per_radius = inf_per_rad.ceil().to_f64() as i64;

    let mut out = Vec::<TileId>::new();

    for dx in -tiles_per_radius..=tiles_per_radius {
        for dy in -tiles_per_radius..=tiles_per_radius {
            out.push(TileId {
                tx: base.tx + dx, 
                ty: base.ty + dy,
            });
        }
    }
    //debug!("Tiles Influnced by Orbit c_ref={:?} and tile_size={:?} and influence_radius={:?} tiles_per_radius={} len={}\n\t\t\t----> {:?}",
    //    c_ref, tile_size, influence_radius, tiles_per_radius, out.len(), out);
    out
}

fn tile_overlaps_orbit(
    tile_geom: &TileGeometry,
    c_ref: &Complex,
    influence_radius: &Float,
) -> bool {
    let dx = tile_geom.center().real().clone() - c_ref.real();
    let dy = tile_geom.center().imag().clone() - c_ref.imag();

    let dx2 = dx.clone() * &dx;
    let dy2 = dy.clone() * &dy;

    let dist = Float::sqrt(dx2 + dy2);
    let threshold = influence_radius + tile_geom.half_diagonal();

    dist <= threshold
}


fn score_tile_view_orbit(view_orbit: LiveOrbit) -> i64 {
    let orb = view_orbit.lock().unwrap();
    (  orb.weights.w_depth 
     + orb.weights.w_age) as i64
}
