// use std::collections::HashMap;

use crate::{
    collision::{
        broadphase::BroadPhase,
        ccd::CCDDetector,
        contact::{ContactManifold, ManifoldCache, ManifoldDebugInfo},
        queries::{Raycast, RaycastHit, RaycastQuery},
    },
    config::{DEFAULT_BROADPHASE_CELL_SIZE, DEFAULT_GRAVITY, DEFAULT_TIME_STEP},
    core::{
        articulations::Multibody,
        collider::{Collider, CollisionFilter},
        constraints::Joint,
        rigidbody::RigidBody,
        soa::{BodiesSoA, BodyMut, BodyRef},
    },
    dynamics::{
        forces::ForceRegistry,
        integrator::Integrator,
        island::IslandManager,
        pci::PredictiveCorrectiveIntegrator,
        solver::{Contact, PGSSolver, SolverStepMetrics},
    },
    gpu::{ComputeBackend, GpuWorldState, NoopBackend},
    utils::{
        allocator::{Arena, EntityId},
        logging::ScopedTimer,
        profiling::PhysicsProfiler,
    },
};
use glam::Vec3;
use log::debug;
// use rayon::prelude::*;
use std::time::Duration;

/// Central simulation container orchestrating all subsystems.
pub struct PhysicsWorld {
    pub bodies: BodiesSoA,
    pub colliders: Arena<Collider>,
    pub integrator: Integrator,
    pub solver: PGSSolver,
    pub joints: Vec<Joint>,
    pub gravity: Vec3,
    pub time_accumulated: f32,
    pub time_step: f32,
    pub force_registry: ForceRegistry,
    broadphase: BroadPhase,
    islands: IslandManager,
    ccd: CCDDetector,
    parallel_enabled: bool,
    gpu_state: GpuWorldState,
    gpu_backend: Box<dyn ComputeBackend>,
    manifold_cache: ManifoldCache,
    frame_index: u32,
    manifold_debug_logging: bool,
    last_solver_metrics: SolverStepMetrics,
    solver_metrics_logging: bool,
    pci: PredictiveCorrectiveIntegrator,
    pci_enabled: bool,
    pub profiler: PhysicsProfiler,
    pub articulated_bodies: Arena<Multibody>,
}

impl PhysicsWorld {
    pub fn new(time_step: f32) -> Self {
        Self::builder().time_step(time_step).build()
    }

    pub fn builder() -> PhysicsWorldBuilder {
        PhysicsWorldBuilder::new()
    }
}

pub struct PhysicsWorldBuilder {
    time_step: f32,
    gravity: Vec3,
    parallel_enabled: bool,
    gpu_backend: Option<Box<dyn ComputeBackend>>,
}

impl PhysicsWorldBuilder {
    pub fn new() -> Self {
        Self {
            time_step: DEFAULT_TIME_STEP,
            gravity: Vec3::from_slice(&DEFAULT_GRAVITY),
            parallel_enabled: false,
            gpu_backend: None,
        }
    }

    pub fn time_step(mut self, dt: f32) -> Self {
        self.time_step = if dt <= 0.0 { DEFAULT_TIME_STEP } else { dt };
        self
    }

    pub fn gravity(mut self, gravity: Vec3) -> Self {
        self.gravity = gravity;
        self
    }

    pub fn parallel(mut self, enabled: bool) -> Self {
        self.parallel_enabled = enabled;
        self
    }

    pub fn gpu_backend<B: ComputeBackend + 'static>(mut self, backend: B) -> Self {
        self.gpu_backend = Some(Box::new(backend));
        self
    }

    pub fn build(self) -> PhysicsWorld {
        let ts = self.time_step;
        PhysicsWorld {
            bodies: BodiesSoA::new(),
            colliders: Arena::new(),
            integrator: Integrator::new(ts, 2),
            solver: PGSSolver::new(),
            joints: Vec::new(),
            gravity: self.gravity,
            time_accumulated: 0.0,
            time_step: ts,
            force_registry: ForceRegistry::new(),
            broadphase: BroadPhase::new(DEFAULT_BROADPHASE_CELL_SIZE),
            islands: IslandManager::new(),
            ccd: CCDDetector::new(),
            parallel_enabled: self.parallel_enabled,
            gpu_state: GpuWorldState::new(),
            gpu_backend: self
                .gpu_backend
                .unwrap_or_else(|| Box::new(NoopBackend::new())),
            manifold_cache: ManifoldCache::new(),
            frame_index: 0,
            manifold_debug_logging: false,
            last_solver_metrics: SolverStepMetrics::default(),
            solver_metrics_logging: false,
            pci: PredictiveCorrectiveIntegrator::default(),
            pci_enabled: false,
            profiler: PhysicsProfiler::default(),
            articulated_bodies: Arena::new(),
        }
    }
}

impl Default for PhysicsWorldBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl PhysicsWorld {
    pub fn set_parallel_enabled(&mut self, enabled: bool) {
        self.parallel_enabled = enabled;
        self.integrator.set_parallel(enabled);
    }

    pub fn parallel_enabled(&self) -> bool {
        self.parallel_enabled
    }

    pub fn set_gpu_backend<B>(&mut self, backend: B)
    where
        B: ComputeBackend + 'static,
    {
        self.gpu_backend = Box::new(backend);
    }

    pub fn gpu_backend_name(&self) -> &str {
        self.gpu_backend.name()
    }

    pub fn ccd(&self) -> &CCDDetector {
        &self.ccd
    }

    pub fn ccd_mut(&mut self) -> &mut CCDDetector {
        &mut self.ccd
    }

    pub fn set_ccd_enabled(&mut self, enabled: bool) {
        self.ccd.set_enabled(enabled);
    }

    pub fn set_ccd_threshold(&mut self, threshold: f32) {
        self.ccd.set_ccd_threshold(threshold);
    }

    pub fn set_ccd_angular_padding(&mut self, padding: f32) {
        self.ccd.set_angular_padding(padding);
    }

    pub fn raycast(&self, query: &RaycastQuery) -> Vec<RaycastHit> {
        Raycast::cast(query, &self.colliders, &self.bodies)
    }

    pub fn raycast_with_filter<F>(&self, query: &RaycastQuery, filter: F) -> Vec<RaycastHit>
    where
        F: FnMut(EntityId, &Collider) -> bool,
    {
        Raycast::cast_with_filter(query, &self.colliders, &self.bodies, filter)
    }

    pub fn set_manifold_debug_hook<F>(&mut self, hook: Option<F>)
    where
        F: FnMut(&ManifoldDebugInfo) + Send + 'static,
    {
        self.manifold_cache.set_debug_hook(hook);
    }

    pub fn manifold_debug_snapshots(&self) -> Vec<ManifoldDebugInfo> {
        self.manifold_cache.debug_snapshots()
    }

    pub fn set_manifold_logging_enabled(&mut self, enabled: bool) {
        self.manifold_debug_logging = enabled;
    }

    pub fn last_solver_metrics(&self) -> &SolverStepMetrics {
        &self.last_solver_metrics
    }

    pub fn set_solver_metrics_logging(&mut self, enabled: bool) {
        self.solver_metrics_logging = enabled;
    }

    pub fn set_pci_enabled(&mut self, enabled: bool) {
        self.pci_enabled = enabled;
    }

    pub fn set_pci_iterations(&mut self, iterations: u32) {
        self.pci.set_iterations(iterations);
    }

    pub fn add_joint(&mut self, joint: Joint) {
        self.joints.push(joint);
    }

    pub fn add_multibody(&mut self, mb: Multibody) -> EntityId {
        self.articulated_bodies.insert(mb)
    }

    pub fn clear_joints(&mut self) {
        self.joints.clear();
    }

    pub fn add_rigidbody(&mut self, mut body: RigidBody) -> EntityId {
        body.recompute_inverses();
        self.bodies.insert(body)
    }

    pub fn add_collider(&mut self, collider: Collider) -> EntityId {
        let id = self.colliders.insert(collider);
        if let Some(stored) = self.colliders.get_mut(id) {
            stored.id = id;
        }
        id
    }

    pub fn body(&self, id: EntityId) -> Option<BodyRef<'_>> {
        self.bodies.get(id)
    }

    pub fn body_mut(&mut self, id: EntityId) -> Option<BodyMut<'_>> {
        self.bodies.get_mut(id)
    }

    pub fn collider(&self, id: EntityId) -> Option<&Collider> {
        self.colliders.get(id)
    }

    /// Collects contacts for the current world state without advancing the simulation.
    /// Useful for debugging and tests.
    pub fn collect_contacts(&mut self) -> Vec<Contact> {
        self.generate_contacts()
    }

    /// Advances the simulation using a fixed timestep accumulator.
    pub fn step(&mut self, dt: f32) {
        self.time_accumulated += dt;

        while self.time_accumulated >= self.time_step {
            self.time_accumulated -= self.time_step;
            self.frame_index = self.frame_index.wrapping_add(1);
            self.manifold_cache.begin_frame(self.frame_index);

            self.profiler.reset();
            self.profiler.total_frame_time = Duration::ZERO;
            let frame_start = std::time::Instant::now();

            self.apply_gravity();
            self.force_registry
                .apply_all(&mut self.bodies, self.time_step);
            self.sync_gpu_state();
            // TODO: Timer for GPU dispatch?
            self.gpu_backend.dispatch_broadphase(&self.gpu_state);

            let mut contacts = {
                let start = std::time::Instant::now();
                let c = self.generate_contacts();
                self.profiler.broad_phase_time = start.elapsed();
                c
            };
            self.profiler.contact_count = contacts.len();

            {
                let start = std::time::Instant::now();
                self.islands
                    .build_islands(&self.bodies, &contacts, &self.joints);
                self.profiler.narrow_phase_time = start.elapsed();
            }
            self.profiler.active_island_count = self.islands.islands().len();

            {
                let start = std::time::Instant::now();
                /*
                if self.parallel_enabled {
                    self.solve_islands_parallel();
                } else {
                    self.solve_islands_sequential();
                }
                */
                // The global solver is used for SoA-based dynamics.
                self.solver.solve(
                    &mut self.bodies,
                    &self.joints,
                    &mut contacts,
                    self.time_step,
                );

                self.apply_predictive_corrections(&contacts);
                self.profiler.solver_time = start.elapsed();
            }

            self.log_solver_metrics_if_needed();
            self.gpu_backend.dispatch_solver(&self.gpu_state);

            {
                // 5. Articulation Step (ABA)
                for mb in self.articulated_bodies.iter_mut() {
                    /*
                    // Apply Gravity Force to all links
                    for i in 0..mb.links.len() {
                        let link = &mb.links[i];
                        // tau = G(q)
                    }
                    */

                    crate::dynamics::aba::ABASolver::solve(mb, self.gravity);

                    // Integrate Generalized coordinates (Semi-implicit Euler)
                    for i in 0..mb.total_dofs {
                        mb.dq[i] += mb.ddq[i] * self.time_step;
                        mb.q[i] += mb.dq[i] * self.time_step;
                    }

                    mb.update_kinematics();
                }

                let start = std::time::Instant::now();
                self.integrator.step(&mut self.bodies);
                self.profiler.integrator_time = start.elapsed();
            }

            {
                // Sleeping update
                // self.islands.update_sleeping(&mut self.bodies);
            }

            self.manifold_cache.prune_stale();
            self.log_manifolds_if_needed();

            self.profiler.total_frame_time = frame_start.elapsed();
            self.profiler.body_count = self.bodies.len();
            // self.profiler.report();
        }
    }

    fn apply_gravity(&mut self) {
        for body in self.bodies.iter_mut() {
            if body.is_static() {
                continue;
            }
            let force = self.gravity * (*body.gravity_scale);
            *body.acceleration += force;
        }
    }

    fn sync_gpu_state(&mut self) {
        // self.gpu_state.sync(&self.bodies, &self.colliders);
        // self.gpu_backend.prepare_step(&self.gpu_state);
    }

    fn generate_contacts(&mut self) -> Vec<Contact> {
        if self.colliders.len() < 2 {
            return Vec::new();
        }

        let mut contacts = Vec::new();
        let potential_pairs = self
            .broadphase
            .get_potential_pairs(&self.colliders, &self.bodies);

        for (collider_a_id, collider_b_id) in potential_pairs {
            let collider_a = match self.colliders.get(collider_a_id) {
                Some(collider) => collider,
                None => continue,
            };
            let collider_b = match self.colliders.get(collider_b_id) {
                Some(collider) => collider,
                None => continue,
            };

            if !Self::filters_match(&collider_a.collision_filter, &collider_b.collision_filter) {
                continue;
            }

            let (body_a_mut, body_b_mut) = match self
                .bodies
                .get2_mut(collider_a.rigidbody_id, collider_b.rigidbody_id)
            {
                Some(pair) => pair,
                None => continue,
            };

            let rb_a = body_a_mut.to_rigid_body();
            let rb_b = body_b_mut.to_rigid_body();

            if let Some(manifold) = ContactManifold::generate(collider_a, &rb_a, collider_b, &rb_b)
            {
                contacts.extend(
                    self.manifold_cache
                        .update_pair(rb_a.id, rb_b.id, manifold, &rb_a, &rb_b),
                );
                continue;
            }

            if collider_a.is_trigger || collider_b.is_trigger {
                continue;
            }

            if let Some(ccd_hit) =
                self.ccd
                    .detect_ccd(&rb_a, collider_a, &rb_b, collider_b, self.time_step)
            {
                // Advance bodies to the Time-of-Impact (TOI) to ensure contact is resolved
                // at the correct temporal position. The SoA proxies are updated directly.
                let advance_dt = ccd_hit.time_of_impact;
                body_a_mut.transform.position += body_a_mut.velocity.linear * advance_dt;
                body_b_mut.transform.position += body_b_mut.velocity.linear * advance_dt;

                self.manifold_cache.record_contact(&ccd_hit.contact);
                contacts.push(ccd_hit.contact);
                continue;
            }

            if let Some(speculative) = self.ccd.generate_speculative_contact(
                &rb_a,
                collider_a,
                &rb_b,
                collider_b,
                self.time_step,
            ) {
                self.manifold_cache.record_contact(&speculative);
                contacts.push(speculative);
            }
        }

        contacts
    }

    fn filters_match(filter_a: &CollisionFilter, filter_b: &CollisionFilter) -> bool {
        (filter_a.mask & filter_b.layer) != 0 && (filter_b.mask & filter_a.layer) != 0
    }

    fn log_manifolds_if_needed(&self) {
        if !self.manifold_debug_logging {
            return;
        }
        let snapshots = self.manifold_debug_snapshots();
        if snapshots.is_empty() {
            return;
        }
        for snapshot in snapshots.iter().take(5) {
            let avg_depth =
                snapshot.points.iter().map(|p| p.depth).sum::<f32>() / snapshot.points.len() as f32;
            debug!(
                "Manifold {:?}-{:?}: normal {:?} avg_depth {:.4} points {}",
                snapshot.body_a,
                snapshot.body_b,
                snapshot.normal,
                avg_depth,
                snapshot.points.len()
            );
        }
        if snapshots.len() > 5 {
            debug!(
                "Manifold debug logging truncated: showing 5 of {} manifolds",
                snapshots.len()
            );
        }
    }

    fn log_solver_metrics_if_needed(&self) {
        if !self.solver_metrics_logging {
            return;
        }
        let metrics = &self.last_solver_metrics;
        debug!(
            "Solver metrics: islands={} contacts={} joints={} normal_sum={:.4} tangent_sum={:.4} rolling_sum={:.4} torsional_sum={:.4}",
            metrics.islands_solved,
            metrics.contacts_solved,
            metrics.joints_solved,
            metrics.normal_impulse_sum,
            metrics.tangent_impulse_sum,
            metrics.rolling_impulse_sum,
            metrics.torsional_impulse_sum
        );
    }

    fn apply_predictive_corrections(&mut self, contacts: &[Contact]) {
        if !self.pci_enabled {
            return;
        }
        if contacts.is_empty() {
            return;
        }
        let _timer = ScopedTimer::new("pci::apply");
        self.pci.apply(&mut self.bodies, contacts, self.time_step);
    }

    /*
    fn solve_islands_sequential(&mut self) {
        let mut metrics = SolverStepMetrics::default();
        for island in self.islands.islands() {
            if !island.is_awake {
                continue;
            }
            let mut contacts = island.contacts.clone();
            self.solver
                .solve(&mut self.bodies, &island.joints, &mut contacts, self.time_step);
            self.manifold_cache.apply_impulses(&contacts);
            metrics.record_island(&contacts, island.joints.len());
        }
        self.last_solver_metrics = metrics;
    }

    fn solve_islands_parallel(&mut self) {
        let solver = self.solver.clone();
        let mut jobs: Vec<IslandJob> = self
            .islands
            .islands()
            .iter()
            .filter(|island| island.is_awake)
            .filter_map(|island| self.prepare_island_job(island))
            .collect();

        jobs.par_iter_mut().for_each(|job| {
            solver.solve_island_slice(&mut job.bodies, &job.id_map, &mut job.contacts, &job.joints);
        });

        let mut metrics = SolverStepMetrics::default();

        for job in &jobs {
            self.manifold_cache.apply_impulses(&job.contacts);
            metrics.record_island(&job.contacts, job.joints.len());
        }

        for job in jobs {
            for (id, body_state) in job.ids.into_iter().zip(job.bodies.into_iter()) {
                if let Some(slot) = self.bodies.get_mut(id) {
                    *slot = body_state;
                }
            }
        }
        self.last_solver_metrics = metrics;
    }

    fn prepare_island_job(&self, island: &crate::dynamics::island::Island) -> Option<IslandJob> {
        if island.bodies.is_empty() {
            return None;
        }

        let mut ids = Vec::with_capacity(island.bodies.len());
        let mut bodies = Vec::with_capacity(island.bodies.len());
        let mut id_map = HashMap::with_capacity(island.bodies.len());

        for (idx, body_id) in island.bodies.iter().enumerate() {
            if let Some(body) = self.bodies.get(*body_id) {
                ids.push(*body_id);
                bodies.push(body.clone());
                id_map.insert(*body_id, idx);
            }
        }

        if bodies.is_empty() {
            return None;
        }

        Some(IslandJob {
            ids,
            bodies,
            id_map,
            contacts: island.contacts.clone(),
            joints: island.joints.clone(),
        })
    }
    */
}

/*
struct IslandJob {
    ids: Vec<EntityId>,
    bodies: Vec<RigidBody>,
    id_map: HashMap<EntityId, usize>,
    contacts: Vec<Contact>,
    joints: Vec<Joint>,
}
*/
