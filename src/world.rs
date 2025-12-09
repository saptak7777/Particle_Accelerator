use std::collections::HashMap;

use crate::{
    collision::{
        broadphase::BroadPhase,
        ccd::CCDDetector,
        contact::ContactManifold,
        queries::{Raycast, RaycastHit, RaycastQuery},
    },
    config::{DEFAULT_BROADPHASE_CELL_SIZE, DEFAULT_GRAVITY, DEFAULT_TIME_STEP},
    core::{
        collider::{Collider, CollisionFilter},
        constraints::Joint,
        rigidbody::RigidBody,
    },
    dynamics::{
        forces::ForceRegistry,
        integrator::Integrator,
        island::IslandManager,
        solver::{Contact, PGSSolver},
    },
    gpu::{ComputeBackend, GpuWorldState, NoopBackend},
    utils::{
        allocator::{Arena, EntityId},
        logging::ScopedTimer,
    },
};
use glam::Vec3;
use rayon::prelude::*;

/// Central simulation container orchestrating all subsystems.
pub struct PhysicsWorld {
    pub bodies: Arena<RigidBody>,
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
}

impl PhysicsWorld {
    pub fn new(time_step: f32) -> Self {
        let ts = if time_step <= 0.0 {
            DEFAULT_TIME_STEP
        } else {
            time_step
        };

        Self {
            bodies: Arena::new(),
            colliders: Arena::new(),
            integrator: Integrator::new(ts, 2),
            solver: PGSSolver::new(),
            joints: Vec::new(),
            gravity: Vec3::from_slice(&DEFAULT_GRAVITY),
            time_accumulated: 0.0,
            time_step: ts,
            force_registry: ForceRegistry::new(),
            broadphase: BroadPhase::new(DEFAULT_BROADPHASE_CELL_SIZE),
            islands: IslandManager::new(),
            ccd: CCDDetector::new(),
            parallel_enabled: false,
            gpu_state: GpuWorldState::new(),
            gpu_backend: Box::new(NoopBackend::new()),
        }
    }

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

    pub fn add_joint(&mut self, joint: Joint) {
        self.joints.push(joint);
    }

    pub fn clear_joints(&mut self) {
        self.joints.clear();
    }

    pub fn add_rigidbody(&mut self, body: RigidBody) -> EntityId {
        let id = self.bodies.insert(body);
        if let Some(stored) = self.bodies.get_mut(id) {
            stored.id = id;
        }
        id
    }

    pub fn add_collider(&mut self, collider: Collider) -> EntityId {
        let id = self.colliders.insert(collider);
        if let Some(stored) = self.colliders.get_mut(id) {
            stored.id = id;
        }
        id
    }

    pub fn body(&self, id: EntityId) -> Option<&RigidBody> {
        self.bodies.get(id)
    }

    pub fn body_mut(&mut self, id: EntityId) -> Option<&mut RigidBody> {
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

            self.apply_gravity();
            self.force_registry.apply_all(&mut self.bodies, self.time_step);
            self.sync_gpu_state();
            self.gpu_backend.dispatch_broadphase(&self.gpu_state);

            let contacts = {
                let _timer = ScopedTimer::new("contacts::generate");
                self.generate_contacts()
            };
            {
                let _timer = ScopedTimer::new("islands::build");
                self.islands
                    .build_islands(&self.bodies, &contacts, &self.joints);
            }

            let solver_label = if self.parallel_enabled {
                "solver::parallel"
            } else {
                "solver::sequential"
            };
            let _solver_timer = ScopedTimer::new(solver_label);
            if self.parallel_enabled {
                self.solve_islands_parallel();
            } else {
                self.solve_islands_sequential();
            }
            self.gpu_backend.dispatch_solver(&self.gpu_state);
            {
                let _timer = ScopedTimer::new("integrator");
                self.integrator.step(&mut self.bodies);
            }
            {
                let _timer = ScopedTimer::new("sleeping::update");
                self.islands.update_sleeping(&mut self.bodies);
            }
        }
    }

    fn apply_gravity(&mut self) {
        for body in self.bodies.iter_mut() {
            if body.is_static {
                continue;
            }
            body.acceleration += self.gravity * body.gravity_scale;
        }
    }

    fn sync_gpu_state(&mut self) {
        self.gpu_state.sync(&self.bodies, &self.colliders);
        self.gpu_backend.prepare_step(&self.gpu_state);
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

            let body_a = match self.bodies.get(collider_a.rigidbody_id) {
                Some(body) => body,
                None => continue,
            };
            let body_b = match self.bodies.get(collider_b.rigidbody_id) {
                Some(body) => body,
                None => continue,
            };

            if let Some(manifold) = ContactManifold::generate(collider_a, body_a, collider_b, body_b)
            {
                contacts.extend(manifold.contacts);
                continue;
            }

            if collider_a.is_trigger || collider_b.is_trigger {
                continue;
            }

            if let Some(ccd_hit) =
                self.ccd
                    .detect_ccd(body_a, collider_a, body_b, collider_b, self.time_step)
            {
                contacts.push(ccd_hit.contact);
            }
        }

        contacts
    }

    fn filters_match(filter_a: &CollisionFilter, filter_b: &CollisionFilter) -> bool {
        (filter_a.mask & filter_b.layer) != 0 && (filter_b.mask & filter_a.layer) != 0
    }

    fn solve_islands_sequential(&mut self) {
        for island in self.islands.islands() {
            if !island.is_awake {
                continue;
            }
            self.solver
                .solve(&mut self.bodies, &island.contacts, &island.joints);
        }
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
            solver.solve_island_slice(&mut job.bodies, &job.id_map, &job.contacts, &job.joints);
        });

        for job in jobs {
            for (id, body_state) in job.ids.into_iter().zip(job.bodies.into_iter()) {
                if let Some(slot) = self.bodies.get_mut(id) {
                    *slot = body_state;
                }
            }
        }
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
}

struct IslandJob {
    ids: Vec<EntityId>,
    bodies: Vec<RigidBody>,
    id_map: HashMap<EntityId, usize>,
    contacts: Vec<Contact>,
    joints: Vec<Joint>,
}
