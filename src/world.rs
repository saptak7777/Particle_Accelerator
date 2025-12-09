use crate::{
    collision::{broadphase::BroadPhase, contact::ContactManifold},
    config::{
        DEFAULT_BROADPHASE_CELL_SIZE, DEFAULT_GRAVITY, DEFAULT_SOLVER_ITERATIONS, DEFAULT_TIME_STEP,
    },
    core::{collider::Collider, rigidbody::RigidBody},
    dynamics::{
        forces::ForceRegistry,
        integrator::Integrator,
        solver::{ConstraintSolver, Contact},
    },
    utils::allocator::{Arena, EntityId},
};
use glam::Vec3;

/// Central simulation container orchestrating all subsystems.
pub struct PhysicsWorld {
    pub bodies: Arena<RigidBody>,
    pub colliders: Arena<Collider>,
    pub integrator: Integrator,
    pub solver: ConstraintSolver,
    pub gravity: Vec3,
    pub time_accumulated: f32,
    pub time_step: f32,
    pub force_registry: ForceRegistry,
    broadphase: BroadPhase,
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
            solver: ConstraintSolver::new(DEFAULT_SOLVER_ITERATIONS),
            gravity: Vec3::from_slice(&DEFAULT_GRAVITY),
            time_accumulated: 0.0,
            time_step: ts,
            force_registry: ForceRegistry::new(),
            broadphase: BroadPhase::new(DEFAULT_BROADPHASE_CELL_SIZE),
        }
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

    /// Advances the simulation using a fixed timestep accumulator.
    pub fn step(&mut self, dt: f32) {
        self.time_accumulated += dt;

        while self.time_accumulated >= self.time_step {
            self.time_accumulated -= self.time_step;

            self.apply_gravity();
            self.force_registry.apply_all(&mut self.bodies, self.time_step);

            let contacts = self.generate_contacts();

            self.solver.solve(&mut self.bodies, &contacts);
            self.integrator.step(&mut self.bodies);
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
            }
        }

        contacts
    }
}
