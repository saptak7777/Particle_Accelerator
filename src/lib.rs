//! Particle Accelerator – Physics Engine for Rust.
//!
//! This crate exposes a modular physics engine architecture built around
//! ECS-friendly patterns, offering collision detection, dynamics,
//! constraint solving, and utility modules out of the box.

pub mod collision;
pub mod config;
pub mod core;
pub mod dynamics;
pub mod gpu;
pub mod utils;
pub mod world;

pub use glam::{Mat3, Mat4, Quat, Vec3};

pub use core::soa::{BodyMut, BodyRef};

pub use collision::{
    broadphase::BroadPhase,
    contact::ContactManifold,
    queries::{Raycast, RaycastHit, RaycastQuery},
};
pub use core::{
    collider::{Collider, ColliderShape, CollisionFilter},
    rigidbody::RigidBody,
    types::{MassProperties, Material, Transform, Velocity},
};
pub use dynamics::{
    forces::{DragForce, ForceGenerator, ForceRegistry, GravityForce, SpringForce},
    integrator::Integrator,
    solver::{ConstraintSolver, Contact},
};
pub use gpu::{ComputeBackend, GpuWorldState, NoopBackend};
pub use utils::allocator::{Arena, EntityId, GenerationalId};
pub use world::PhysicsWorld;

/// High-level convenience wrapper that owns a [`PhysicsWorld`].
pub struct PhysicsEngine {
    world: PhysicsWorld,
}

impl PhysicsEngine {
    /// Creates a new physics engine with the provided fixed timestep.
    pub fn new(timestep: f32) -> Self {
        Self {
            world: PhysicsWorld::new(timestep),
        }
    }

    /// Adds a rigid body to the world and returns its generated [`EntityId`].
    pub fn add_body(&mut self, body: RigidBody) -> EntityId {
        self.world.add_rigidbody(body)
    }

    /// Adds a collider associated with a rigid body and returns its [`EntityId`].
    pub fn add_collider(&mut self, collider: Collider) -> EntityId {
        self.world.add_collider(collider)
    }

    /// Advances the simulation by the provided delta time.
    pub fn step(&mut self, dt: f32) {
        self.world.step(dt);
    }

    /// Enables or disables parallel execution for integration and solver passes.
    pub fn set_parallel_enabled(&mut self, enabled: bool) {
        self.world.set_parallel_enabled(enabled);
    }

    /// Returns whether the engine is currently using parallel execution.
    pub fn parallel_enabled(&self) -> bool {
        self.world.parallel_enabled()
    }

    /// Immutable access to a rigid body by id.
    pub fn get_body(&self, id: EntityId) -> Option<BodyRef<'_>> {
        self.world.body(id)
    }

    /// Mutable access to a rigid body by id.
    pub fn get_body_mut(&mut self, id: EntityId) -> Option<BodyMut<'_>> {
        self.world.body_mut(id)
    }
}
