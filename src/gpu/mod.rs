pub mod ash_backend;
pub use ash_backend::AshBackend;

use glam::Vec3;

use crate::{
    core::{collider::Collider, rigidbody::RigidBody},
    utils::allocator::Arena,
};

/// Snapshot of world data converted into a GPU-friendly structure-of-arrays layout.
#[derive(Debug, Default, Clone)]
pub struct GpuWorldState {
    pub positions: Vec<Vec3>,
    pub velocities: Vec<Vec3>,
    pub inverse_masses: Vec<f32>,
    pub collider_bounds: Vec<f32>,
}

impl GpuWorldState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Synchronizes the CPU arenas into the GPU-friendly buffers.
    pub fn sync(&mut self, bodies: &Arena<RigidBody>, colliders: &Arena<Collider>) {
        self.positions.clear();
        self.velocities.clear();
        self.inverse_masses.clear();
        self.collider_bounds.clear();

        for body_id in bodies.ids() {
            if let Some(body) = bodies.get(body_id) {
                self.positions.push(body.transform.position);
                self.velocities.push(body.velocity.linear);
                self.inverse_masses.push(body.inverse_mass);
            }
        }

        for collider_id in colliders.ids() {
            if let Some(collider) = colliders.get(collider_id) {
                self.collider_bounds.push(collider.bounding_radius());
            }
        }
    }

    pub fn body_count(&self) -> usize {
        self.positions.len()
    }
}

/// Trait implemented by GPU/compute backends that can accelerate parts of the pipeline.
pub trait ComputeBackend: Send + Sync {
    fn name(&self) -> &str;

    /// Called once per step after the world state has been synchronized into GPU-friendly buffers.
    fn prepare_step(&self, _state: &GpuWorldState) {}

    /// Optional hook for accelerating broad-phase workloads.
    fn dispatch_broadphase(&self, _state: &GpuWorldState) {}

    /// Optional hook for accelerating solver workloads.
    fn dispatch_solver(&self, _state: &GpuWorldState) {}
}

/// Default backend that keeps all work on the CPU.
#[derive(Debug, Default)]
pub struct NoopBackend;

impl NoopBackend {
    pub fn new() -> Self {
        Self
    }
}

impl ComputeBackend for NoopBackend {
    fn name(&self) -> &str {
        "cpu-noop"
    }
}
