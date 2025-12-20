pub mod ash_backend;
pub use ash_backend::AshBackend;

use glam::Vec3;

use crate::{
    core::{collider::Collider, rigidbody::RigidBody},
    utils::allocator::Arena,
};

/// GPU-friendly rigid body data.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct GpuBody {
    pub position: Vec3,
    pub radius: f32,
}

/// Snapshot of world data converted into a GPU-friendly structure-of-arrays layout.
#[derive(Debug, Default, Clone)]
pub struct GpuWorldState {
    pub bodies: Vec<GpuBody>,
}

impl GpuWorldState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Synchronizes the CPU arenas into the GPU-friendly buffers.
    pub fn sync(&mut self, bodies: &Arena<RigidBody>, colliders: &Arena<Collider>) {
        self.bodies.clear();

        for body_id in bodies.ids() {
            if let Some(body) = bodies.get(body_id) {
                // Find associated collider radius
                let mut radius = 0.0;
                for col_id in colliders.ids() {
                    if let Some(col) = colliders.get(col_id) {
                        if col.rigidbody_id == body_id {
                            radius = col.bounding_radius();
                            break;
                        }
                    }
                }

                self.bodies.push(GpuBody {
                    position: body.transform.position,
                    radius,
                });
            }
        }
    }

    pub fn body_count(&self) -> usize {
        self.bodies.len()
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
