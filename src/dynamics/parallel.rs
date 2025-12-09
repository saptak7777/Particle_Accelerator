use rayon::prelude::*;

use crate::core::rigidbody::RigidBody;

/// Parallel integrator using Rayon for data-parallel updates.
pub struct ParallelIntegrator {
    pub dt: f32,
}

impl ParallelIntegrator {
    pub fn new(dt: f32) -> Self {
        Self { dt }
    }

    pub fn step(&self, bodies: &mut [RigidBody]) {
        bodies.par_iter_mut().for_each(|body| {
            if !body.is_static {
                body.velocity.linear += body.acceleration * self.dt;
                body.acceleration = glam::Vec3::ZERO;
            }
        });

        bodies.par_iter_mut().for_each(|body| {
            if !body.is_static {
                body.transform.position += body.velocity.linear * self.dt;
            }
        });
    }
}
