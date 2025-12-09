use glam::{Quat, Vec3};

use crate::{
    core::rigidbody::RigidBody,
    utils::allocator::Arena,
};

/// Integrator responsible for stepping rigid bodies forward in time.
#[derive(Debug, Clone)]
pub struct Integrator {
    pub dt: f32,
    pub substeps: u32,
    pub gravity: Vec3,
}

impl Integrator {
    pub fn new(dt: f32, substeps: u32) -> Self {
        let substep_dt = dt / substeps.max(1) as f32;
        Self {
            dt: substep_dt,
            substeps: substeps.max(1),
            gravity: Vec3::new(0.0, -9.81, 0.0),
        }
    }

    pub fn integrate_position(&self, body: &mut RigidBody, dt: f32) {
        if body.is_static {
            return;
        }

        body.transform.position += body.velocity.linear * dt;

        let omega_mag = body.velocity.angular.length();
        if omega_mag > 1e-6 {
            let axis = body.velocity.angular / omega_mag;
            let angle = omega_mag * dt;
            let delta = Quat::from_axis_angle(axis, angle);
            body.transform.rotation = (delta * body.transform.rotation).normalize();
        }
    }

    pub fn integrate_velocity(&self, body: &mut RigidBody, dt: f32) {
        if body.is_static {
            return;
        }

        let gravity_force = self.gravity * body.mass_properties.mass * body.gravity_scale;
        body.apply_force(gravity_force);

        body.velocity.linear += body.acceleration * dt;

        body.velocity.linear *= (1.0 - body.linear_velocity_damping * dt).max(0.0);
        body.velocity.angular *= (1.0 - body.angular_velocity_damping * dt).max(0.0);

        body.acceleration = Vec3::ZERO;
    }

    pub fn step(&self, bodies: &mut Arena<RigidBody>) {
        for _ in 0..self.substeps {
            for body in bodies.iter_mut() {
                self.integrate_velocity(body, self.dt);
            }

            for body in bodies.iter_mut() {
                self.integrate_position(body, self.dt);
            }
        }
    }
}
