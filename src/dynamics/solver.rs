use glam::Vec3;

use crate::{
    core::rigidbody::RigidBody,
    utils::allocator::{Arena, EntityId},
};

/// Contact info shared between broad/narrow phase and solver.
#[derive(Debug, Clone)]
pub struct Contact {
    pub body_a: EntityId,
    pub body_b: EntityId,
    pub point: Vec3,
    pub normal: Vec3,
    pub depth: f32,
    pub relative_velocity: f32,
}

/// Basic constraint solver placeholder (Phase 2).
pub struct ConstraintSolver {
    pub iterations: u32,
    pub bias_factor: f32,
}

impl ConstraintSolver {
    pub fn new(iterations: u32) -> Self {
        Self {
            iterations,
            bias_factor: 0.2,
        }
    }

    pub fn solve(&self, bodies: &mut Arena<RigidBody>, contacts: &[Contact]) {
        for _ in 0..self.iterations {
            for contact in contacts {
                if let Some((body_a, body_b)) = bodies.get2_mut(contact.body_a, contact.body_b) {
                    Self::resolve_contact(body_a, body_b, contact, self.bias_factor);
                }
            }
        }
    }

    fn resolve_contact(
        body_a: &mut RigidBody,
        body_b: &mut RigidBody,
        contact: &Contact,
        bias_factor: f32,
    ) {
        if body_a.is_static && body_b.is_static {
            return;
        }

        let r_a = contact.point - body_a.transform.position;
        let r_b = contact.point - body_b.transform.position;

        let v_a = body_a.velocity.linear + body_a.velocity.angular.cross(r_a);
        let v_b = body_b.velocity.linear + body_b.velocity.angular.cross(r_b);
        let relative_vel = v_b - v_a;

        let vel_along_normal = relative_vel.dot(contact.normal);
        if vel_along_normal >= 0.0 {
            return;
        }

        let restitution = (body_a.material.restitution * body_b.material.restitution).sqrt();
        let bias = bias_factor * contact.depth.max(0.0) / (1.0 / 60.0);

        let impulse_mag = -(vel_along_normal + bias * restitution)
            / (body_a.inverse_mass + body_b.inverse_mass + 1e-6);

        let impulse = contact.normal * impulse_mag;

        body_a.apply_impulse(-impulse, contact.point);
        body_b.apply_impulse(impulse, contact.point);
    }
}

/// Phase 4 solver placeholder (PGS).
pub struct PGSSolver {
    pub velocity_iterations: u32,
    pub position_iterations: u32,
    pub bias_factor: f32,
    pub slop: f32,
}

impl Default for PGSSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl PGSSolver {
    pub fn new() -> Self {
        Self {
            velocity_iterations: 4,
            position_iterations: 1,
            bias_factor: 0.2,
            slop: 0.01,
        }
    }

    pub fn solve(
        &self,
        bodies: &mut Arena<RigidBody>,
        contacts: &[Contact],
    ) {
        let basic = ConstraintSolver::new(self.velocity_iterations);
        basic.solve(bodies, contacts);
        // Position iterations and joint constraints would be added in later phases.
    }
}
