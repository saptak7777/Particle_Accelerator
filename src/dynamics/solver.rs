use std::collections::HashMap;

use glam::Vec3;

use crate::{
    core::{constraints::Joint, rigidbody::RigidBody},
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
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
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
        joints: &[Joint],
    ) {
        for _ in 0..self.velocity_iterations {
            for contact in contacts {
                if let Some((body_a, body_b)) = bodies.get2_mut(contact.body_a, contact.body_b) {
                    ConstraintSolver::resolve_contact(body_a, body_b, contact, self.bias_factor);
                }
            }

            for joint in joints {
                self.resolve_velocity_joint(bodies, joint);
            }
        }

        for _ in 0..self.position_iterations {
            for contact in contacts {
                if let Some((body_a, body_b)) = bodies.get2_mut(contact.body_a, contact.body_b) {
                    Self::correct_position(body_a, body_b, contact, self.bias_factor, self.slop);
                }
            }
        }
    }

    /// Parallel-friendly solver path operating on a dense slice of rigid bodies.
    pub fn solve_island_slice(
        &self,
        bodies: &mut [RigidBody],
        id_map: &HashMap<EntityId, usize>,
        contacts: &[Contact],
        joints: &[Joint],
    ) {
        for _ in 0..self.velocity_iterations {
            for contact in contacts {
                if let Some((body_a, body_b)) =
                    get_pair_mut_from_slice(bodies, id_map, contact.body_a, contact.body_b)
                {
                    ConstraintSolver::resolve_contact(body_a, body_b, contact, self.bias_factor);
                }
            }

            for joint in joints {
                resolve_velocity_joint_slice(bodies, id_map, joint, self.bias_factor);
            }
        }

        for _ in 0..self.position_iterations {
            for contact in contacts {
                if let Some((body_a, body_b)) =
                    get_pair_mut_from_slice(bodies, id_map, contact.body_a, contact.body_b)
                {
                    Self::correct_position(body_a, body_b, contact, self.bias_factor, self.slop);
                }
            }
        }
    }

    fn correct_position(
        body_a: &mut RigidBody,
        body_b: &mut RigidBody,
        contact: &Contact,
        bias_factor: f32,
        slop: f32,
    ) {
        if body_a.is_static && body_b.is_static {
            return;
        }

        let correction = (contact.depth - slop).max(0.0) * bias_factor;
        let total_inv_mass = body_a.inverse_mass + body_b.inverse_mass;
        if total_inv_mass <= 1e-6 {
            return;
        }

        let impulse = contact.normal * (correction / total_inv_mass);

        if !body_a.is_static {
            body_a.transform.position -= impulse * body_a.inverse_mass;
        }
        if !body_b.is_static {
            body_b.transform.position += impulse * body_b.inverse_mass;
        }
    }

    fn resolve_velocity_joint(&self, bodies: &mut Arena<RigidBody>, joint: &Joint) {
        match joint {
            Joint::Distance {
                body_a,
                body_b,
                distance,
            } => {
                if let Some((a, b)) = bodies.get2_mut(*body_a, *body_b) {
                    Self::enforce_distance_joint(a, b, *distance, self.bias_factor);
                }
            }
            Joint::Spring {
                body_a,
                body_b,
                rest_length,
                stiffness,
                damping,
            } => {
                if let Some((a, b)) = bodies.get2_mut(*body_a, *body_b) {
                    Self::apply_spring_forces(
                        a,
                        b,
                        *rest_length,
                        *stiffness,
                        *damping,
                        self.bias_factor,
                    );
                }
            }
            Joint::Fixed { .. } | Joint::Revolute { .. } => {
                // Future: implement rotation-preserving constraints.
            }
        }
    }

    fn enforce_distance_joint(
        body_a: &mut RigidBody,
        body_b: &mut RigidBody,
        target_distance: f32,
        bias: f32,
    ) {
        if body_a.is_static && body_b.is_static {
            return;
        }

        let delta = body_b.transform.position - body_a.transform.position;
        let current = delta.length();
        if current <= f32::EPSILON {
            return;
        }

        let direction = delta / current;
        let error = current - target_distance;
        if error.abs() <= f32::EPSILON {
            return;
        }

        let total_inv_mass = body_a.inverse_mass + body_b.inverse_mass;
        if total_inv_mass <= f32::EPSILON {
            return;
        }

        let impulse_mag = -(error * bias) / total_inv_mass;
        let impulse = direction * impulse_mag;

        if !body_a.is_static {
            body_a.velocity.linear += impulse * body_a.inverse_mass;
        }
        if !body_b.is_static {
            body_b.velocity.linear -= impulse * body_b.inverse_mass;
        }
    }

    fn apply_spring_forces(
        body_a: &mut RigidBody,
        body_b: &mut RigidBody,
        rest_length: f32,
        stiffness: f32,
        damping: f32,
        bias: f32,
    ) {
        if body_a.is_static && body_b.is_static {
            return;
        }

        let delta = body_b.transform.position - body_a.transform.position;
        let current_length = delta.length();
        if current_length <= f32::EPSILON {
            return;
        }

        let direction = delta / current_length;
        let extension = current_length - rest_length;
        let relative_velocity =
            (body_b.velocity.linear - body_a.velocity.linear).dot(direction);
        let force_mag = -stiffness * extension - damping * relative_velocity;
        let impulse = direction * force_mag * bias;

        if !body_a.is_static {
            body_a.velocity.linear += impulse * body_a.inverse_mass;
        }
        if !body_b.is_static {
            body_b.velocity.linear -= impulse * body_b.inverse_mass;
        }
    }
}

fn resolve_velocity_joint_slice(
    bodies: &mut [RigidBody],
    id_map: &HashMap<EntityId, usize>,
    joint: &Joint,
    bias: f32,
) {
    match joint {
        Joint::Distance {
            body_a,
            body_b,
            distance,
        } => {
            if let Some((a, b)) =
                get_pair_mut_from_slice(bodies, id_map, *body_a, *body_b)
            {
                PGSSolver::enforce_distance_joint(a, b, *distance, bias);
            }
        }
        Joint::Spring {
            body_a,
            body_b,
            rest_length,
            stiffness,
            damping,
        } => {
            if let Some((a, b)) =
                get_pair_mut_from_slice(bodies, id_map, *body_a, *body_b)
            {
                PGSSolver::apply_spring_forces(
                    a,
                    b,
                    *rest_length,
                    *stiffness,
                    *damping,
                    bias,
                );
            }
        }
        Joint::Fixed { .. } | Joint::Revolute { .. } => {}
    }
}

fn get_pair_mut_from_slice<'a>(
    bodies: &'a mut [RigidBody],
    id_map: &HashMap<EntityId, usize>,
    a: EntityId,
    b: EntityId,
) -> Option<(&'a mut RigidBody, &'a mut RigidBody)> {
    let idx_a = *id_map.get(&a)?;
    let idx_b = *id_map.get(&b)?;
    if idx_a == idx_b {
        return None;
    }

    if idx_a < idx_b {
        let (left, right) = bodies.split_at_mut(idx_b);
        Some((&mut left[idx_a], &mut right[0]))
    } else {
        let (left, right) = bodies.split_at_mut(idx_a);
        Some((&mut right[0], &mut left[idx_b]))
    }
}
