use glam::Vec3;

use crate::{core::soa::BodyMut, dynamics::solver::Contact};

/// Model for anisotropic friction (different coefficients along different axes).
#[derive(Debug, Clone, Copy)]
pub struct AnisotropicFrictionModel {
    pub friction_x: f32,
    pub friction_z: f32,
}

impl Default for AnisotropicFrictionModel {
    fn default() -> Self {
        Self {
            friction_x: 0.5,
            friction_z: 0.5,
        }
    }
}

/// Applies friction to the contacting bodies.
/// Includes support for Coulomb friction and rolling/torsional resistance.
pub fn apply_friction(
    body_a: &mut BodyMut,
    body_b: &mut BodyMut,
    contact: &mut Contact,
    normal_impulse: f32,
) {
    apply_tangential_friction(body_a, body_b, contact, normal_impulse);
    apply_rolling_friction(body_a, body_b, contact, normal_impulse);
    apply_torsional_friction(body_a, body_b, contact, normal_impulse);
}

fn apply_tangential_friction(
    body_a: &mut BodyMut,
    body_b: &mut BodyMut,
    contact: &mut Contact,
    normal_impulse: f32,
) {
    if body_a.is_static() && body_b.is_static() {
        return;
    }

    let normal_impulse = normal_impulse.max(0.0);
    if normal_impulse <= f32::EPSILON {
        contact.accumulated_tangent_impulse = Vec3::ZERO;
        return;
    }

    let r_a = contact.point - body_a.transform.position;
    let r_b = contact.point - body_b.transform.position;

    let v_a = body_a.velocity.linear + body_a.velocity.angular.cross(r_a);
    let v_b = body_b.velocity.linear + body_b.velocity.angular.cross(r_b);
    let relative_vel = v_b - v_a;

    let tangent_velocity = relative_vel - contact.normal * relative_vel.dot(contact.normal);

    let inv_mass_sum = *body_a.inverse_mass + *body_b.inverse_mass + 1e-6;
    let mut new_impulse = contact.accumulated_tangent_impulse - tangent_velocity / inv_mass_sum;

    // Remove any numerical drift along the normal axis.
    new_impulse -= contact.normal * new_impulse.dot(contact.normal);

    let tangent_dir = pick_tangent_direction(new_impulse, tangent_velocity, contact.normal);

    let mu_static = friction_coefficient(
        contact.material.static_friction,
        contact.normal,
        tangent_dir,
    );
    let mut mu_dynamic = friction_coefficient(
        contact.material.dynamic_friction,
        contact.normal,
        tangent_dir,
    );

    if mu_dynamic > mu_static {
        mu_dynamic = mu_static;
    }

    let max_static = mu_static * normal_impulse;
    let max_dynamic = mu_dynamic * normal_impulse;

    let mut clamped_impulse = new_impulse;
    let length = clamped_impulse.length();

    if length > max_static && length > 0.0 {
        if max_dynamic > 0.0 {
            clamped_impulse = clamped_impulse.normalize() * max_dynamic;
        } else {
            clamped_impulse = Vec3::ZERO;
        }
    }

    let impulse_delta = clamped_impulse - contact.accumulated_tangent_impulse;
    if impulse_delta.length_squared() <= 1e-12 {
        contact.accumulated_tangent_impulse = clamped_impulse;
        return;
    }

    contact.accumulated_tangent_impulse = clamped_impulse;
    body_a.apply_impulse(-impulse_delta, contact.point);
    body_b.apply_impulse(impulse_delta, contact.point);
}

fn apply_rolling_friction(
    body_a: &mut BodyMut,
    body_b: &mut BodyMut,
    contact: &mut Contact,
    normal_impulse: f32,
) {
    let limit = contact.material.rolling_friction.max(0.0) * normal_impulse.max(0.0);
    if limit <= f32::EPSILON {
        contact.accumulated_rolling_impulse = Vec3::ZERO;
        return;
    }

    let relative_ang = body_b.velocity.angular - body_a.velocity.angular;
    let rolling_axis = relative_ang - contact.normal * relative_ang.dot(contact.normal);
    let axis = rolling_axis.normalize_or_zero();
    if axis == Vec3::ZERO {
        return;
    }

    let eff_mass = axis.dot((*body_a.inverse_inertia) * axis)
        + axis.dot((*body_b.inverse_inertia) * axis)
        + 1e-6;
    let lambda = -axis.dot(relative_ang) / eff_mass;
    let desired = contact.accumulated_rolling_impulse + axis * lambda;
    let clamped = if desired.length() > limit {
        desired.normalize() * limit
    } else {
        desired
    };

    let delta = clamped - contact.accumulated_rolling_impulse;
    if delta.length_squared() <= 1e-12 {
        return;
    }

    contact.accumulated_rolling_impulse = clamped;
    // apply_angular_impulse not on BodyMut yet, use manual
    body_a.velocity.angular += (*body_a.inverse_inertia) * (-delta);
    body_b.velocity.angular += (*body_b.inverse_inertia) * delta;
}

fn apply_torsional_friction(
    body_a: &mut BodyMut,
    body_b: &mut BodyMut,
    contact: &mut Contact,
    normal_impulse: f32,
) {
    let limit = contact.material.torsional_friction.max(0.0) * normal_impulse.max(0.0);
    if limit <= f32::EPSILON {
        contact.accumulated_torsional_impulse = 0.0;
        return;
    }

    let axis = contact.normal.normalize_or_zero();
    if axis == Vec3::ZERO {
        return;
    }

    let relative_twist = (body_b.velocity.angular - body_a.velocity.angular).dot(axis);
    let eff_mass = axis.dot((*body_a.inverse_inertia) * axis)
        + axis.dot((*body_b.inverse_inertia) * axis)
        + 1e-6;
    let lambda = -relative_twist / eff_mass;
    let desired = (contact.accumulated_torsional_impulse + lambda).clamp(-limit, limit);
    let delta = desired - contact.accumulated_torsional_impulse;
    if delta.abs() <= 1e-10 {
        return;
    }
    contact.accumulated_torsional_impulse = desired;

    let angular_impulse = axis * delta;
    body_a.velocity.angular += (*body_a.inverse_inertia) * (-angular_impulse);
    body_b.velocity.angular += (*body_b.inverse_inertia) * angular_impulse;
}

// Helper functions

fn friction_coefficient(coeff: Vec3, normal: Vec3, tangent_dir: Vec3) -> f32 {
    let tangent = (tangent_dir - normal * tangent_dir.dot(normal)).normalize_or_zero();
    if tangent == Vec3::ZERO {
        coeff.abs().max_element()
    } else {
        coeff.abs().dot(tangent.abs()).max(0.0)
    }
}

fn pick_tangent_direction(candidate: Vec3, fallback: Vec3, normal: Vec3) -> Vec3 {
    let projected_candidate = project_onto_tangent(candidate, normal);
    if projected_candidate.length_squared() > 1e-12 {
        return projected_candidate.normalize();
    }

    let projected_fallback = project_onto_tangent(fallback, normal);
    if projected_fallback.length_squared() > 1e-12 {
        return projected_fallback.normalize();
    }

    orthogonal_to_normal(normal)
}

fn project_onto_tangent(vector: Vec3, normal: Vec3) -> Vec3 {
    vector - normal * vector.dot(normal)
}

fn orthogonal_to_normal(normal: Vec3) -> Vec3 {
    let mut tangent = normal.cross(Vec3::X);
    if tangent.length_squared() <= 1e-6 {
        tangent = normal.cross(Vec3::Y);
    }
    tangent.normalize_or_zero()
}
