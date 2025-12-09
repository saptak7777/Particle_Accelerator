//! Additional math helpers layered on top of `glam`.

use glam::{Mat3, Quat, Vec3};

/// Converts angular velocity vector (radians/sec) into a quaternion delta.
pub fn angular_velocity_to_quat(angular: Vec3, dt: f32) -> Quat {
    let angle = angular.length() * dt;
    if angle.abs() < 1e-6 {
        return Quat::IDENTITY;
    }
    let axis = angular.normalize();
    Quat::from_axis_angle(axis, angle)
}

/// Builds an inertia tensor for a solid capsule aligned along Y.
pub fn inertia_capsule(radius: f32, height: f32, mass: f32) -> Mat3 {
    let cylinder_mass = mass * 0.6;
    let sphere_mass = (mass - cylinder_mass) / 2.0;

    let cylinder_inertia = Mat3::from_diagonal(Vec3::new(
        (1.0 / 12.0) * cylinder_mass * (3.0 * radius * radius + height * height),
        0.5 * cylinder_mass * radius * radius,
        (1.0 / 12.0) * cylinder_mass * (3.0 * radius * radius + height * height),
    ));

    let sphere_inertia = Mat3::from_diagonal(Vec3::splat(0.4 * sphere_mass * radius * radius));

    cylinder_inertia + sphere_inertia
}
