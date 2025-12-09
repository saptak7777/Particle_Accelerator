use glam::{Mat3, Mat4, Quat, Vec3};
use serde::{Deserialize, Serialize};

/// Common math types re-exported for convenience.
pub use glam::{Mat2, Vec2};

/// Position, orientation, and non-uniform scale of an entity.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Transform {
    pub position: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        }
    }
}

impl Transform {
    /// Builds a homogeneous matrix representation of the transform.
    pub fn to_matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.position)
    }

    /// Applies another transform on top of this one, returning the composition.
    pub fn combine(&self, other: &Transform) -> Transform {
        Transform {
            position: self.position + self.rotation * (self.scale * other.position),
            rotation: (self.rotation * other.rotation).normalize(),
            scale: self.scale * other.scale,
        }
    }
}

/// Linear and angular velocity of a rigid body.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Velocity {
    pub linear: Vec3,
    pub angular: Vec3,
}

impl Default for Velocity {
    fn default() -> Self {
        Self {
            linear: Vec3::ZERO,
            angular: Vec3::ZERO,
        }
    }
}

/// Mass and inertia tensor data.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MassProperties {
    pub mass: f32,
    pub inertia: Mat3,
}

impl Default for MassProperties {
    fn default() -> Self {
        Self {
            mass: 1.0,
            inertia: Mat3::IDENTITY,
        }
    }
}

/// Material coefficients that affect interactions.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Material {
    pub density: f32,
    pub restitution: f32,
    pub static_friction: f32,
    pub dynamic_friction: f32,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            density: 1.0,
            restitution: 0.1,
            static_friction: 0.5,
            dynamic_friction: 0.3,
        }
    }
}

/// Helper methods for inertia calculations.
pub trait InertiaTensorExt {
    fn for_solid_box(half_extents: Vec3, mass: f32) -> Mat3;
    fn for_solid_sphere(radius: f32, mass: f32) -> Mat3;
}

impl InertiaTensorExt for Mat3 {
    fn for_solid_box(half_extents: Vec3, mass: f32) -> Mat3 {
        let lx = half_extents.x * 2.0;
        let ly = half_extents.y * 2.0;
        let lz = half_extents.z * 2.0;
        let factor = mass / 12.0;
        Mat3::from_diagonal(Vec3::new(
            factor * (ly * ly + lz * lz),
            factor * (lx * lx + lz * lz),
            factor * (lx * lx + ly * ly),
        ))
    }

    fn for_solid_sphere(radius: f32, mass: f32) -> Mat3 {
        let value = 0.4 * mass * radius * radius;
        Mat3::from_diagonal(Vec3::splat(value))
    }
}
