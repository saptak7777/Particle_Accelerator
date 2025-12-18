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
#[serde(default)]
pub struct Material {
    pub density: f32,
    pub restitution: f32,
    pub static_friction: f32,
    pub dynamic_friction: f32,
    /// Resistance to rolling (symmetric for now, specialized models can extend later).
    pub rolling_friction: f32,
    /// Resistance to twisting at the contact patch.
    pub torsional_friction: f32,
    /// Per-axis scaling for tangential friction (1.0 = isotropic).
    pub friction_anisotropy: Vec3,
    /// How this material mixes its coefficients with another material.
    pub mixing: MaterialMixing,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            density: 1.0,
            restitution: 0.1,
            static_friction: 0.5,
            dynamic_friction: 0.3,
            rolling_friction: 0.02,
            torsional_friction: 0.01,
            friction_anisotropy: Vec3::ONE,
            mixing: MaterialMixing::default(),
        }
    }
}

impl Material {
    pub fn rubber() -> Self {
        Self {
            density: 1.4,
            restitution: 0.8,
            static_friction: 1.2,
            dynamic_friction: 1.0,
            rolling_friction: 0.04,
            torsional_friction: 0.03,
            friction_anisotropy: Vec3::ONE,
            mixing: MaterialMixing::default(),
        }
    }

    pub fn steel() -> Self {
        Self {
            density: 7.8,
            restitution: 0.4,
            static_friction: 0.58,
            dynamic_friction: 0.44,
            rolling_friction: 0.015,
            torsional_friction: 0.012,
            friction_anisotropy: Vec3::splat(0.95),
            mixing: MaterialMixing::default(),
        }
    }

    pub fn ice() -> Self {
        Self {
            density: 0.9,
            restitution: 0.05,
            static_friction: 0.05,
            dynamic_friction: 0.03,
            rolling_friction: 0.005,
            torsional_friction: 0.003,
            friction_anisotropy: Vec3::splat(0.8),
            mixing: MaterialMixing::default(),
        }
    }

    pub fn combine_with(&self, other: &Self) -> MaterialPairProperties {
        let friction_mode = self.mixing.friction.resolve(other.mixing.friction);
        let restitution_mode = self.mixing.restitution.resolve(other.mixing.restitution);

        let static_scalar = friction_mode.combine(self.static_friction, other.static_friction);
        let dynamic_scalar = friction_mode.combine(self.dynamic_friction, other.dynamic_friction);
        let rolling = friction_mode.combine(self.rolling_friction, other.rolling_friction);
        let torsional = friction_mode.combine(self.torsional_friction, other.torsional_friction);
        let restitution = restitution_mode.combine(self.restitution, other.restitution);

        let anisotropy = (self.friction_anisotropy + other.friction_anisotropy) * 0.5;

        MaterialPairProperties {
            static_friction: anisotropy * Vec3::splat(static_scalar),
            dynamic_friction: anisotropy * Vec3::splat(dynamic_scalar),
            rolling_friction: rolling,
            torsional_friction: torsional,
            restitution,
        }
    }

    pub fn combine_pair(a: &Self, b: &Self) -> MaterialPairProperties {
        let ab = a.combine_with(b);
        let ba = b.combine_with(a);
        MaterialPairProperties {
            static_friction: (ab.static_friction + ba.static_friction) * 0.5,
            dynamic_friction: (ab.dynamic_friction + ba.dynamic_friction) * 0.5,
            rolling_friction: 0.5 * (ab.rolling_friction + ba.rolling_friction),
            torsional_friction: 0.5 * (ab.torsional_friction + ba.torsional_friction),
            restitution: 0.5 * (ab.restitution + ba.restitution),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MaterialMixing {
    pub friction: MixingMode,
    pub restitution: MixingMode,
}

impl Default for MaterialMixing {
    fn default() -> Self {
        Self {
            friction: MixingMode::Average,
            restitution: MixingMode::Average,
        }
    }
}

impl MaterialMixing {
    pub fn with_friction(mut self, mode: MixingMode) -> Self {
        self.friction = mode;
        self
    }

    pub fn with_restitution(mut self, mode: MixingMode) -> Self {
        self.restitution = mode;
        self
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub enum MixingMode {
    #[default]
    Average,
    Min,
    Max,
    GeometricMean,
}

impl MixingMode {
    fn combine(self, a: f32, b: f32) -> f32 {
        match self {
            MixingMode::Average => 0.5 * (a + b),
            MixingMode::Min => a.min(b),
            MixingMode::Max => a.max(b),
            MixingMode::GeometricMean => (a.abs() * b.abs()).sqrt().copysign(0.5 * (a.signum() + b.signum())),
        }
    }

    fn resolve(self, other: MixingMode) -> MixingMode {
        if matches!(self, MixingMode::Average) {
            other
        } else {
            self
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MaterialPairProperties {
    pub static_friction: Vec3,
    pub dynamic_friction: Vec3,
    pub rolling_friction: f32,
    pub torsional_friction: f32,
    pub restitution: f32,
}

impl Default for MaterialPairProperties {
    fn default() -> Self {
        MaterialPairProperties::from_materials(&Material::default(), &Material::default())
    }
}

impl MaterialPairProperties {
    pub fn from_materials(a: &Material, b: &Material) -> Self {
        Material::combine_pair(a, b)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixing_modes_combine_expected_values() {
        let mode = MixingMode::Average;
        assert!((mode.combine(0.6, 0.2) - 0.4).abs() < 1e-5);

        let mode = MixingMode::Min;
        assert!((mode.combine(0.6, 0.2) - 0.2).abs() < 1e-5);

        let mode = MixingMode::Max;
        assert!((mode.combine(0.6, 0.2) - 0.6).abs() < 1e-5);

        let mode = MixingMode::GeometricMean;
        let expected = (0.6_f32 * 0.2_f32).sqrt();
        assert!((mode.combine(0.6, 0.2) - expected).abs() < 1e-5);
    }

    #[test]
    fn material_pair_properties_reflect_anisotropy() {
        let mut mat_a = Material::default();
        mat_a.static_friction = 0.8;
        mat_a.dynamic_friction = 0.6;
        mat_a.friction_anisotropy = Vec3::new(1.0, 0.8, 1.2);

        let mut mat_b = Material::default();
        mat_b.static_friction = 0.4;
        mat_b.dynamic_friction = 0.2;
        mat_b.friction_anisotropy = Vec3::splat(1.0);

        let pair = mat_a.combine_with(&mat_b);

        assert!(pair.static_friction.x > pair.static_friction.y);
        assert!(pair.dynamic_friction.z > pair.dynamic_friction.y);
        assert!(pair.restitution > 0.0);
    }
}
