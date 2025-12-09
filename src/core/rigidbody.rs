use crate::utils::allocator::EntityId;

use super::types::{MassProperties, Material, Transform, Velocity};
use glam::{Mat3, Vec3};

/// Core rigid body description storing kinematic state and properties.
#[derive(Debug, Clone)]
pub struct RigidBody {
    pub id: EntityId,
    pub transform: Transform,
    pub velocity: Velocity,
    pub acceleration: Vec3,
    pub mass_properties: MassProperties,
    pub material: Material,
    pub gravity_scale: f32,
    pub is_static: bool,
    pub is_kinematic: bool,
    pub is_awake: bool,
    pub is_enabled: bool,
    pub linear_velocity_damping: f32,
    pub angular_velocity_damping: f32,
    pub inverse_mass: f32,
    pub inverse_inertia: Mat3,
}

impl Default for RigidBody {
    fn default() -> Self {
        let mut body = Self {
            id: EntityId::default(),
            transform: Transform::default(),
            velocity: Velocity::default(),
            acceleration: Vec3::ZERO,
            mass_properties: MassProperties::default(),
            material: Material::default(),
            gravity_scale: 1.0,
            is_static: false,
            is_kinematic: false,
            is_awake: true,
            is_enabled: true,
            linear_velocity_damping: 0.02,
            angular_velocity_damping: 0.02,
            inverse_mass: 1.0,
            inverse_inertia: Mat3::IDENTITY,
        };
        body.recompute_inverses();
        body
    }
}

impl RigidBody {
    pub fn new(id: EntityId) -> Self {
        Self {
            id,
            ..Self::default()
        }
    }

    pub fn set_velocity(&mut self, linear: Vec3, angular: Vec3) {
        self.velocity.linear = linear;
        self.velocity.angular = angular;
    }

    pub fn apply_force(&mut self, force: Vec3) {
        if self.is_static {
            return;
        }
        self.acceleration += force * self.inverse_mass;
    }

    pub fn apply_impulse(&mut self, impulse: Vec3, position: Vec3) {
        if self.is_static {
            return;
        }

        self.velocity.linear += impulse * self.inverse_mass;
        let torque = (position - self.transform.position).cross(impulse);
        self.velocity.angular += self.inverse_inertia * torque;
        self.is_awake = true;
    }

    pub fn set_mass_properties(&mut self, props: MassProperties) {
        self.mass_properties = props;
        self.recompute_inverses();
    }

    fn recompute_inverses(&mut self) {
        self.inverse_mass = if self.mass_properties.mass.abs() < f32::EPSILON {
            0.0
        } else {
            1.0 / self.mass_properties.mass
        };
        let inverse_inertia = self.mass_properties.inertia.inverse();
        if inverse_inertia.determinant().abs() < f32::EPSILON {
            self.inverse_inertia = Mat3::IDENTITY;
        } else {
            self.inverse_inertia = inverse_inertia;
        }
    }
}
