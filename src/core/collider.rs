use super::types::Transform;
use crate::utils::allocator::EntityId;
use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};

/// Enumeration of supported collider geometries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ColliderShape {
    Sphere { radius: f32 },
    Box { half_extents: Vec3 },
    Capsule { radius: f32, height: f32 },
    Cylinder { radius: f32, height: f32 },
    ConvexHull { vertices: Vec<Vec3> },
    Compound { shapes: Vec<(Transform, ColliderShape)> },
}

/// Simple collision filtering mask.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CollisionFilter {
    pub layer: u32,
    pub mask: u32,
}

impl Default for CollisionFilter {
    fn default() -> Self {
        Self {
            layer: 1,
            mask: u32::MAX,
        }
    }
}

/// Collider component referencing a rigid body.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Collider {
    pub id: EntityId,
    pub rigidbody_id: EntityId,
    pub shape: ColliderShape,
    pub offset: Transform,
    pub is_trigger: bool,
    pub collision_filter: CollisionFilter,
}

impl Collider {
    pub fn sphere(radius: f32) -> ColliderShape {
        ColliderShape::Sphere { radius }
    }

    pub fn cuboid(half_extents: Vec3) -> ColliderShape {
        ColliderShape::Box { half_extents }
    }

    pub fn world_transform(&self, rigidbody_transform: &Transform) -> Transform {
        rigidbody_transform.combine(&self.offset)
    }
}

/// Convenience constructors for transforms.
impl Transform {
    pub fn from_position(position: Vec3) -> Self {
        Self {
            position,
            ..Self::default()
        }
    }

    pub fn from_position_rotation(position: Vec3, rotation: Quat) -> Self {
        Self {
            position,
            rotation,
            ..Self::default()
        }
    }
}
