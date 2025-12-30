use super::{mesh::TriangleMesh, types::Transform};
use crate::utils::allocator::EntityId;
use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};

/// Enumeration of supported collider geometries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ColliderShape {
    Sphere {
        radius: f32,
    },
    Box {
        half_extents: Vec3,
    },
    Capsule {
        radius: f32,
        height: f32,
    },
    Cylinder {
        radius: f32,
        height: f32,
    },
    ConvexHull {
        vertices: Vec<Vec3>,
    },
    Compound {
        shapes: Vec<(Transform, ColliderShape)>,
    },
    Mesh {
        mesh: TriangleMesh,
    },
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

impl Default for Collider {
    fn default() -> Self {
        Self::builder().build()
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

    pub fn mesh(vertices: Vec<Vec3>, indices: Vec<[u32; 3]>) -> ColliderShape {
        ColliderShape::Mesh {
            mesh: TriangleMesh::builder(vertices, indices).build(),
        }
    }

    pub fn world_transform(&self, rigidbody_transform: &Transform) -> Transform {
        rigidbody_transform.combine(&self.offset)
    }

    pub fn bounding_radius(&self) -> f32 {
        self.shape.bounding_radius()
    }

    pub fn builder() -> ColliderBuilder {
        ColliderBuilder::new()
    }
}

pub struct ColliderBuilder {
    shape: ColliderShape,
    offset: Transform,
    is_trigger: bool,
    filter: CollisionFilter,
}

impl Default for ColliderBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ColliderBuilder {
    pub fn new() -> Self {
        Self {
            shape: ColliderShape::Sphere { radius: 1.0 },
            offset: Transform::default(),
            is_trigger: false,
            filter: CollisionFilter::default(),
        }
    }

    pub fn sphere(mut self, radius: f32) -> Self {
        self.shape = ColliderShape::Sphere { radius };
        self
    }

    pub fn box_shape(mut self, half_extents: Vec3) -> Self {
        self.shape = ColliderShape::Box { half_extents };
        self
    }

    pub fn capsule(mut self, radius: f32, height: f32) -> Self {
        self.shape = ColliderShape::Capsule { radius, height };
        self
    }

    pub fn offset(mut self, offset: Transform) -> Self {
        self.offset = offset;
        self
    }

    pub fn is_trigger(mut self, is_trigger: bool) -> Self {
        self.is_trigger = is_trigger;
        self
    }

    pub fn filter(mut self, layer: u32, mask: u32) -> Self {
        self.filter = CollisionFilter { layer, mask };
        self
    }

    pub fn build(self) -> Collider {
        Collider {
            id: EntityId::default(),
            rigidbody_id: EntityId::default(),
            shape: self.shape,
            offset: self.offset,
            is_trigger: self.is_trigger,
            collision_filter: self.filter,
        }
    }
}

impl ColliderShape {
    pub fn bounding_radius(&self) -> f32 {
        match self {
            ColliderShape::Sphere { radius } => *radius,
            ColliderShape::Box { half_extents } => half_extents.length(),
            ColliderShape::Capsule { radius, height } => radius + height * 0.5,
            ColliderShape::Cylinder { radius, height } => {
                (radius.powi(2) + (height * 0.5).powi(2)).sqrt()
            }
            ColliderShape::ConvexHull { vertices } => {
                vertices.iter().map(|v| v.length()).fold(0.0, f32::max)
            }
            ColliderShape::Compound { shapes } => shapes
                .iter()
                .map(|(transform, shape)| transform.position.length() + shape.bounding_radius())
                .fold(0.0, f32::max),
            ColliderShape::Mesh { mesh } => mesh.bounding_radius(),
        }
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
