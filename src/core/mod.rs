//! Core types describing physics entities, components, and shared data.

pub mod types;
pub mod rigidbody;
pub mod collider;
pub mod constraints;
pub mod mesh;

pub use types::{MassProperties, Material, Transform, Velocity};
pub use rigidbody::RigidBody;
pub use collider::{Collider, ColliderShape, CollisionFilter};
pub use constraints::Joint;
pub use mesh::{TriangleMesh, MeshBuilder, Aabb, MeshBvh};
