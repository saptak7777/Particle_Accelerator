//! Core types describing physics entities, components, and shared data.

pub mod articulations;
pub mod collider;
pub mod constraints;
pub mod mesh;
pub mod rigidbody;
pub mod soa;
pub mod types;

pub use articulations::{JointType as ArticulatedJointType, Link, Multibody};
pub use collider::{Collider, ColliderShape, CollisionFilter};
pub use constraints::Joint;
pub use mesh::{Aabb, MeshBuilder, MeshBvh, TriangleMesh};
pub use rigidbody::RigidBody;
pub use types::{MassProperties, Material, Transform, Velocity};
