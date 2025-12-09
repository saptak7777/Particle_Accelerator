//! Collision detection modules: broad-phase, narrow-phase, contact manifolds, queries, CCD.

pub mod shapes;
pub mod broadphase;
pub mod narrowphase;
pub mod contact;
pub mod queries;
pub mod ccd;

pub use broadphase::{BroadPhase, SpatialGrid};
pub use contact::{ContactManifold, ContactSolverInput};
pub use queries::{Raycast, RaycastHit, RaycastQuery};
pub use ccd::CCDDetector;
