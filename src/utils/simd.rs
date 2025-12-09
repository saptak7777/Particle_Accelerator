//! SIMD-related helpers for batch math operations (stub for future expansion).

use glam::{Mat4, Vec3};

/// Placeholder implementation that falls back to scalar math.
pub fn batch_transform_points(points: &[Vec3], matrix: &Mat4) -> Vec<Vec3> {
    points.iter().map(|p| (*matrix * p.extend(1.0)).truncate()).collect()
}
