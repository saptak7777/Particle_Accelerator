use glam::Vec3;

use crate::core::collider::ColliderShape;

/// Helper utilities for computing support points, bounding radii, etc.
pub struct ShapeUtil;

impl ShapeUtil {
    pub fn support(shape: &ColliderShape, direction: Vec3) -> Vec3 {
        match shape {
            ColliderShape::Sphere { radius } => direction.normalize_or_zero() * *radius,
            ColliderShape::Box { half_extents } => Vec3::new(
                if direction.x >= 0.0 { half_extents.x } else { -half_extents.x },
                if direction.y >= 0.0 { half_extents.y } else { -half_extents.y },
                if direction.z >= 0.0 { half_extents.z } else { -half_extents.z },
            ),
            ColliderShape::Capsule { radius, height } => {
                let half_height = height / 2.0;
                let mut point = direction.normalize_or_zero() * *radius;
                point.y += half_height * direction.y.signum();
                point
            }
            ColliderShape::Cylinder { radius, height } => {
                Vec3::new(
                    radius * direction.x.signum(),
                    (height / 2.0) * direction.y.signum(),
                    radius * direction.z.signum(),
                )
            }
            ColliderShape::ConvexHull { vertices } => {
                vertices
                    .iter()
                    .copied()
                    .max_by(|a, b| a.dot(direction).partial_cmp(&b.dot(direction)).unwrap())
                    .unwrap_or(Vec3::ZERO)
            }
            ColliderShape::Compound { shapes } => shapes
                .iter()
                .map(|(transform, sub_shape)| transform.position + Self::support(sub_shape, direction))
                .max_by(|a, b| a.dot(direction).partial_cmp(&b.dot(direction)).unwrap())
                .unwrap_or(Vec3::ZERO),
        }
    }

    pub fn bounding_radius(shape: &ColliderShape) -> f32 {
        match shape {
            ColliderShape::Sphere { radius } => *radius,
            ColliderShape::Box { half_extents } => half_extents.length(),
            ColliderShape::Capsule { radius, height } => (*radius * *radius + (height / 2.0).powi(2)).sqrt(),
            ColliderShape::Cylinder { radius, height } => (*radius * *radius + (height / 2.0).powi(2)).sqrt(),
            ColliderShape::ConvexHull { vertices } => vertices
                .iter()
                .map(|v| v.length())
                .fold(0.0, f32::max),
            ColliderShape::Compound { shapes } => shapes
                .iter()
                .map(|(transform, shape)| transform.position.length() + Self::bounding_radius(shape))
                .fold(0.0, f32::max),
        }
    }
}
