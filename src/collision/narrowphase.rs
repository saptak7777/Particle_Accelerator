use glam::Vec3;

use crate::{
    core::{
        collider::{Collider, ColliderShape},
        rigidbody::RigidBody,
        types::{MaterialPairProperties, Transform},
    },
    dynamics::solver::Contact,
    utils::allocator::EntityId,
};

/// Gilbert-Johnson-Keerthi (GJK) collision test (simplified placeholder).
pub struct GJKAlgorithm;

impl GJKAlgorithm {
    const MAX_ITERATIONS: usize = 20;
    const EPSILON: f32 = 1e-6;

    pub fn intersect(
        shape_a: &ColliderShape,
        transform_a: &Transform,
        shape_b: &ColliderShape,
        transform_b: &Transform,
        body_a: EntityId,
        body_b: EntityId,
    ) -> Option<Contact> {
        let mut simplex: Vec<Vec3> = Vec::new();
        let mut direction = transform_b.position - transform_a.position;
        if direction.length_squared() < Self::EPSILON {
            direction = Vec3::X;
        }

        for _ in 0..Self::MAX_ITERATIONS {
            let support_a = Self::support(shape_a, transform_a, direction);
            let support_b = Self::support(shape_b, transform_b, -direction);
            let point = support_a - support_b;

            if point.dot(direction) < 0.0 {
                return None;
            }

            simplex.push(point);
            if Self::contains_origin(&mut simplex, &mut direction) {
                return Some(Contact {
                    body_a,
                    body_b,
                    point: transform_a.position,
                    normal: direction.normalize_or_zero(),
                    depth: 0.01,
                    relative_velocity: 0.0,
                    feature_id: 0,
                    accumulated_normal_impulse: 0.0,
                    accumulated_tangent_impulse: Vec3::ZERO,
                    accumulated_rolling_impulse: Vec3::ZERO,
                    accumulated_torsional_impulse: 0.0,
                    material: MaterialPairProperties::default(),
                });
            }
        }

        None
    }

    fn support(shape: &ColliderShape, transform: &Transform, direction: Vec3) -> Vec3 {
        match shape {
            ColliderShape::Sphere { radius } => {
                transform.position + direction.normalize_or_zero() * *radius
            }
            ColliderShape::Box { half_extents } => {
                let local = Vec3::new(
                    if direction.x >= 0.0 { half_extents.x } else { -half_extents.x },
                    if direction.y >= 0.0 { half_extents.y } else { -half_extents.y },
                    if direction.z >= 0.0 { half_extents.z } else { -half_extents.z },
                );
                transform.position + transform.rotation * local
            }
            _ => transform.position,
        }
    }

    fn contains_origin(simplex: &mut Vec<Vec3>, direction: &mut Vec3) -> bool {
        match simplex.len() {
            1 => {
                *direction = -simplex[0];
                false
            }
            2 => {
                let a = simplex[1];
                let b = simplex[0];
                let ab = b - a;
                let ao = -a;
                *direction = ab.cross(ao).cross(ab);
                false
            }
            3 => {
                let a = simplex[2];
                let b = simplex[1];
                let c = simplex[0];
                let ab = b - a;
                let ac = c - a;
                let ao = -a;
                let abc = ab.cross(ac);
                if abc.cross(ac).dot(ao) > 0.0 {
                    simplex.remove(1);
                    *direction = ac.cross(ao).cross(ac);
                    false
                } else if ab.cross(abc).dot(ao) > 0.0 {
                    simplex.remove(0);
                    *direction = ab.cross(ao).cross(ab);
                    false
                } else {
                    if abc.dot(ao) > 0.0 {
                        *direction = abc;
                    } else {
                        *direction = -abc;
                    }
                    false
                }
            }
            4 => true,
            _ => false,
        }
    }
}

/// Separating axis theorem for box-box collisions (AABB placeholder).
pub struct SATAlgorithm;

impl SATAlgorithm {
    pub fn intersect_boxes(
        half_extents_a: Vec3,
        transform_a: &Transform,
        half_extents_b: Vec3,
        transform_b: &Transform,
        body_a: EntityId,
        body_b: EntityId,
    ) -> Option<Contact> {
        let relative_pos = transform_b.position - transform_a.position;
        let axes = [
            Vec3::X, Vec3::Y, Vec3::Z,
        ];

        let mut min_overlap = f32::MAX;
        let mut min_axis = Vec3::X;

        for axis in axes {
            let extent_a = half_extents_a.x * axis.x.abs()
                + half_extents_a.y * axis.y.abs()
                + half_extents_a.z * axis.z.abs();
            let extent_b = half_extents_b.x * axis.x.abs()
                + half_extents_b.y * axis.y.abs()
                + half_extents_b.z * axis.z.abs();

            let projection = relative_pos.dot(axis);
            let overlap = (extent_a + extent_b) - projection.abs();

            if overlap < 0.0 {
                return None;
            }

            if overlap < min_overlap {
                min_overlap = overlap;
                min_axis = if projection < 0.0 { -axis } else { axis };
            }
        }

        Some(Contact {
            body_a,
            body_b,
            point: transform_a.position,
            normal: min_axis.normalize_or_zero(),
            depth: min_overlap,
            relative_velocity: 0.0,
            feature_id: 0,
            accumulated_normal_impulse: 0.0,
            accumulated_tangent_impulse: Vec3::ZERO,
            accumulated_rolling_impulse: Vec3::ZERO,
            accumulated_torsional_impulse: 0.0,
            material: MaterialPairProperties::default(),
        })
    }
}

/// Narrow phase dispatcher (placeholder hooking algorithms to colliders).
pub struct NarrowPhase;

impl NarrowPhase {
    pub fn collide(
        collider_a: &Collider,
        body_a: &RigidBody,
        collider_b: &Collider,
        body_b: &RigidBody,
    ) -> Option<Contact> {
        let transform_a = collider_a.world_transform(&body_a.transform);
        let transform_b = collider_b.world_transform(&body_b.transform);

        let mut contact = match (&collider_a.shape, &collider_b.shape) {
            (ColliderShape::Box { half_extents: he_a }, ColliderShape::Box { half_extents: he_b }) => {
                SATAlgorithm::intersect_boxes(*he_a, &transform_a, *he_b, &transform_b, body_a.id, body_b.id)
            }
            _ => GJKAlgorithm::intersect(
                &collider_a.shape,
                &transform_a,
                &collider_b.shape,
                &transform_b,
                body_a.id,
                body_b.id,
            ),
        }?;

        contact.material = MaterialPairProperties::from_materials(&body_a.material, &body_b.material);
        Some(contact)
    }
}
