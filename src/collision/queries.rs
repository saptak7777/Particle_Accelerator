use glam::Vec3;

use crate::{
    core::{collider::{Collider, ColliderShape}, rigidbody::RigidBody, types::Transform},
    utils::allocator::{Arena, EntityId},
};

/// Result of a ray cast against colliders.
#[derive(Debug, Clone)]
pub struct RaycastHit {
    pub body_id: EntityId,
    pub collider_id: EntityId,
    pub point: Vec3,
    pub normal: Vec3,
    pub distance: f32,
}

pub struct RaycastQuery {
    pub origin: Vec3,
    pub direction: Vec3,
    pub max_distance: f32,
}

pub struct Raycast;

impl Raycast {
    pub fn cast(
        query: &RaycastQuery,
        colliders: &Arena<Collider>,
        bodies: &Arena<RigidBody>,
    ) -> Vec<RaycastHit> {
        let mut hits = Vec::new();

        for collider_id in colliders.ids() {
            let collider = match colliders.get(collider_id) {
                Some(c) => c,
                None => continue,
            };
            let body = match bodies.get(collider.rigidbody_id) {
                Some(b) => b,
                None => continue,
            };
            let world_transform = collider.world_transform(&body.transform);

            if let Some(hit) = Self::ray_shape_test(
                query,
                &collider.shape,
                &world_transform,
                collider.id,
                body.id,
            ) {
                hits.push(hit);
            }
        }

        hits.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        hits
    }

    fn ray_shape_test(
        query: &RaycastQuery,
        shape: &ColliderShape,
        transform: &Transform,
        collider_id: EntityId,
        body_id: EntityId,
    ) -> Option<RaycastHit> {
        match shape {
            ColliderShape::Sphere { radius } => {
                Self::ray_sphere(query, transform.position, *radius).map(|(point, distance)| {
                    RaycastHit {
                        body_id,
                        collider_id,
                        point,
                        normal: (point - transform.position).normalize(),
                        distance,
                    }
                })
            }
            ColliderShape::Box { half_extents } => {
                Self::ray_aabb(query, transform.position, *half_extents).map(|(point, distance, normal)| {
                    RaycastHit {
                        body_id,
                        collider_id,
                        point,
                        normal,
                        distance,
                    }
                })
            }
            _ => None,
        }
    }

    fn ray_sphere(
        query: &RaycastQuery,
        center: Vec3,
        radius: f32,
    ) -> Option<(Vec3, f32)> {
        let oc = query.origin - center;
        let dir = query.direction.normalize_or_zero();
        let a = dir.length_squared();
        let b = 2.0 * oc.dot(dir);
        let c = oc.length_squared() - radius * radius;
        let discriminant = b * b - 4.0 * a * c;
        if discriminant < 0.0 {
            return None;
        }
        let sqrt_disc = discriminant.sqrt();
        let t = (-b - sqrt_disc) / (2.0 * a);
        if t < 0.0 || t > query.max_distance {
            return None;
        }
        let point = query.origin + dir * t;
        Some((point, t))
    }

    fn ray_aabb(
        query: &RaycastQuery,
        center: Vec3,
        half_extents: Vec3,
    ) -> Option<(Vec3, f32, Vec3)> {
        let dir = query.direction.normalize_or_zero();
        let mut t_min = 0.0;
        let mut t_max = query.max_distance;
        let mut normal = Vec3::ZERO;

        for i in 0..3 {
            let origin_component = query.origin[i];
            let dir_component = dir[i];
            let min = center[i] - half_extents[i];
            let max = center[i] + half_extents[i];

            if dir_component.abs() < 1e-6 {
                if origin_component < min || origin_component > max {
                    return None;
                }
            } else {
                let inv_dir = 1.0 / dir_component;
                let mut t1 = (min - origin_component) * inv_dir;
                let mut t2 = (max - origin_component) * inv_dir;
                let mut axis_normal = Vec3::ZERO;
                axis_normal[i] = -dir_component.signum();

                if t1 > t2 {
                    std::mem::swap(&mut t1, &mut t2);
                    axis_normal = -axis_normal;
                }

                if t1 > t_min {
                    t_min = t1;
                    normal = axis_normal;
                }

                t_max = t_max.min(t2);
                if t_min > t_max {
                    return None;
                }
            }
        }

        let point = query.origin + dir * t_min;
        Some((point, t_min, normal))
    }
}
