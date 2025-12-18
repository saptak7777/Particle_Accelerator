use glam::Vec3;

use crate::{
    core::{
        collider::{Collider, ColliderShape},
        soa::BodiesSoA,
        types::Transform,
    },
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

#[derive(Debug, Clone)]
pub struct RaycastQuery {
    pub origin: Vec3,
    pub direction: Vec3,
    pub max_distance: f32,
    pub layer_mask: u32,
    pub query_layer: u32,
    pub ignore_triggers: bool,
    pub closest_only: bool,
}

impl Default for RaycastQuery {
    fn default() -> Self {
        Self {
            origin: Vec3::ZERO,
            direction: Vec3::Z,
            max_distance: f32::INFINITY,
            layer_mask: u32::MAX,
            query_layer: 1,
            ignore_triggers: false,
            closest_only: true,
        }
    }
}

impl RaycastQuery {
    pub fn new(origin: Vec3, direction: Vec3, max_distance: f32) -> Self {
        Self {
            origin,
            direction,
            max_distance,
            query_layer: 1,
            ..Self::default()
        }
    }
}

pub struct Raycast;

impl Raycast {
    pub fn cast(
        query: &RaycastQuery,
        colliders: &Arena<Collider>,
        bodies: &BodiesSoA,
    ) -> Vec<RaycastHit> {
        Self::cast_with_filter(query, colliders, bodies, |_, _| true)
    }

    pub fn cast_with_filter<F>(
        query: &RaycastQuery,
        colliders: &Arena<Collider>,
        bodies: &BodiesSoA,
        mut filter: F,
    ) -> Vec<RaycastHit>
    where
        F: FnMut(EntityId, &Collider) -> bool,
    {
        let mut hits = Vec::new();

        for collider_id in colliders.ids() {
            let collider = match colliders.get(collider_id) {
                Some(c) => c,
                None => continue,
            };

            if query.ignore_triggers && collider.is_trigger {
                continue;
            }
            if collider.collision_filter.layer & query.layer_mask == 0 {
                continue;
            }
            if collider.collision_filter.mask & query.query_layer == 0 {
                continue;
            }
            if !filter(collider.id, collider) {
                continue;
            }

            let body = match bodies.get(collider.rigidbody_id) {
                Some(b) => b,
                None => continue,
            };
            let world_transform = collider.world_transform(body.transform());

            if let Some(hit) = Self::ray_shape_test(
                query,
                &collider.shape,
                &world_transform,
                collider.id,
                body.id(),
            ) {
                hits.push(hit);
            }
        }

        hits.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        if query.closest_only {
            hits.into_iter().take(1).collect()
        } else {
            hits
        }
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
                Self::ray_aabb(query, transform.position, *half_extents).map(
                    |(point, distance, normal)| RaycastHit {
                        body_id,
                        collider_id,
                        point,
                        normal,
                        distance,
                    },
                )
            }
            ColliderShape::Capsule { radius, height } => {
                Self::ray_capsule(query, transform.position, *radius, *height).map(
                    |(point, distance, normal)| RaycastHit {
                        body_id,
                        collider_id,
                        point,
                        normal,
                        distance,
                    },
                )
            }
            ColliderShape::Cylinder { radius, height } => {
                Self::ray_cylinder(query, transform.position, *radius, *height).map(
                    |(point, distance, normal)| RaycastHit {
                        body_id,
                        collider_id,
                        point,
                        normal,
                        distance,
                    },
                )
            }
            ColliderShape::Compound { shapes } => shapes
                .iter()
                .filter_map(|(local_transform, shape)| {
                    let combined = transform.combine(local_transform);
                    Self::ray_shape_test(query, shape, &combined, collider_id, body_id)
                })
                .min_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap()),
            ColliderShape::Mesh { mesh } => {
                Self::ray_mesh(query, mesh, transform, collider_id, body_id)
            }
            _ => None,
        }
    }

    fn ray_sphere(query: &RaycastQuery, center: Vec3, radius: f32) -> Option<(Vec3, f32)> {
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

    fn ray_capsule(
        query: &RaycastQuery,
        center: Vec3,
        radius: f32,
        height: f32,
    ) -> Option<(Vec3, f32, Vec3)> {
        let half_height = (height * 0.5).max(0.0);
        let base_cylinder_hit = Self::ray_cylinder(query, center, radius, height);

        let offsets = [Vec3::Y * half_height, -Vec3::Y * half_height];
        let mut best_hit = base_cylinder_hit;

        for offset in offsets {
            let sphere_center = center + offset;
            if let Some((point, distance)) = Self::ray_sphere(query, sphere_center, radius) {
                let normal = (point - sphere_center).normalize_or_zero();
                best_hit = Self::closer_hit(best_hit, Some((point, distance, normal)));
            }
        }

        best_hit
    }

    fn ray_mesh(
        query: &RaycastQuery,
        mesh: &crate::core::mesh::TriangleMesh,
        transform: &Transform,
        collider_id: EntityId,
        body_id: EntityId,
    ) -> Option<RaycastHit> {
        let dir = query.direction.normalize_or_zero();
        if dir == Vec3::ZERO {
            return None;
        }

        let matrix = transform.to_matrix();
        let mut best: Option<(Vec3, f32, Vec3)> = None;

        for tri in &mesh.indices {
            let v0 = matrix.transform_point3(mesh.vertices[tri[0] as usize]);
            let v1 = matrix.transform_point3(mesh.vertices[tri[1] as usize]);
            let v2 = matrix.transform_point3(mesh.vertices[tri[2] as usize]);

            if let Some((distance, normal)) = Self::ray_triangle(query.origin, dir, v0, v1, v2) {
                if distance <= query.max_distance {
                    let point = query.origin + dir * distance;
                    best = Self::closer_hit(best, Some((point, distance, normal)));
                }
            }
        }

        best.map(|(point, distance, normal)| RaycastHit {
            body_id,
            collider_id,
            point,
            normal,
            distance,
        })
    }

    fn ray_triangle(origin: Vec3, dir: Vec3, v0: Vec3, v1: Vec3, v2: Vec3) -> Option<(f32, Vec3)> {
        let edge1 = v1 - v0;
        let edge2 = v2 - v0;
        let pvec = dir.cross(edge2);
        let det = edge1.dot(pvec);
        if det.abs() < 1e-6 {
            return None;
        }
        let inv_det = 1.0 / det;
        let tvec = origin - v0;
        let u = tvec.dot(pvec) * inv_det;
        if !(0.0..=1.0).contains(&u) {
            return None;
        }
        let qvec = tvec.cross(edge1);
        let v = dir.dot(qvec) * inv_det;
        if v < 0.0 || u + v > 1.0 {
            return None;
        }
        let t = edge2.dot(qvec) * inv_det;
        if t < 0.0 {
            return None;
        }
        let normal = edge1.cross(edge2).normalize_or_zero();
        if normal == Vec3::ZERO {
            return None;
        }
        Some((t, normal))
    }

    fn ray_cylinder(
        query: &RaycastQuery,
        center: Vec3,
        radius: f32,
        height: f32,
    ) -> Option<(Vec3, f32, Vec3)> {
        let half_height = (height * 0.5).max(0.0);
        let dir = query.direction.normalize_or_zero();
        if dir == Vec3::ZERO {
            return None;
        }

        let origin = query.origin;
        let rel = origin - center;
        let mut best: Option<(Vec3, f32, Vec3)> = None;

        let a = dir.x * dir.x + dir.z * dir.z;
        if a.abs() > 1e-6 {
            let b = 2.0 * (rel.x * dir.x + rel.z * dir.z);
            let c = rel.x * rel.x + rel.z * rel.z - radius * radius;
            let disc = b * b - 4.0 * a * c;
            if disc >= 0.0 {
                let sqrt_disc = disc.sqrt();
                let mut t = (-b - sqrt_disc) / (2.0 * a);
                if t < 0.0 || t > query.max_distance {
                    t = (-b + sqrt_disc) / (2.0 * a);
                }
                if (0.0..=query.max_distance).contains(&t) {
                    let y = rel.y + dir.y * t;
                    if y >= -half_height && y <= half_height {
                        let point = origin + dir * t;
                        let radial = Vec3::new(point.x - center.x, 0.0, point.z - center.z)
                            .normalize_or_zero();
                        if radial != Vec3::ZERO {
                            best = Some((point, t, radial));
                        }
                    }
                }
            }
        }

        for cap in [-half_height, half_height] {
            if dir.y.abs() < 1e-6 {
                continue;
            }
            let plane_y = cap;
            let t = (plane_y - rel.y) / dir.y;
            if !(0.0..=query.max_distance).contains(&t) {
                continue;
            }
            let hit_point = origin + dir * t;
            let offset = hit_point - center;
            if offset.x * offset.x + offset.z * offset.z <= radius * radius + 1e-6 {
                let normal = Vec3::new(0.0, plane_y.signum(), 0.0);
                best = Self::closer_hit(best, Some((hit_point, t, normal)));
            }
        }

        best
    }

    fn closer_hit(
        current: Option<(Vec3, f32, Vec3)>,
        candidate: Option<(Vec3, f32, Vec3)>,
    ) -> Option<(Vec3, f32, Vec3)> {
        match (current, candidate) {
            (None, None) => None,
            (Some(hit), None) => Some(hit),
            (None, Some(hit)) => Some(hit),
            (Some(a), Some(b)) => {
                if a.1 <= b.1 {
                    Some(a)
                } else {
                    Some(b)
                }
            }
        }
    }
}
