#[cfg(test)]
use glam::Quat;
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

/// Gilbert-Johnson-Keerthi (GJK) collision test with EPA penetration depth.
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
                // GJK confirmed intersection, now compute penetration with EPA
                let (depth, mut normal) = EPAAlgorithm::compute_penetration(
                    &simplex,
                    shape_a,
                    transform_a,
                    shape_b,
                    transform_b,
                );

                // Enforce A→B convention: normal must point from A toward B
                let relative_pos = transform_b.position - transform_a.position;
                if normal.dot(relative_pos) < 0.0 {
                    normal = -normal;
                }

                // In degenerate or touching cases (depth ~ 0), EPA may return a normal perpendicular
                // to the actual approach axis due to simplex initialization.
                // If the centers are significantly separated, favor the center-to-center axis.
                let center_dist_sq = relative_pos.length_squared();
                if center_dist_sq > Self::EPSILON {
                    let center_dir = relative_pos / center_dist_sq.sqrt();
                    // If normal is nearly perpendicular to the approach direction (dot < 0.5)
                    // and depth is very small, we trust the center direction more for spheres
                    // and generally for centered impacts.
                    if normal.dot(center_dir) < 0.5 && depth < 0.01 {
                        normal = center_dir;
                    }
                }

                let contact_point =
                    Self::support(shape_a, transform_a, normal) - normal * depth * 0.5;

                let contact = Contact {
                    body_a,
                    body_b,
                    point: contact_point,
                    normal,
                    depth,
                    relative_velocity: 0.0,
                    feature_id: 0,
                    accumulated_normal_impulse: 0.0,
                    accumulated_tangent_impulse: Vec3::ZERO,
                    accumulated_rolling_impulse: Vec3::ZERO,
                    accumulated_torsional_impulse: 0.0,
                    material: MaterialPairProperties::default(),
                };

                return Some(contact);
            }
        }

        None
    }

    fn support(shape: &ColliderShape, transform: &Transform, direction: Vec3) -> Vec3 {
        match shape {
            ColliderShape::Sphere { radius } => {
                transform.position
                    + direction.normalize_or_zero() * (*radius * transform.scale.max_element())
            }
            ColliderShape::Box { half_extents } => {
                let dir_local = transform.rotation.conjugate() * direction;
                let local = Vec3::new(
                    if dir_local.x >= 0.0 {
                        half_extents.x
                    } else {
                        -half_extents.x
                    },
                    if dir_local.y >= 0.0 {
                        half_extents.y
                    } else {
                        -half_extents.y
                    },
                    if dir_local.z >= 0.0 {
                        half_extents.z
                    } else {
                        -half_extents.z
                    },
                ) * transform.scale;
                transform.position + transform.rotation * local
            }
            ColliderShape::Capsule { radius, height } => {
                let axis = Vec3::Y;
                let cap_offset = axis * 0.5 * height * transform.scale.y;
                let top = transform.position + transform.rotation * cap_offset;
                let bottom = transform.position - transform.rotation * cap_offset;
                let dir = direction.normalize_or_zero();
                let radial_scale = transform.scale.x.abs().max(transform.scale.z.abs());
                if dir.dot(transform.rotation * axis) >= 0.0 {
                    top + dir * (*radius * radial_scale)
                } else {
                    bottom + dir * (*radius * radial_scale)
                }
            }
            ColliderShape::Cylinder { radius, height } => {
                let axis = transform.rotation * Vec3::Y;
                let dir = direction.normalize_or_zero();
                let lateral = (dir - axis * dir.dot(axis)).normalize_or_zero();
                let radial_scale = transform.scale.x.abs().max(transform.scale.z.abs());
                let radial = lateral * (*radius * radial_scale);
                let axial = axis * (0.5 * height * transform.scale.y).copysign(dir.dot(axis));
                transform.position + radial + axial
            }
            ColliderShape::ConvexHull { vertices } => {
                let mut best_point = transform.position;
                let mut best_dot = f32::MIN;
                for v in vertices {
                    let world_v = transform.position + transform.rotation * (*v * transform.scale);
                    let dot = world_v.dot(direction);
                    if dot > best_dot {
                        best_dot = dot;
                        best_point = world_v;
                    }
                }
                best_point
            }
            ColliderShape::Mesh { mesh } => {
                let mut best_point = transform.position;
                let mut best_dot = f32::MIN;
                for v in &mesh.vertices {
                    let world_v = transform.position + transform.rotation * (*v * transform.scale);
                    let dot = world_v.dot(direction);
                    if dot > best_dot {
                        best_dot = dot;
                        best_point = world_v;
                    }
                }
                best_point
            }
            ColliderShape::Compound { shapes } => {
                let mut best_point = transform.position;
                let mut best_dot = f32::MIN;
                for (local_transform, shape) in shapes {
                    let child_world = transform.combine(local_transform);
                    let point = Self::support(shape, &child_world, direction);
                    let dot = point.dot(direction);
                    if dot > best_dot {
                        best_dot = dot;
                        best_point = point;
                    }
                }
                best_point
            }
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

                let dir = ab.cross(ao).cross(ab);
                if dir.length_squared() < Self::EPSILON {
                    // Origin is on the line AB. Pick a direction perpendicular to AB.
                    let axis = if ab.x.abs() < 0.1 { Vec3::X } else { Vec3::Y };
                    *direction = ab.cross(axis);
                } else {
                    *direction = dir;
                }
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
                    if abc.length_squared() < Self::EPSILON {
                        // Triangle is degenerate.
                        *direction = Vec3::Y; // arbitrary
                    } else if abc.dot(ao) > 0.0 {
                        *direction = abc;
                    } else {
                        *direction = -abc;
                    }
                    false
                }
            }
            4 => {
                let a = simplex[3];
                let b = simplex[2];
                let c = simplex[1];
                let d = simplex[0];
                let ab = b - a;
                let ac = c - a;
                let ad = d - a;
                let ao = -a;
                let abc = ab.cross(ac);
                let acd = ac.cross(ad);
                let adb = ad.cross(ab);

                if abc.dot(ao) > 0.0 {
                    simplex.remove(0);
                    *direction = abc;
                    false
                } else if acd.dot(ao) > 0.0 {
                    simplex.remove(2);
                    *direction = acd;
                    false
                } else if adb.dot(ao) > 0.0 {
                    simplex.remove(1);
                    *direction = adb;
                    false
                } else {
                    true
                }
            }
            _ => false,
        }
    }
}

/// Expanding Polytope Algorithm for penetration depth calculation.
struct EPAAlgorithm;

impl EPAAlgorithm {
    const MAX_ITERATIONS: usize = 32;
    const EPSILON: f32 = 1e-6;

    fn compute_penetration(
        simplex: &[Vec3],
        shape_a: &ColliderShape,
        transform_a: &Transform,
        shape_b: &ColliderShape,
        transform_b: &Transform,
    ) -> (f32, Vec3) {
        if simplex.len() < 4 {
            // Degenerate simplex, fall back to center-to-center
            let normal = (transform_b.position - transform_a.position).normalize_or_zero();
            let fallback_normal = if normal == Vec3::ZERO {
                Vec3::X
            } else {
                normal
            };

            // Estimate depth from simplex if we have points
            let depth = if !simplex.is_empty() {
                simplex
                    .iter()
                    .map(|p| p.length())
                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap_or(0.01)
            } else {
                0.01
            };

            return (depth, fallback_normal);
        }

        let mut polytope = vec![simplex[0], simplex[1], simplex[2], simplex[3]];
        let mut faces = Self::build_initial_faces(&polytope);

        for _ in 0..Self::MAX_ITERATIONS {
            let (_, min_dist, normal) = Self::find_closest_face(&polytope, &faces);

            // Check for valid result
            if min_dist >= f32::MAX * 0.5 {
                // No valid faces found, fall back
                let fallback_normal =
                    (transform_b.position - transform_a.position).normalize_or_zero();
                return (
                    0.01,
                    if fallback_normal == Vec3::ZERO {
                        Vec3::X
                    } else {
                        fallback_normal
                    },
                );
            }

            if min_dist < Self::EPSILON {
                return (Self::EPSILON, normal);
            }

            let support_a = GJKAlgorithm::support(shape_a, transform_a, normal);
            let support_b = GJKAlgorithm::support(shape_b, transform_b, -normal);
            let support = support_a - support_b;
            let distance = support.dot(normal);

            // Convergence check: if new support point doesn't expand polytope significantly
            if distance - min_dist < Self::EPSILON {
                return (min_dist, normal);
            }

            // Expand polytope by adding new support point
            Self::expand_polytope(&mut polytope, &mut faces, support);
        }

        // Max iterations reached, return best estimate
        let (_, min_dist, normal) = Self::find_closest_face(&polytope, &faces);
        if min_dist >= f32::MAX * 0.5 {
            let fallback_normal = (transform_b.position - transform_a.position).normalize_or_zero();
            (
                0.01,
                if fallback_normal == Vec3::ZERO {
                    Vec3::X
                } else {
                    fallback_normal
                },
            )
        } else {
            (min_dist, normal)
        }
    }

    fn build_initial_faces(polytope: &[Vec3]) -> Vec<(usize, usize, usize)> {
        let mut faces = vec![(0, 1, 2), (0, 2, 3), (0, 3, 1), (1, 3, 2)];

        // Ensure all normals point outwards from origin (which is inside)
        for face in &mut faces {
            let ab = polytope[face.1] - polytope[face.0];
            let ac = polytope[face.2] - polytope[face.0];
            let normal = ab.cross(ac);
            if polytope[face.0].dot(normal) < 0.0 {
                std::mem::swap(&mut face.1, &mut face.2);
            }
        }
        faces
    }

    fn find_closest_face(polytope: &[Vec3], faces: &[(usize, usize, usize)]) -> (usize, f32, Vec3) {
        let mut min_dist = f32::MAX;
        let mut min_normal = Vec3::ZERO;
        let mut min_idx = 0;

        for (idx, &(a, b, c)) in faces.iter().enumerate() {
            let ab = polytope[b] - polytope[a];
            let ac = polytope[c] - polytope[a];
            let normal = ab.cross(ac).normalize_or_zero();

            if normal == Vec3::ZERO {
                continue;
            }

            let dist = polytope[a].dot(normal);
            if dist < min_dist {
                min_dist = dist;
                min_normal = normal;
                min_idx = idx;
            }
        }

        (min_idx, min_dist, min_normal)
    }

    fn expand_polytope(
        polytope: &mut Vec<Vec3>,
        faces: &mut Vec<(usize, usize, usize)>,
        support: Vec3,
    ) {
        let new_idx = polytope.len();
        polytope.push(support);

        let mut edges = Vec::new();
        let mut i = 0;
        while i < faces.len() {
            let (a, b, c) = faces[i];
            let ab = polytope[b] - polytope[a];
            let ac = polytope[c] - polytope[a];
            let normal = ab.cross(ac).normalize_or_zero();

            if normal.dot(support - polytope[a]) > 0.0 {
                edges.push((a, b));
                edges.push((b, c));
                edges.push((c, a));
                faces.swap_remove(i);
            } else {
                i += 1;
            }
        }

        let mut boundary_edges = Vec::new();
        for (u, v) in edges {
            let mut found = false;
            for j in 0..boundary_edges.len() {
                if boundary_edges[j] == (v, u) {
                    boundary_edges.remove(j);
                    found = true;
                    break;
                }
            }
            if !found {
                boundary_edges.push((u, v));
            }
        }

        for (u, v) in boundary_edges {
            faces.push((u, v, new_idx));
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

        // Get local axes in world space
        let axes_a = [
            transform_a.rotation * Vec3::X,
            transform_a.rotation * Vec3::Y,
            transform_a.rotation * Vec3::Z,
        ];
        let axes_b = [
            transform_b.rotation * Vec3::X,
            transform_b.rotation * Vec3::Y,
            transform_b.rotation * Vec3::Z,
        ];

        let mut test_axes = Vec::with_capacity(15);
        test_axes.extend_from_slice(&axes_a);
        test_axes.extend_from_slice(&axes_b);

        for axis_a in &axes_a {
            for axis_b in &axes_b {
                let axis = axis_a.cross(*axis_b);
                if axis.length_squared() > 1e-6 {
                    test_axes.push(axis.normalize());
                }
            }
        }

        let mut min_overlap = f32::MAX;
        let mut min_axis = Vec3::ZERO;

        for axis in test_axes {
            let extent_a = (axes_a[0].dot(axis).abs() * half_extents_a.x)
                + (axes_a[1].dot(axis).abs() * half_extents_a.y)
                + (axes_a[2].dot(axis).abs() * half_extents_a.z);

            let extent_b = (axes_b[0].dot(axis).abs() * half_extents_b.x)
                + (axes_b[1].dot(axis).abs() * half_extents_b.y)
                + (axes_b[2].dot(axis).abs() * half_extents_b.z);

            let projection = relative_pos.dot(axis);
            let overlap = (extent_a + extent_b) - projection.abs();

            if overlap <= 0.0 {
                return None;
            }

            if overlap < min_overlap {
                min_overlap = overlap;
                min_axis = if projection < 0.0 { -axis } else { axis };
            }
        }

        let contact_point = {
            let axis_a = transform_a.rotation * min_axis.normalize_or_zero();
            let support_a = (transform_a.rotation.conjugate() * axis_a)
                .abs()
                .dot(half_extents_a);
            transform_a.position + axis_a * (support_a - min_overlap * 0.5)
        };

        Some(Contact {
            body_a,
            body_b,
            point: contact_point,
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

        let contact = match (&collider_a.shape, &collider_b.shape) {
            (
                ColliderShape::Box { half_extents: he_a },
                ColliderShape::Box { half_extents: he_b },
            ) => SATAlgorithm::intersect_boxes(
                *he_a,
                &transform_a,
                *he_b,
                &transform_b,
                body_a.id,
                body_b.id,
            ),
            _ => GJKAlgorithm::intersect(
                &collider_a.shape,
                &transform_a,
                &collider_b.shape,
                &transform_b,
                body_a.id,
                body_b.id,
            ),
        };

        let mut unwrapped_contact = contact?;
        unwrapped_contact.material =
            MaterialPairProperties::from_materials(&body_a.material, &body_b.material);
        Some(unwrapped_contact)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        core::{
            collider::{Collider, CollisionFilter},
            rigidbody::RigidBody,
        },
        utils::allocator::EntityId,
    };

    fn make_sphere_body(id: u32, radius: f32, position: Vec3) -> (RigidBody, Collider) {
        let mut body = RigidBody::new(EntityId::from_index(id));
        body.transform.position = position;

        let collider = Collider {
            id: EntityId::from_index(id + 100),
            rigidbody_id: body.id,
            shape: ColliderShape::Sphere { radius },
            offset: Transform::default(),
            is_trigger: false,
            collision_filter: CollisionFilter::default(),
        };

        (body, collider)
    }

    #[test]
    fn epa_computes_nonzero_depth_for_overlapping_spheres() {
        let (body_a, collider_a) = make_sphere_body(0, 1.0, Vec3::ZERO);
        let (body_b, collider_b) = make_sphere_body(1, 1.0, Vec3::new(1.5, 0.0, 0.0));

        let contact = NarrowPhase::collide(&collider_a, &body_a, &collider_b, &body_b)
            .expect("overlapping spheres should collide");

        // Depth should be approximately 0.5 (2.0 combined radius - 1.5 distance)
        assert!(
            contact.depth > 0.4 && contact.depth < 0.6,
            "depth was {}",
            contact.depth
        );
        assert!(contact.depth != 0.01, "depth should not be hardcoded");
    }

    #[test]
    fn epa_computes_correct_normal_for_overlapping_spheres() {
        let (body_a, collider_a) = make_sphere_body(0, 1.0, Vec3::ZERO);
        let (body_b, collider_b) = make_sphere_body(1, 1.0, Vec3::new(1.5, 0.0, 0.0));

        let contact = NarrowPhase::collide(&collider_a, &body_a, &collider_b, &body_b)
            .expect("overlapping spheres should collide");

        // Normal should point from A to B (along X axis)
        assert!(contact.normal.x > 0.9, "normal.x was {}", contact.normal.x);
        assert!(
            contact.normal.y.abs() < 0.1,
            "normal.y was {}",
            contact.normal.y
        );
        assert!(
            contact.normal.z.abs() < 0.1,
            "normal.z was {}",
            contact.normal.z
        );
    }

    #[test]
    fn epa_handles_deeply_penetrating_spheres() {
        let (body_a, collider_a) = make_sphere_body(0, 1.0, Vec3::ZERO);
        let (body_b, collider_b) = make_sphere_body(1, 1.0, Vec3::new(0.5, 0.0, 0.0));

        let contact = NarrowPhase::collide(&collider_a, &body_a, &collider_b, &body_b)
            .expect("deeply overlapping spheres should collide");

        // Depth should be approximately 1.5 (2.0 combined radius - 0.5 distance)
        assert!(
            contact.depth > 1.4 && contact.depth < 1.6,
            "depth was {}",
            contact.depth
        );
    }

    #[test]
    fn sat_computes_overlap_for_rotated_boxes() {
        let mut body_a = RigidBody::new(EntityId::from_index(0));
        body_a.transform.position = Vec3::ZERO;
        let collider_a = Collider {
            id: EntityId::from_index(100),
            rigidbody_id: body_a.id,
            shape: ColliderShape::Box {
                half_extents: Vec3::ONE,
            },
            offset: Transform::default(),
            is_trigger: false,
            collision_filter: CollisionFilter::default(),
        };

        let mut body_b = RigidBody::new(EntityId::from_index(1));
        // Place B such that it wouldn't overlap if AABB but overlaps when rotated
        // Axis-aligned, they are at 0 and 2.1. Gap of 0.1.
        body_b.transform.position = Vec3::new(2.1, 0.0, 0.0);
        let collider_b = Collider {
            id: EntityId::from_index(101),
            rigidbody_id: body_b.id,
            shape: ColliderShape::Box {
                half_extents: Vec3::ONE,
            },
            offset: Transform::default(),
            is_trigger: false,
            collision_filter: CollisionFilter::default(),
        };

        // Rotate A by 45 degrees around Z.
        // Its half-width along X becomes sqrt(2).
        body_a.transform.rotation = Quat::from_rotation_z(45.0f32.to_radians());

        let contact = NarrowPhase::collide(&collider_a, &body_a, &collider_b, &body_b)
            .expect("rotated boxes should collide");

        assert!(contact.depth > 0.0, "depth was {}", contact.depth);
        // Normal should be roughly along X
        assert!(contact.normal.x.abs() > 0.9);
    }

    #[test]
    fn gjk_computes_correct_contact_point_for_spheres() {
        let (body_a, collider_a) = make_sphere_body(0, 1.0, Vec3::ZERO);
        let (body_b, collider_b) = make_sphere_body(1, 1.0, Vec3::new(1.8, 0.0, 0.0));

        let contact = NarrowPhase::collide(&collider_a, &body_a, &collider_b, &body_b)
            .expect("spheres should collide");

        // Expected overlap depth: 2.0 - 1.8 = 0.2
        // Normal (A to B): (1, 0, 0)
        // Point on A surface along normal: (1, 0, 0)
        // Contact point = 1.0 - 0.5 * 0.2 = 0.9 along X
        assert!(
            contact.point.x > 0.8 && contact.point.x < 1.0,
            "point was {:?}",
            contact.point
        );
        assert!(contact.point.y.abs() < 1e-3);
        assert!(contact.point.z.abs() < 1e-3);
    }

    #[test]
    fn gjk_returns_none_for_separated_spheres() {
        let (body_a, collider_a) = make_sphere_body(0, 1.0, Vec3::ZERO);
        let (body_b, collider_b) = make_sphere_body(1, 1.0, Vec3::new(3.0, 0.0, 0.0));

        let contact = NarrowPhase::collide(&collider_a, &body_a, &collider_b, &body_b);
        assert!(contact.is_none(), "separated spheres should not collide");
    }
}
