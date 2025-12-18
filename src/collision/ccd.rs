use glam::{Quat, Vec3};

use crate::{
    collision::narrowphase::NarrowPhase,
    core::{
        collider::{Collider, ColliderShape},
        rigidbody::RigidBody,
        types::{MaterialPairProperties, Transform},
    },
    dynamics::solver::Contact,
    utils::simd,
};

/// Result of a continuous collision query.
#[derive(Debug, Clone)]
pub struct CCDResult {
    pub contact: Contact,
    pub time_of_impact: f32,
}

/// Continuous collision detector placeholder.
pub struct CCDDetector {
    pub enabled: bool,
    pub ccd_threshold: f32,
    pub angular_padding: f32,
    pub max_toi_iterations: usize,
    pub speculative_margin: f32,
}

impl Default for CCDDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl CCDDetector {
    pub fn new() -> Self {
        Self {
            enabled: true,
            ccd_threshold: 10.0,
            angular_padding: 0.5,
            max_toi_iterations: 8,
            speculative_margin: 0.05,
        }
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    pub fn set_ccd_threshold(&mut self, threshold: f32) {
        self.ccd_threshold = threshold.max(0.0);
    }

    pub fn set_angular_padding(&mut self, padding: f32) {
        self.angular_padding = padding.max(0.0);
    }

    pub fn detect_ccd(
        &self,
        body_a: &RigidBody,
        collider_a: &Collider,
        body_b: &RigidBody,
        collider_b: &Collider,
        dt: f32,
    ) -> Option<CCDResult> {
        if !self.enabled {
            return None;
        }

        let (relative_velocity, relative_speed) =
            relative_velocity_and_speed(body_a, body_b, collider_a, collider_b, dt);

        relative_velocity_and_speed(body_a, body_b, collider_a, collider_b, dt);

        if relative_speed < self.ccd_threshold {
            return None;
        }

        // Rewind bodies to start of frame for CCD check
        let mut start_a = body_a.clone();
        start_a.transform.position -= start_a.velocity.linear * dt;

        let mut start_b = body_b.clone();
        start_b.transform.position -= start_b.velocity.linear * dt;

        let toi = self.compute_time_of_impact(
            body_a,
            collider_a,
            body_b,
            collider_b,
            dt,
            relative_velocity,
        )?;

        // println!("DEBUG: CCD Hit Detected! TOI: {:.4}", toi);

        // Use fallback contact logic which derives normal from approximate shape logic.
        // This is necessary because NarrowPhase (GJK) is currently a stub and returns incorrect normals.
        let contact = self.build_fallback_contact(
            body_a,
            collider_a,
            body_b,
            collider_b,
            toi,
            relative_velocity,
        );
        // println!("DEBUG: CCD Contact Normal: {:?}", contact.normal);

        Some(CCDResult {
            contact,
            time_of_impact: toi,
        })
    }

    pub fn generate_speculative_contact(
        &self,
        body_a: &RigidBody,
        collider_a: &Collider,
        body_b: &RigidBody,
        collider_b: &Collider,
        dt: f32,
    ) -> Option<Contact> {
        if self.speculative_margin <= f32::EPSILON {
            return None;
        }

        // Project positions to end of frame for speculative check
        let predicted_a = body_a.transform.position + body_a.velocity.linear * dt;
        let predicted_b = body_b.transform.position + body_b.velocity.linear * dt;
        let direction = (predicted_b - predicted_a).normalize_or_zero();
        if direction == Vec3::ZERO {
            return None;
        }

        let support_a = support_radius(collider_a, body_a, direction);
        let support_b = support_radius(collider_b, body_b, -direction);
        let distance = (predicted_b - predicted_a).length();

        // Penetration is SumRadii - Distance.
        // If Distance > SumRadii, Penetration is negative (Gap).
        // If Distance < SumRadii, Penetration is positive (Overlap).
        // Speculative margin is included: a contact is generated if the gap is less than the margin.
        // Gap = Distance - SumRadii.
        // Condition: Gap < Margin => Distance - SumRadii < Margin => SumRadii - Distance > -Margin.
        // penetration > -Margin.

        let penetration = support_a + support_b - distance;

        if penetration < -self.speculative_margin {
            return None;
        }

        // The depth passed to the solver must represent the current separation (negative),
        // not the predicted penetration at the end of the frame. Providing a predicted
        // penetration value would cause the solver to apply a corrective impulse
        // prematurely, resulting in non-physical bouncing before contact occurs.

        // The predicted relative direction is used as the contact normal, as it
        // represents the primary axis of impact. Depth is calculated along this
        // normal using the support points of both colliders at their current positions.

        let point = predicted_a + direction * support_a;

        // Recalculate CURRENT depth
        let current_distance_along_normal =
            (body_b.transform.position - body_a.transform.position).dot(direction);
        let current_depth = support_a + support_b - current_distance_along_normal;

        Some(Contact {
            body_a: body_a.id,
            body_b: body_b.id,
            point,
            normal: direction,
            depth: current_depth,
            relative_velocity: (body_b.velocity.linear - body_a.velocity.linear).dot(direction),
            feature_id: 0,
            accumulated_normal_impulse: 0.0,
            accumulated_tangent_impulse: Vec3::ZERO,
            accumulated_rolling_impulse: Vec3::ZERO,
            accumulated_torsional_impulse: 0.0,
            material: MaterialPairProperties::from_materials(&body_a.material, &body_b.material),
        })
    }

    fn compute_time_of_impact(
        &self,
        body_a: &RigidBody,
        collider_a: &Collider,
        body_b: &RigidBody,
        collider_b: &Collider,
        dt: f32,
        relative_velocity: Vec3,
    ) -> Option<f32> {
        if !self.enabled || self.max_toi_iterations == 0 {
            return None;
        }

        // 1. Solve bounding sphere collision times
        let motion_dir = relative_velocity.normalize_or_zero();

        if motion_dir == Vec3::ZERO {
            return None;
        }

        let base_radius = support_radius(collider_a, body_a, motion_dir)
            + support_radius(collider_b, body_b, -motion_dir);
        // Add angular expansion safety margin
        let angular_expand = self.angular_padding
            * dt
            * (body_a.velocity.angular.length() * collider_a.bounding_radius()
                + body_b.velocity.angular.length() * collider_b.bounding_radius());
        let combined_radius = base_radius + angular_expand;

        let relative_position = body_b.transform.position - body_a.transform.position;
        let a = relative_velocity.length_squared();
        if a <= f32::EPSILON {
            return None; // No relative motion
        }
        let b = 2.0 * relative_position.dot(relative_velocity);
        let c = relative_position.length_squared() - combined_radius.powi(2);

        let discriminant = b * b - 4.0 * a * c;
        if discriminant < 0.0 {
            return None; // Miss
        }

        let sqrt_disc = discriminant.sqrt();
        let t0 = (-b - sqrt_disc) / (2.0 * a);
        let t1 = (-b + sqrt_disc) / (2.0 * a);

        /*
        println!(
            "DEBUG: TOI Quadratic: a={a:.2} b={b:.2} c={c:.2} disc={discriminant:.2} t0={t0:.6} t1={t1:.6} dt={dt:.6}"
        );
        */

        // Clamp to current frame
        let t_in = t0.clamp(0.0, dt);
        let t_out = t1.clamp(0.0, dt);

        if t_in >= t_out {
            return None; // No overlap interval in this frame
        }

        // 2. Check for collision at critical points (Entry and Midpoint)
        // The midpoint is checked initially as it represents the deepest sphere penetration.
        let t_mid = (t_in + t_out) * 0.5;

        // Contact at t_in indicates an existing overlap; TOI is set to t_in.
        if self
            .sample_contact(body_a, collider_a, body_b, collider_b, t_in)
            .is_some()
        {
            return Some(t_in);
        }

        // Absence of collision at t_mid suggesting a miss or insufficient penetration for the interval.
        // The bracket [t_in, t_mid] is established when a collision is detected at t_mid.
        if self
            .sample_contact(body_a, collider_a, body_b, collider_b, t_mid)
            .is_some()
        {
            // Binary Search Refinement in [t_in, t_mid]
            // println!("DEBUG: Collision found at t_mid ({:.4}). Bisecting.", t_mid);
            return Some(self.bisect_toi(body_a, collider_a, body_b, collider_b, t_in, t_mid));
        }

        if self
            .sample_contact(body_a, collider_a, body_b, collider_b, t_out)
            .is_some()
        {
            return Some(self.bisect_toi(body_a, collider_a, body_b, collider_b, t_mid, t_out));
        }

        None
    }

    fn bisect_toi(
        &self,
        body_a: &RigidBody,
        collider_a: &Collider,
        body_b: &RigidBody,
        collider_b: &Collider,
        min_t: f32,
        max_t: f32,
    ) -> f32 {
        let mut lo = min_t;
        let mut hi = max_t;

        for _ in 0..self.max_toi_iterations {
            let mid = (lo + hi) * 0.5;
            if self
                .sample_contact(body_a, collider_a, body_b, collider_b, mid)
                .is_some()
            {
                // Colliding at mid, so impact was earlier
                hi = mid;
            } else {
                // Midpoint is safe; the impact occurs in the latter half of the interval.
                lo = mid;
            }
        }
        lo
    }

    fn sample_contact(
        &self,
        body_a: &RigidBody,
        collider_a: &Collider,
        body_b: &RigidBody,
        collider_b: &Collider,
        time: f32,
    ) -> Option<Contact> {
        let sample_a = integrate_body_state(body_a, time);
        let sample_b = integrate_body_state(body_b, time);
        NarrowPhase::collide(collider_a, &sample_a, collider_b, &sample_b)
    }

    fn build_fallback_contact(
        &self,
        body_a: &RigidBody,
        collider_a: &Collider,
        body_b: &RigidBody,
        collider_b: &Collider,
        toi: f32,
        relative_velocity: Vec3,
    ) -> Contact {
        let sample_a = integrate_body_state(body_a, toi);
        let sample_b = integrate_body_state(body_b, toi);

        let mut normal = self.approximate_normal(body_a, collider_a, body_b, collider_b);
        if normal == Vec3::ZERO {
            normal = Vec3::Z;
        }

        // println!("DEBUG: CCD Fallback Normal: {:?}", normal);

        let point_a = support_point_world(collider_a, &sample_a, normal);
        let point_b = support_point_world(collider_b, &sample_b, -normal);
        let depth = (point_a - point_b).dot(normal).max(0.0);
        let contact_point = point_a;

        Contact {
            body_a: body_a.id,
            body_b: body_b.id,
            point: contact_point,
            normal,
            depth,
            relative_velocity: relative_velocity.dot(normal),
            feature_id: 0,
            accumulated_normal_impulse: 0.0,
            accumulated_tangent_impulse: Vec3::ZERO,
            accumulated_rolling_impulse: Vec3::ZERO,
            accumulated_torsional_impulse: 0.0,
            material: MaterialPairProperties::from_materials(&body_a.material, &body_b.material),
        }
    }

    fn approximate_normal(
        &self,
        body_a: &RigidBody,
        collider_a: &Collider,
        body_b: &RigidBody,
        collider_b: &Collider,
    ) -> Vec3 {
        // Try to be smarter than center-to-center for Box shapes
        // If B is a Box/Wall, find closest point on B to A's center
        if let ColliderShape::Box { half_extents } = &collider_b.shape {
            let rel_pos = body_a.transform.position - body_b.transform.position;
            // Rotate into B's local space (conjugate rotation)
            let local_pos = body_b.transform.rotation.conjugate() * rel_pos;
            let extents = *half_extents * body_b.transform.scale.abs();

            // Trivial normal for strictly external points.
            let clamped = local_pos.clamp(-extents, extents);
            let diff = local_pos - clamped;

            if diff.length_squared() > 1e-6 {
                // The difference vector points from SurfaceB to CenterA (B->A). Orientation is adjusted to A->B.
                return -(body_b.transform.rotation * diff).normalize_or_zero();
            } else {
                // Inside or surface. Find axis of min penetration (closest face)
                let mut min_dist = f32::MAX;
                let mut best_axis = Vec3::Z;
                let axes = [Vec3::X, Vec3::Y, Vec3::Z];
                let coords = [local_pos.x, local_pos.y, local_pos.z];
                let dims = [extents.x, extents.y, extents.z];

                for i in 0..3 {
                    let dist_pos = dims[i] - coords[i];
                    let dist_neg = coords[i] + dims[i];
                    if dist_pos < min_dist {
                        min_dist = dist_pos;
                        best_axis = axes[i];
                    }
                    if dist_neg < min_dist {
                        min_dist = dist_neg;
                        best_axis = -axes[i];
                    }
                }
                return -(body_b.transform.rotation * best_axis);
            }
        }

        // Symmetric check for A being Box
        if let ColliderShape::Box { half_extents } = &collider_a.shape {
            let rel_pos = body_b.transform.position - body_a.transform.position;
            let local_pos = body_a.transform.rotation.conjugate() * rel_pos;
            let extents = *half_extents * body_a.transform.scale.abs();

            let clamped = local_pos.clamp(-extents, extents);
            let diff = local_pos - clamped;
            if diff.length_squared() > 1e-6 {
                return (body_a.transform.rotation * diff).normalize_or_zero();
            } else {
                let mut min_dist = f32::MAX;
                let mut best_axis = Vec3::Z;
                let axes = [Vec3::X, Vec3::Y, Vec3::Z];
                let coords = [local_pos.x, local_pos.y, local_pos.z];
                let dims = [extents.x, extents.y, extents.z];

                for i in 0..3 {
                    let dist_pos = dims[i] - coords[i];
                    let dist_neg = coords[i] + dims[i];
                    if dist_pos < min_dist {
                        min_dist = dist_pos;
                        best_axis = axes[i];
                    }
                    if dist_neg < min_dist {
                        min_dist = dist_neg;
                        best_axis = -axes[i];
                    }
                }
                // Normal orientation is adjusted to point B->A.
                // Logic above finds direction to surface.
                // Using rel_pos = B - A. Local pos of B in A.
                // Clamped is on A's surface. Diff = PosB - SurfaceA. Points Outwards from A towards B.
                // So Normal is A->B.
                // Orientation is negated to align with B->A.
                return -(body_a.transform.rotation * best_axis);
            }
        }

        // Fallback to center-center
        (body_b.transform.position - body_a.transform.position).normalize_or_zero()
    }
}
#[cfg(test)]
mod tests {
    use super::CCDDetector;
    use crate::{
        core::{
            collider::{Collider, ColliderShape, CollisionFilter},
            rigidbody::RigidBody,
            types::Transform,
        },
        utils::allocator::EntityId,
    };
    use glam::Vec3;

    fn make_sphere(id: u32, radius: f32, position: Vec3, velocity: Vec3) -> (RigidBody, Collider) {
        let mut body = RigidBody::new(EntityId::from_index(id));
        body.transform.position = position;
        body.velocity.linear = velocity;

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
    fn detects_swept_sphere_hit() {
        let detector = CCDDetector::new();
        let (body_a, collider_a) = make_sphere(0, 0.5, Vec3::ZERO, Vec3::ZERO);
        let (mut body_b, collider_b) =
            make_sphere(1, 0.5, Vec3::new(0.0, 0.0, 5.0), Vec3::new(0.0, 0.0, -20.0));
        body_b.is_static = false;

        let hit = detector
            .detect_ccd(&body_a, &collider_a, &body_b, &collider_b, 0.5)
            .expect("swept spheres should collide");

        assert!(hit.contact.point.z > 0.0);
        assert!(hit.contact.point.z < 5.0);
        assert!(hit.time_of_impact < 0.5);
    }

    #[test]
    fn handles_compound_offset_geometry() {
        let detector = CCDDetector::new();
        let mut body_a = RigidBody::new(EntityId::from_index(10));
        body_a.transform.position = Vec3::ZERO;
        let compound = Collider {
            id: EntityId::from_index(11),
            rigidbody_id: body_a.id,
            shape: ColliderShape::Compound {
                shapes: vec![(
                    Transform::from_position(Vec3::new(0.0, 0.0, 1.0)),
                    ColliderShape::Sphere { radius: 0.5 },
                )],
            },
            offset: Transform::default(),
            is_trigger: false,
            collision_filter: CollisionFilter::default(),
        };

        let (mut body_b, collider_b) = make_sphere(
            2,
            0.25,
            Vec3::new(0.0, 0.0, 5.0),
            Vec3::new(0.0, 0.0, -30.0),
        );
        body_b.is_static = false;

        let hit = detector
            .detect_ccd(&body_a, &compound, &body_b, &collider_b, 0.2)
            .expect("compound offset sphere should be detected");

        assert!(
            hit.contact.point.z > 0.5,
            "contact point should respect compound offset"
        );
        assert!(hit.time_of_impact < 0.2);
    }
}

fn support_radius(collider: &Collider, body: &RigidBody, direction: Vec3) -> f32 {
    let dir = direction.normalize_or_zero();
    if dir == Vec3::ZERO {
        return 0.0;
    }
    let world = collider.world_transform(&body.transform);
    shape_support_radius(&collider.shape, dir, &world)
}

fn support_point_world(collider: &Collider, body: &RigidBody, direction: Vec3) -> Vec3 {
    let dir = direction.normalize_or_zero();
    if dir == Vec3::ZERO {
        return collider.world_transform(&body.transform).position;
    }
    let world = collider.world_transform(&body.transform);
    shape_support_point(&collider.shape, dir, &world)
}

fn shape_support_radius(shape: &ColliderShape, dir_world: Vec3, world: &Transform) -> f32 {
    let dir_local = world.rotation.conjugate() * dir_world;
    let dir_local = dir_local.normalize_or_zero();
    if dir_local == Vec3::ZERO {
        return 0.0;
    }
    match shape {
        ColliderShape::Sphere { radius } => radius.max(0.0) * max_scale(world.scale),
        ColliderShape::Box { half_extents } => {
            let extents = *half_extents * world.scale.abs();
            dir_local.abs().dot(extents)
        }
        ColliderShape::Capsule { radius, height } => {
            let half_height = 0.5 * height * world.scale.y.abs();
            let radial_scale = radial_scale(world.scale);
            radius.max(0.0) * radial_scale + half_height * dir_local.y.abs()
        }
        ColliderShape::Cylinder { radius, height } => {
            let half_height = 0.5 * height * world.scale.y.abs();
            let radial_scale = radial_scale(world.scale);
            let lateral = (dir_local.x * dir_local.x + dir_local.z * dir_local.z).sqrt();
            radius.max(0.0) * radial_scale * lateral + half_height * dir_local.y.abs()
        }
        ColliderShape::ConvexHull { vertices } => {
            let mut scaled: Vec<Vec3> = Vec::with_capacity(vertices.len());
            scaled.extend(vertices.iter().map(|vertex| (*vertex) * world.scale));
            simd::max_dot(&scaled, dir_local)
        }
        ColliderShape::Mesh { mesh } => {
            let mut scaled: Vec<Vec3> = Vec::with_capacity(mesh.vertices.len());
            scaled.extend(mesh.vertices.iter().map(|vertex| (*vertex) * world.scale));
            simd::max_dot(&scaled, dir_local)
        }
        ColliderShape::Compound { shapes } => {
            let mut max_proj = 0.0f32;
            for (local_transform, shape) in shapes {
                let child_world = world.combine(local_transform);
                let offset = child_world.position - world.position;
                let projection = dir_world.dot(offset);
                let radius = shape_support_radius(shape, dir_world, &child_world);
                max_proj = max_proj.max(projection + radius);
            }
            max_proj
        }
    }
}

fn shape_support_point(shape: &ColliderShape, dir_world: Vec3, world: &Transform) -> Vec3 {
    let dir_local = world.rotation.conjugate() * dir_world;
    let dir_local = dir_local.normalize_or_zero();
    match shape {
        ColliderShape::Sphere { radius } => {
            world.position
                + dir_world.normalize_or_zero() * radius.max(0.0) * max_scale(world.scale)
        }
        ColliderShape::Box { half_extents } => {
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
            ) * world.scale;
            world.position + world.rotation * local
        }
        ColliderShape::Capsule { radius, height } => {
            let radius = *radius;
            let height = *height;
            let axis = Vec3::Y;
            let cap_offset = axis * 0.5 * height * world.scale.y;
            let top = world.position + world.rotation * cap_offset;
            let bottom = world.position - world.rotation * cap_offset;
            let dir = dir_world.normalize_or_zero();
            if dir.dot(world.rotation * axis) >= 0.0 {
                top + dir * radius.max(0.0) * radial_scale(world.scale)
            } else {
                bottom + dir * radius.max(0.0) * radial_scale(world.scale)
            }
        }
        ColliderShape::Cylinder { radius, height } => {
            let radius = *radius;
            let height = *height;
            let axis = world.rotation * Vec3::Y;
            let dir = dir_world.normalize_or_zero();
            let lateral = (dir - axis * dir.dot(axis)).normalize_or_zero();
            let radial = lateral * radius.max(0.0) * radial_scale(world.scale);
            let axial = axis * (0.5 * height * world.scale.y).copysign(dir.dot(axis));
            world.position + radial + axial
        }
        ColliderShape::ConvexHull { vertices } => {
            let mut best = world.position;
            let mut best_dot = f32::MIN;
            for vertex in vertices {
                let world_vertex = world.position + world.rotation * (*vertex * world.scale);
                let dot = world_vertex.dot(dir_world);
                if dot > best_dot {
                    best_dot = dot;
                    best = world_vertex;
                }
            }
            best
        }
        ColliderShape::Mesh { mesh } => {
            let mut best = world.position;
            let mut best_dot = f32::MIN;
            for vertex in &mesh.vertices {
                let world_vertex = world.position + world.rotation * (*vertex * world.scale);
                let dot = world_vertex.dot(dir_world);
                if dot > best_dot {
                    best_dot = dot;
                    best = world_vertex;
                }
            }
            best
        }
        ColliderShape::Compound { shapes } => {
            let mut best_point = world.position;
            let mut best_dot = f32::MIN;
            for (local_transform, shape) in shapes {
                let child_world = world.combine(local_transform);
                let point = shape_support_point(shape, dir_world, &child_world);
                let dot = point.dot(dir_world);
                if dot > best_dot {
                    best_dot = dot;
                    best_point = point;
                }
            }
            best_point
        }
    }
}

fn radial_scale(scale: Vec3) -> f32 {
    scale.x.abs().max(scale.z.abs())
}

fn max_scale(scale: Vec3) -> f32 {
    scale.x.abs().max(scale.y.abs()).max(scale.z.abs())
}

fn integrate_body_state(body: &RigidBody, dt: f32) -> RigidBody {
    let mut sample = body.clone();
    if sample.is_static {
        return sample;
    }

    sample.transform.position += sample.velocity.linear * dt;
    let angular = sample.velocity.angular;
    let angle = angular.length() * dt;
    if angle > 1e-6 {
        let axis = angular.normalize();
        let delta = Quat::from_axis_angle(axis, angle);
        sample.transform.rotation = (delta * sample.transform.rotation).normalize();
    }
    sample
}

fn relative_velocity_and_speed(
    body_a: &RigidBody,
    body_b: &RigidBody,
    collider_a: &Collider,
    collider_b: &Collider,
    dt: f32,
) -> (Vec3, f32) {
    let linear = body_b.velocity.linear - body_a.velocity.linear;
    let ang_a = body_a.velocity.angular.length() * collider_a.bounding_radius();
    let ang_b = body_b.velocity.angular.length() * collider_b.bounding_radius();
    let angular_effect = (ang_a + ang_b) * dt;
    let speed = (linear.length_squared() + angular_effect * angular_effect).sqrt();
    (linear, speed)
}
