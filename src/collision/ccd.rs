use glam::Vec3;

use crate::{
    core::{
        collider::{Collider, ColliderShape},
        rigidbody::RigidBody,
        types::Transform,
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

        let relative_velocity = body_b.velocity.linear - body_a.velocity.linear;
        let relative_speed = relative_velocity.length();
        if relative_speed < self.ccd_threshold {
            return None;
        }

        let motion_dir = relative_velocity.normalize_or_zero();
        if motion_dir == Vec3::ZERO {
            return None;
        }

        let relative_position = body_b.transform.position - body_a.transform.position;
        let base_radius = support_radius(collider_a, body_a, motion_dir)
            + support_radius(collider_b, body_b, -motion_dir);
        let angular_expand = self.angular_padding * dt
            * (body_a.velocity.angular.length() * collider_a.bounding_radius()
                + body_b.velocity.angular.length() * collider_b.bounding_radius());
        let combined_radius = base_radius + angular_expand;

        if combined_radius <= 0.0 {
            return None;
        }

        let a = relative_velocity.length_squared();
        if a <= f32::EPSILON {
            return None;
        }
        let b = 2.0 * relative_position.dot(relative_velocity);
        let c = relative_position.length_squared() - combined_radius.powi(2);

        let discriminant = b * b - 4.0 * a * c;
        if discriminant < 0.0 {
            return None;
        }

        let sqrt_disc = discriminant.sqrt();
        let toi = (-b - sqrt_disc) / (2.0 * a);
        if !(0.0..=dt).contains(&toi) {
            return None;
        }

        let position_a = body_a.transform.position + body_a.velocity.linear * toi;
        let position_b = body_b.transform.position + body_b.velocity.linear * toi;
        let normal = (position_b - position_a).normalize_or_zero();
        if normal == Vec3::ZERO {
            return None;
        }

        let support_a = support_radius(collider_a, body_a, normal);
        let support_b = support_radius(collider_b, body_b, -normal);
        let separation = (position_b - position_a).length();
        let contact_point = position_a + normal * support_a;
        let relative_velocity_along_normal = relative_velocity.dot(normal);
        let depth = (support_a + support_b - separation).max(0.0);

        Some(CCDResult {
            contact: Contact {
                body_a: body_a.id,
                body_b: body_b.id,
                point: contact_point,
                normal,
                depth,
                relative_velocity: relative_velocity_along_normal,
            },
            time_of_impact: toi,
        })
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

        let (mut body_b, collider_b) =
            make_sphere(2, 0.25, Vec3::new(0.0, 0.0, 5.0), Vec3::new(0.0, 0.0, -30.0));
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

fn radial_scale(scale: Vec3) -> f32 {
    scale.x.abs().max(scale.z.abs())
}

fn max_scale(scale: Vec3) -> f32 {
    scale.x
        .abs()
        .max(scale.y.abs())
        .max(scale.z.abs())
}
