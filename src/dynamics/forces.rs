use crate::core::soa::{BodiesSoA, BodyMut};
use crate::utils::allocator::EntityId;
use glam::Vec3;

/// Trait describing an external force generator applied to rigid bodies.
pub trait ForceGenerator: Send + Sync {
    fn apply(&self, body: &mut BodyMut, dt: f32);
}

/// Constant gravity force scaled per body.
pub struct GravityForce {
    pub gravity: Vec3,
}

impl GravityForce {
    pub fn new(gravity: Vec3) -> Self {
        Self { gravity }
    }
}

impl ForceGenerator for GravityForce {
    fn apply(&self, body: &mut BodyMut, _dt: f32) {
        if body.is_static() {
            return;
        }
        let force = self.gravity * body.mass_properties.mass * (*body.gravity_scale);
        body.apply_force(force);
    }
}

/// Quadratic drag resisting the direction of motion.
pub struct DragForce {
    pub drag_coefficient: f32,
}

impl ForceGenerator for DragForce {
    fn apply(&self, body: &mut BodyMut, _dt: f32) {
        if body.is_static() {
            return;
        }

        let speed = body.velocity.linear.length();
        if speed < 1e-6 {
            return;
        }

        let drag = -body.velocity.linear.normalize() * speed * speed * self.drag_coefficient;
        body.apply_force(drag);
    }
}

/// Hookean spring connecting a body to a target point/stiff anchor.
pub struct SpringForce {
    pub other_body_pos: Vec3,
    pub rest_length: f32,
    pub spring_constant: f32,
    pub damping: f32,
}

impl ForceGenerator for SpringForce {
    fn apply(&self, body: &mut BodyMut, _dt: f32) {
        let displacement = body.transform.position - self.other_body_pos;
        let distance = displacement.length();
        if distance < 1e-6 {
            return;
        }

        let extension = distance - self.rest_length;
        let spring_force = -self.spring_constant * extension * (displacement / distance);
        let damping_force = -self.damping * body.velocity.linear;

        body.apply_force(spring_force + damping_force);
    }
}

/// Collection of forces that can be applied each frame.
pub struct ForceRegistry {
    forces: Vec<Box<dyn ForceGenerator>>,
}

impl Default for ForceRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ForceRegistry {
    pub fn new() -> Self {
        Self { forces: Vec::new() }
    }

    pub fn add_force<F: ForceGenerator + 'static>(&mut self, force: F) {
        self.forces.push(Box::new(force));
    }

    pub fn apply_all(&self, bodies: &mut BodiesSoA, dt: f32) {
        for force in &self.forces {
            for mut body in bodies.iter_mut() {
                force.apply(&mut body, dt);
            }
        }
    }

    pub fn apply_force_to(&self, bodies: &mut BodiesSoA, id: EntityId, dt: f32) {
        if let Some(mut body) = bodies.get_mut(id) {
            for force in &self.forces {
                force.apply(&mut body, dt);
            }
        }
    }
}
