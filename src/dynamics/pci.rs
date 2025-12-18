use crate::{core::soa::BodiesSoA, dynamics::solver::Contact};

/// Predictive-corrective integrator that damps future interpenetrations.
#[derive(Debug, Clone)]
pub struct PredictiveCorrectiveIntegrator {
    pub correction_iterations: u32,
    pub penetration_slop: f32,
}

impl Default for PredictiveCorrectiveIntegrator {
    fn default() -> Self {
        Self::new(0)
    }
}

impl PredictiveCorrectiveIntegrator {
    pub fn new(correction_iterations: u32) -> Self {
        Self {
            correction_iterations,
            penetration_slop: 1e-3,
        }
    }

    pub fn set_iterations(&mut self, iterations: u32) {
        self.correction_iterations = iterations;
    }

    pub fn apply(&self, bodies: &mut BodiesSoA, contacts: &[Contact], dt: f32) {
        if self.correction_iterations == 0 {
            return;
        }
        if contacts.is_empty() {
            return;
        }
        let dt = dt.max(1e-4);

        // Loop over iterations
        for _ in 0..self.correction_iterations {
            for contact in contacts {
                let (body_a, body_b) = match bodies.get2_mut(contact.body_a, contact.body_b) {
                    Some(pair) => pair,
                    None => continue,
                };

                if body_a.is_static() && body_b.is_static() {
                    continue;
                }

                let normal = contact.normal;

                // 1. Calculate relative velocity at contact point
                let r_a = contact.point - body_a.transform.position;
                let r_b = contact.point - body_b.transform.position;

                let v_a = body_a.velocity.linear + body_a.velocity.angular.cross(r_a);
                let v_b = body_b.velocity.linear + body_b.velocity.angular.cross(r_b);
                let rel_vel = v_b - v_a;
                let vel_along_normal = rel_vel.dot(normal);

                // 2. Predict penetration at end of frame
                // depth is positive for penetration. contact.depth is current penetration.
                // If velocity is separating (vel_along_normal > 0), penetration decreases.
                // predicted_penetration = current_penetration - vel_along_normal * dt
                // Note: Speculative contacts might have negative depth (separation).
                let predicted_penetration = contact.depth - vel_along_normal * dt;

                if predicted_penetration <= self.penetration_slop {
                    continue;
                }

                // 3. Solve for correction impulse
                // The goal is to reduce predicted_penetration to penetration_slop.
                // delta_penetration = predicted_penetration - slop
                // required_vel_change = delta_penetration / dt

                let penetration_error = predicted_penetration - self.penetration_slop;
                let velocity_bias = penetration_error / dt;

                // Compute effective mass (Inverse K matrix diagonal)
                // Compute effective mass (Inverse K matrix diagonal)
                let mut k_mass = *body_a.inverse_mass + *body_b.inverse_mass;

                if !body_a.is_static() {
                    let rn_a = r_a.cross(normal);
                    let inertia_a = *body_a.inverse_inertia * rn_a;
                    k_mass += inertia_a.cross(r_a).dot(normal);
                }

                if !body_b.is_static() {
                    let rn_b = r_b.cross(normal);
                    let inertia_b = *body_b.inverse_inertia * rn_b;
                    k_mass += inertia_b.cross(r_b).dot(normal);
                }

                if k_mass <= 1e-6 {
                    continue;
                }

                let impulse_mag = velocity_bias / k_mass;

                // Clamp impulse??
                // PCI usually accumulates or applies greedily?
                // Standard implementation applies immediately.
                // Impulse should be positive (pushing apart).
                // If velocity_bias is positive (need to separate), impulse is positive.
                let impulse = normal * impulse_mag;

                if !body_a.is_static() {
                    body_a.velocity.linear -= impulse * (*body_a.inverse_mass);
                    body_a.velocity.angular -= (*body_a.inverse_inertia) * r_a.cross(impulse);
                }

                if !body_b.is_static() {
                    body_b.velocity.linear += impulse * (*body_b.inverse_mass);
                    body_b.velocity.angular += (*body_b.inverse_inertia) * r_b.cross(impulse);
                }
            }
        }
    }
}
