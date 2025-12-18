use glam::{Quat, Vec3};

use crate::core::soa::{BodiesSoA, BodyMut};

/// Integrator responsible for stepping rigid bodies forward in time.
#[derive(Debug, Clone)]
pub struct Integrator {
    pub dt: f32,
    pub substeps: u32,
    parallel: bool,
}

impl Integrator {
    pub fn new(dt: f32, substeps: u32) -> Self {
        let substep_dt = dt / substeps.max(1) as f32;
        Self {
            dt: substep_dt,
            substeps: substeps.max(1),
            parallel: false,
        }
    }

    pub fn set_parallel(&mut self, enabled: bool) {
        self.parallel = enabled;
    }

    pub fn integrate_position(&self, body: &mut BodyMut, dt: f32) {
        if body.flags.is_static {
            return;
        }

        body.transform.position += body.velocity.linear * dt;

        let omega_mag = body.velocity.angular.length();
        if omega_mag > 1e-6 {
            let axis = body.velocity.angular / omega_mag;
            let angle = omega_mag * dt;
            let delta = Quat::from_axis_angle(axis, angle);
            body.transform.rotation = (delta * body.transform.rotation).normalize();
        }
    }

    pub fn integrate_velocity(&self, body: &mut BodyMut, dt: f32) {
        if body.flags.is_static {
            return;
        }

        body.velocity.linear += (*body.acceleration) * dt;

        body.velocity.linear *= (1.0 - (*body.linear_damping) * dt).max(0.0);
        body.velocity.angular *= (1.0 - (*body.angular_damping) * dt).max(0.0);

        *body.acceleration = Vec3::ZERO;
    }

    pub fn step(&self, bodies: &mut BodiesSoA) {
        for _ in 0..self.substeps {
            // Parallel disabled for SoA initial implementation
            /*
            if self.parallel {
                bodies.par_for_each_mut(|body| self.integrate_velocity(body, self.dt));
            } else {
            */
            for mut body in bodies.iter_mut() {
                self.integrate_velocity(&mut body, self.dt);
            }
            //}

            /*
            if self.parallel {
                bodies.par_for_each_mut(|body| self.integrate_position(body, self.dt));
            } else {
            */
            for mut body in bodies.iter_mut() {
                self.integrate_position(&mut body, self.dt);
            }
            //}
        }
    }
}
