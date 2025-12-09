use crate::{
    core::rigidbody::RigidBody,
    dynamics::solver::Contact,
};

/// Continuous collision detector placeholder.
pub struct CCDDetector {
    pub enabled: bool,
    pub ccd_threshold: f32,
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
        }
    }

    pub fn detect_ccd(
        &self,
        body_a: &RigidBody,
        body_b: &RigidBody,
        dt: f32,
    ) -> Option<Contact> {
        if !self.enabled {
            return None;
        }

        let speed_a = body_a.velocity.linear.length();
        let speed_b = body_b.velocity.linear.length();

        if speed_a < self.ccd_threshold && speed_b < self.ccd_threshold {
            return None;
        }

        // Placeholder: returns None; real implementation would sweep volumes.
        let _swept_radius = (speed_a + speed_b) * dt;
        None
    }
}
