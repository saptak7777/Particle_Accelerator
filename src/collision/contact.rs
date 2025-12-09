use crate::{
    core::{collider::Collider, rigidbody::RigidBody},
    dynamics::solver::Contact,
};

/// Contact manifold storing collision contacts for a pair.
#[derive(Debug, Clone)]
pub struct ContactManifold {
    pub contacts: Vec<Contact>,
}

impl ContactManifold {
    pub fn generate(
        collider_a: &Collider,
        body_a: &RigidBody,
        collider_b: &Collider,
        body_b: &RigidBody,
    ) -> Option<Self> {
        use crate::collision::narrowphase::NarrowPhase;

        let contact = NarrowPhase::collide(collider_a, body_a, collider_b, body_b)?;

        Some(ContactManifold {
            contacts: vec![contact],
        })
    }
}

/// Simplified structure passed into solvers (placeholder for real data).
pub struct ContactSolverInput<'a> {
    pub bodies: &'a mut [RigidBody],
    pub contacts: &'a [Contact],
}
