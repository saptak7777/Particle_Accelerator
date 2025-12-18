use crate::utils::allocator::EntityId;
use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};

/// Supported joint types for phase 4+ development.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Joint {
    /// Fully locks two bodies together at a specific relative transform.
    Fixed {
        body_a: EntityId,
        body_b: EntityId,
        local_pivot_a: Vec3,
        local_pivot_b: Vec3,
        /// Initial relative rotation captured at joint creation (body_a_space).
        local_frame_a: Quat,
        local_frame_b: Quat,
    },
    /// Allows rotation around a single axis.
    Revolute {
        body_a: EntityId,
        body_b: EntityId,
        local_pivot_a: Vec3,
        local_pivot_b: Vec3,
        local_axis_a: Vec3,
        local_axis_b: Vec3,
        local_basis_a: Vec3,
        local_basis_b: Vec3,

        enable_motor: bool,
        motor_speed: f32,
        max_motor_torque: f32,

        enable_limit: bool,
        lower_angle: f32,
        upper_angle: f32,
    },
    /// Allows translation along a single axis, locking all rotations.
    Prismatic {
        body_a: EntityId,
        body_b: EntityId,
        local_pivot_a: Vec3,
        local_pivot_b: Vec3,
        /// Sliding axis in body A's local space.
        local_axis_a: Vec3,
        /// Rotational reference frames to lock orientation.
        local_frame_a: Quat,
        local_frame_b: Quat,

        enable_limit: bool,
        lower_limit: f32,
        upper_limit: f32,

        enable_motor: bool,
        motor_speed: f32,
        max_motor_force: f32,
    },
    Spring {
        body_a: EntityId,
        body_b: EntityId,
        rest_length: f32,
        stiffness: f32,
        damping: f32,
    },
    Distance {
        body_a: EntityId,
        body_b: EntityId,
        distance: f32,
    },
}

impl Joint {
    pub fn bodies(&self) -> (EntityId, EntityId) {
        match self {
            Joint::Fixed { body_a, body_b, .. }
            | Joint::Revolute { body_a, body_b, .. }
            | Joint::Prismatic { body_a, body_b, .. }
            | Joint::Spring { body_a, body_b, .. }
            | Joint::Distance { body_a, body_b, .. } => (*body_a, *body_b),
        }
    }
}
