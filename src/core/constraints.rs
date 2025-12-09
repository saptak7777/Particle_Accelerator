use crate::utils::allocator::EntityId;
use glam::Vec3;
use serde::{Deserialize, Serialize};

/// Supported joint types for phase 4+ development.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Joint {
    Fixed {
        body_a: EntityId,
        body_b: EntityId,
        offset_a: Vec3,
        offset_b: Vec3,
    },
    Revolute {
        body_a: EntityId,
        body_b: EntityId,
        pivot: Vec3,
        axis: Vec3,
        motor_target_velocity: Option<f32>,
        motor_strength: f32,
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
