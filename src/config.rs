//! Global configuration constants for the Particle Accelerator engine.

/// Default gravity vector applied in the physics world (Y-up).
pub const DEFAULT_GRAVITY: [f32; 3] = [0.0, -9.81, 0.0];

/// Default integration timestep (in seconds).
pub const DEFAULT_TIME_STEP: f32 = 1.0 / 60.0;

/// Number of constraint solver iterations performed per step.
pub const DEFAULT_SOLVER_ITERATIONS: u32 = 4;

/// Default damping applied to linear velocity.
pub const DEFAULT_LINEAR_DAMPING: f32 = 0.02;

/// Default damping applied to angular velocity.
pub const DEFAULT_ANGULAR_DAMPING: f32 = 0.02;

/// Default cell size for the broad-phase uniform grid.
pub const DEFAULT_BROADPHASE_CELL_SIZE: f32 = 5.0;
