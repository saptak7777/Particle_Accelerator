use crate::core::constraints::Joint;
use crate::dynamics::{forces::ForceRegistry, solver::PGSSolver};

pub struct DynamicsManager {
    pub solver: PGSSolver,
    pub joints: Vec<Joint>,
    pub force_registry: ForceRegistry,
}

impl Default for DynamicsManager {
    fn default() -> Self {
        Self::new()
    }
}

impl DynamicsManager {
    pub fn new() -> Self {
        Self {
            solver: PGSSolver::new(),
            joints: Vec::new(),
            force_registry: ForceRegistry::new(),
        }
    }
}
