use crate::collision::{broadphase::BroadPhase, ccd::CCDDetector, contact::ManifoldCache};
use crate::config::DEFAULT_BROADPHASE_CELL_SIZE;

pub struct CollisionManager {
    pub broadphase: BroadPhase,
    pub manifold_cache: ManifoldCache,
    pub ccd: CCDDetector,
}

impl Default for CollisionManager {
    fn default() -> Self {
        Self::new()
    }
}

impl CollisionManager {
    pub fn new() -> Self {
        Self {
            broadphase: BroadPhase::new(DEFAULT_BROADPHASE_CELL_SIZE),
            manifold_cache: ManifoldCache::new(),
            ccd: CCDDetector::new(),
        }
    }
}
