use std::collections::{HashMap, HashSet};

use glam::Vec3;

use crate::{
    core::{
        collider::{Collider, ColliderShape},
        rigidbody::RigidBody,
    },
    utils::{
        allocator::{Arena, EntityId},
        simd,
    },
};

/// Uniform grid spatial partitioning used by the broad-phase.
pub struct SpatialGrid {
    cell_size: f32,
    grid: HashMap<(i32, i32, i32), Vec<EntityId>>,
}

impl SpatialGrid {
    pub fn new(cell_size: f32) -> Self {
        Self {
            cell_size,
            grid: HashMap::new(),
        }
    }

    fn world_to_grid(&self, pos: Vec3) -> (i32, i32, i32) {
        (
            (pos.x / self.cell_size).floor() as i32,
            (pos.y / self.cell_size).floor() as i32,
            (pos.z / self.cell_size).floor() as i32,
        )
    }

    pub fn insert(&mut self, entity_id: EntityId, position: Vec3, radius: f32) {
        let min_cell = self.world_to_grid(position - Vec3::splat(radius));
        let max_cell = self.world_to_grid(position + Vec3::splat(radius));

        for x in min_cell.0..=max_cell.0 {
            for y in min_cell.1..=max_cell.1 {
                for z in min_cell.2..=max_cell.2 {
                    self.grid
                        .entry((x, y, z))
                        .or_default()
                        .push(entity_id);
                }
            }
        }
    }

    pub fn query(&self, position: Vec3, radius: f32) -> Vec<EntityId> {
        let mut results = Vec::new();
        let min_cell = self.world_to_grid(position - Vec3::splat(radius));
        let max_cell = self.world_to_grid(position + Vec3::splat(radius));

        for x in min_cell.0..=max_cell.0 {
            for y in min_cell.1..=max_cell.1 {
                for z in min_cell.2..=max_cell.2 {
                    if let Some(entities) = self.grid.get(&(x, y, z)) {
                        results.extend(entities);
                    }
                }
            }
        }

        results.sort();
        results.dedup();
        results
    }

    pub fn update(&mut self, colliders: &Arena<Collider>, bodies: &Arena<RigidBody>) {
        self.grid.clear();

        for collider_id in colliders.ids() {
            let collider = match colliders.get(collider_id) {
                Some(c) => c,
                None => continue,
            };
            let body = match bodies.get(collider.rigidbody_id) {
                Some(b) => b,
                None => continue,
            };

            let transform = collider.world_transform(&body.transform);
            let radius = BroadPhase::get_collider_radius(&collider.shape);
            self.insert(collider.id, transform.position, radius);
        }
    }
}

/// Broad phase driver returning potential collider pairs.
pub struct BroadPhase {
    grid: SpatialGrid,
    pub min_separation: f32,
}

impl BroadPhase {
    pub fn new(cell_size: f32) -> Self {
        Self {
            grid: SpatialGrid::new(cell_size),
            min_separation: 0.01,
        }
    }

    pub fn get_potential_pairs(
        &mut self,
        colliders: &Arena<Collider>,
        bodies: &Arena<RigidBody>,
    ) -> Vec<(EntityId, EntityId)> {
        self.grid.update(colliders, bodies);

        let mut pairs = Vec::new();
        let mut checked = HashSet::new();

        for collider_id in colliders.ids() {
            let collider = match colliders.get(collider_id) {
                Some(c) => c,
                None => continue,
            };
            let body = match bodies.get(collider.rigidbody_id) {
                Some(b) => b,
                None => continue,
            };

            let transform = collider.world_transform(&body.transform);
            let radius = Self::get_collider_radius(&collider.shape);
            let nearby = self.grid.query(transform.position, radius);

            for other_id in nearby {
                if collider.id == other_id {
                    continue;
                }

                let pair_key = if collider.id.index() < other_id.index() {
                    (collider.id, other_id)
                } else {
                    (other_id, collider.id)
                };

                if checked.insert((pair_key.0.index(), pair_key.1.index())) {
                    pairs.push(pair_key);
                }
            }
        }

        pairs
    }

    pub fn get_collider_radius(shape: &ColliderShape) -> f32 {
        match shape {
            ColliderShape::Sphere { radius } => *radius,
            ColliderShape::Box { half_extents } => half_extents.length(),
            ColliderShape::Capsule { radius, height } => {
                (radius * radius + (height / 2.0) * (height / 2.0)).sqrt()
            }
            ColliderShape::Cylinder { radius, height } => {
                (radius * radius + (height / 2.0) * (height / 2.0)).sqrt()
            }
            ColliderShape::ConvexHull { vertices } => simd::max_length(vertices),
            ColliderShape::Compound { shapes } => shapes
                .iter()
                .map(|(transform, shape)| transform.position.length() + Self::get_collider_radius(shape))
                .fold(0.0, f32::max),
            ColliderShape::Mesh { mesh } => mesh.bounding_radius(),
        }
    }
}
