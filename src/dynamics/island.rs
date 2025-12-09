use std::collections::{HashMap, HashSet};

use crate::{
    core::{constraints::Joint, rigidbody::RigidBody},
    dynamics::solver::Contact,
    utils::allocator::{Arena, EntityId},
};

/// Represents a connected set of bodies/contacts that can be solved independently.
pub struct Island {
    pub bodies: Vec<EntityId>,
    pub contacts: Vec<Contact>,
    pub joints: Vec<Joint>,
    pub is_awake: bool,
}

/// Builds islands each step and manages sleeping state.
pub struct IslandManager {
    islands: Vec<Island>,
    adjacency: HashMap<EntityId, Vec<EntityId>>,
}

impl Default for IslandManager {
    fn default() -> Self {
        Self::new()
    }
}

impl IslandManager {
    pub fn new() -> Self {
        Self {
            islands: Vec::new(),
            adjacency: HashMap::new(),
        }
    }

    pub fn build_islands(
        &mut self,
        bodies: &Arena<RigidBody>,
        contacts: &[Contact],
        joints: &[Joint],
    ) {
        self.islands.clear();
        self.adjacency.clear();

        for contact in contacts {
            self.adjacency
                .entry(contact.body_a)
                .or_default()
                .push(contact.body_b);
            self.adjacency
                .entry(contact.body_b)
                .or_default()
                .push(contact.body_b);
        }

        for joint in joints {
            let (body_a, body_b) = joint.bodies();
            self.adjacency.entry(body_a).or_default().push(body_b);
            self.adjacency.entry(body_b).or_default().push(body_a);
        }

        let mut visited = HashSet::new();
        for body_id in bodies.ids() {
            if visited.contains(&body_id) {
                continue;
            }
            let bodies_in_island = self.depth_first_collect(body_id, &mut visited);
            if bodies_in_island.is_empty() {
                continue;
            }

            let id_set: HashSet<_> = bodies_in_island.iter().copied().collect();
            let island_contacts = contacts
                .iter()
                .filter(|c| id_set.contains(&c.body_a) || id_set.contains(&c.body_b))
                .cloned()
                .collect();
            let island_joints = joints
                .iter()
                .filter(|j| {
                    let (a, b) = j.bodies();
                    id_set.contains(&a) || id_set.contains(&b)
                })
                .cloned()
                .collect();
            let is_awake = bodies_in_island.iter().any(|id| {
                bodies
                    .get(*id)
                    .map(|body| body.is_awake && body.is_enabled)
                    .unwrap_or(false)
            });
            self.islands.push(Island {
                bodies: bodies_in_island,
                contacts: island_contacts,
                joints: island_joints,
                is_awake,
            });
        }
    }

    fn depth_first_collect(
        &self,
        start: EntityId,
        visited: &mut HashSet<EntityId>,
    ) -> Vec<EntityId> {
        let mut stack = vec![start];
        let mut result = Vec::new();

        while let Some(node) = stack.pop() {
            if visited.insert(node) {
                result.push(node);
                if let Some(neighbors) = self.adjacency.get(&node) {
                    stack.extend(neighbors.iter().copied());
                }
            }
        }

        result
    }

    pub fn update_sleeping(&mut self, bodies: &mut Arena<RigidBody>) {
        for island in &mut self.islands {
            let mut avg_velocity = 0.0;
            for body_id in &island.bodies {
                if let Some(body) = bodies.get(*body_id) {
                    avg_velocity += body.velocity.linear.length_squared()
                        + body.velocity.angular.length_squared();
                }
            }
            avg_velocity /= island.bodies.len().max(1) as f32;
            if avg_velocity < 0.01 {
                island.is_awake = false;
                for body_id in &island.bodies {
                    if let Some(body) = bodies.get_mut(*body_id) {
                        body.is_awake = false;
                    }
                }
            } else {
                island.is_awake = true;
                for body_id in &island.bodies {
                    if let Some(body) = bodies.get_mut(*body_id) {
                        body.is_awake = true;
                    }
                }
            }
        }
    }

    pub fn islands(&self) -> &[Island] {
        &self.islands
    }
}
