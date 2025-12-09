use std::collections::HashMap;

use glam::{Mat3, Vec3};
use serde::{Deserialize, Serialize};

use super::types::MassProperties;

/// Axis-aligned bounding box used for mesh bounds and BVH nodes.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    pub fn empty() -> Self {
        Self {
            min: Vec3::splat(f32::INFINITY),
            max: Vec3::splat(f32::NEG_INFINITY),
        }
    }

    pub fn extend(&mut self, point: Vec3) {
        self.min = self.min.min(point);
        self.max = self.max.max(point);
    }

    pub fn from_points(points: &[Vec3]) -> Self {
        let mut bounds = Self::empty();
        for &p in points {
            bounds.extend(p);
        }
        bounds
    }

    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    pub fn extent(&self) -> Vec3 {
        (self.max - self.min) * 0.5
    }

    pub fn radius(&self) -> f32 {
        self.extent().length()
    }
}

/// Simple BVH node representation for triangle meshes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeshBvhNode {
    pub bounds: Aabb,
    pub left: Option<usize>,
    pub right: Option<usize>,
    pub start: usize,
    pub count: usize,
}

/// Placeholder BVH storing a single node per mesh for now.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeshBvh {
    pub nodes: Vec<MeshBvhNode>,
}

impl MeshBvh {
    pub fn new(nodes: Vec<MeshBvhNode>) -> Self {
        Self { nodes }
    }
}

/// Triangle mesh collider data used for advanced shapes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TriangleMesh {
    pub vertices: Vec<Vec3>,
    pub indices: Vec<[u32; 3]>,
    pub bounds: Aabb,
    pub bvh: MeshBvh,
}

impl TriangleMesh {
    pub fn builder(vertices: Vec<Vec3>, indices: Vec<[u32; 3]>) -> MeshBuilder {
        MeshBuilder::new(vertices, indices)
    }

    pub fn support_point(&self, direction: Vec3) -> Vec3 {
        let dir = direction.normalize_or_zero();
        if dir == Vec3::ZERO {
            return Vec3::ZERO;
        }
        self.vertices
            .iter()
            .copied()
            .max_by(|a, b| a.dot(dir).partial_cmp(&b.dot(dir)).unwrap())
            .unwrap_or(Vec3::ZERO)
    }

    pub fn support_radius(&self, direction: Vec3) -> f32 {
        let dir = direction.normalize_or_zero();
        if dir == Vec3::ZERO {
            return 0.0;
        }
        self.vertices
            .iter()
            .map(|v| v.dot(dir))
            .fold(f32::NEG_INFINITY, f32::max)
            .max(0.0)
    }

    pub fn bounding_radius(&self) -> f32 {
        self.bounds.radius()
    }

    /// Approximates mass & inertia by treating the mesh bounds as a solid box.
    pub fn approximate_mass_properties(&self, density: f32) -> MassProperties {
        let extents = self.bounds.extent();
        let size = extents * 2.0;
        let volume = size.x * size.y * size.z;
        let density = density.max(0.0001);
        let mass = (volume * density).max(0.0001);
        let factor = mass / 12.0;
        let inertia = Mat3::from_diagonal(Vec3::new(
            factor * (size.y * size.y + size.z * size.z),
            factor * (size.x * size.x + size.z * size.z),
            factor * (size.x * size.x + size.y * size.y),
        ));

        MassProperties { mass, inertia }
    }
}

/// Helper used to cook triangle meshes from raw vertex/index buffers.
#[derive(Debug, Clone)]
pub struct MeshBuilder {
    vertices: Vec<Vec3>,
    indices: Vec<[u32; 3]>,
}

impl MeshBuilder {
    pub fn new(vertices: Vec<Vec3>, indices: Vec<[u32; 3]>) -> Self {
        Self { vertices, indices }
    }

    /// Deduplicates vertices using a quantized grid for stability.
    pub fn weld_vertices(mut self, epsilon: f32) -> Self {
        if epsilon <= 0.0 || self.vertices.is_empty() {
            return self;
        }

        let inv = 1.0 / epsilon;
        let mut map: HashMap<(i32, i32, i32), u32> = HashMap::new();
        let mut new_vertices: Vec<Vec3> = Vec::new();
        let mut remap: Vec<u32> = Vec::with_capacity(self.vertices.len());

        for v in &self.vertices {
            let key = (
                (v.x * inv).round() as i32,
                (v.y * inv).round() as i32,
                (v.z * inv).round() as i32,
            );
            let index = *map.entry(key).or_insert_with(|| {
                let idx = new_vertices.len() as u32;
                new_vertices.push(*v);
                idx
            });
            remap.push(index);
        }

        for tri in &mut self.indices {
            tri[0] = remap[tri[0] as usize];
            tri[1] = remap[tri[1] as usize];
            tri[2] = remap[tri[2] as usize];
        }

        self.vertices = new_vertices;
        self
    }

    /// Recenters vertices around their centroid to keep transforms stable.
    pub fn recenter(mut self) -> Self {
        if self.vertices.is_empty() {
            return self;
        }
        let centroid: Vec3 =
            self.vertices.iter().copied().sum::<Vec3>() / self.vertices.len() as f32;
        for vertex in &mut self.vertices {
            *vertex -= centroid;
        }
        self
    }

    pub fn build(self) -> TriangleMesh {
        let bounds = Aabb::from_points(&self.vertices);
        let node = MeshBvhNode {
            bounds,
            left: None,
            right: None,
            start: 0,
            count: self.indices.len(),
        };
        TriangleMesh {
            vertices: self.vertices,
            indices: self.indices,
            bounds,
            bvh: MeshBvh::new(vec![node]),
        }
    }
}
