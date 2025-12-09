//! SIMD-related helpers for batch math operations.
//!
//! These helpers lean on `glam`'s SIMD-backed types to accelerate common hot
//! loops such as transforming large point clouds or evaluating support
//! functions for convex hulls.

use glam::{Mat4, Vec3, Vec3A};

/// Controls whether SIMD helpers are allowed to use additional threads for
/// bulk operations.
#[derive(Debug, Clone, Copy)]
pub enum SimdJobMode {
    Serial,
    Parallel,
}

/// Transforms all points by the provided matrix using SIMD-friendly `glam`
/// routines and returns the transformed positions.
pub fn batch_transform_points(points: &[Vec3], matrix: &Mat4) -> Vec<Vec3> {
    points
        .iter()
        .map(|p| matrix.transform_point3(*p))
        .collect()
}

/// Transforms the provided points in-place by the given matrix.
pub fn transform_points_in_place(points: &mut [Vec3], matrix: &Mat4, mode: SimdJobMode) {
    match mode {
        SimdJobMode::Serial => {
            for point in points.iter_mut() {
                *point = matrix.transform_point3(*point);
            }
        }
        SimdJobMode::Parallel => {
            use rayon::prelude::*;
            points.par_iter_mut().for_each(|point| {
                *point = matrix.transform_point3(*point);
            });
        }
    }
}

/// Computes the maximum dot product between the supplied direction and a list
/// of vertices. Internally uses `Vec3A` to leverage SIMD instructions.
pub fn max_dot(vertices: &[Vec3], direction: Vec3) -> f32 {
    if vertices.is_empty() || direction.length_squared() == 0.0 {
        return 0.0;
    }

    let dir = Vec3A::from(direction);
    let mut max_value = f32::NEG_INFINITY;

    for vertex in vertices {
        let dot = Vec3A::from(*vertex).dot(dir);
        if dot > max_value {
            max_value = dot;
        }
    }

    if max_value.is_finite() {
        max_value
    } else {
        0.0
    }
}

/// Returns the vertex that produces the maximum dot product with the supplied
/// direction. Useful for support functions in GJK/SAT code paths.
pub fn max_dot_point(vertices: &[Vec3], direction: Vec3) -> Option<(Vec3, f32)> {
    if vertices.is_empty() || direction.length_squared() == 0.0 {
        return None;
    }

    let dir = Vec3A::from(direction);
    let mut best_vertex = Vec3::ZERO;
    let mut max_value = f32::NEG_INFINITY;

    for vertex in vertices {
        let vert = Vec3A::from(*vertex);
        let dot = vert.dot(dir);
        if dot > max_value {
            max_value = dot;
            best_vertex = *vertex;
        }
    }

    if max_value.is_finite() {
        Some((best_vertex, max_value))
    } else {
        None
    }
}

/// Computes the maximum distance of the provided points from the origin.
pub fn max_length(vertices: &[Vec3]) -> f32 {
    if vertices.is_empty() {
        return 0.0;
    }

    let mut max_len_sq = 0.0;
    for vertex in vertices {
        let len_sq = Vec3A::from(*vertex).length_squared();
        if len_sq > max_len_sq {
            max_len_sq = len_sq;
        }
    }

    max_len_sq.sqrt()
}
