use glam::Vec3;

const EPSILON: f32 = 1e-4;

#[derive(Debug, Clone, Copy)]
pub struct Plane {
    normal: Vec3,
    distance: f32,
}

impl Plane {
    fn from_point_normal(point: Vec3, normal: Vec3) -> Self {
        let n = normal.normalize_or_zero();
        Self {
            normal: n,
            distance: n.dot(point),
        }
    }

    fn signed_distance(&self, point: Vec3) -> f32 {
        self.normal.dot(point) - self.distance
    }
}

/// Clips the provided polygon against a set of planes using the Sutherland-Hodgman algorithm.
pub fn clip_polygon(vertices: &[Vec3], planes: &[Plane]) -> Vec<Vec3> {
    let mut output = vertices.to_vec();
    for plane in planes {
        output = clip_against_plane(&output, *plane);
        if output.is_empty() {
            break;
        }
    }
    output
}

fn clip_against_plane(vertices: &[Vec3], plane: Plane) -> Vec<Vec3> {
    if vertices.is_empty() {
        return Vec::new();
    }

    let mut clipped = Vec::new();
    for i in 0..vertices.len() {
        let current = vertices[i];
        let next = vertices[(i + 1) % vertices.len()];

        let current_dist = plane.signed_distance(current);
        let next_dist = plane.signed_distance(next);

        let current_inside = current_dist <= EPSILON;
        let next_inside = next_dist <= EPSILON;

        if current_inside && next_inside {
            clipped.push(next);
        } else if current_inside && !next_inside {
            if let Some(intersection) = line_plane_intersection(current, next, current_dist, next_dist)
            {
                clipped.push(intersection);
            }
        } else if !current_inside && next_inside {
            if let Some(intersection) = line_plane_intersection(current, next, current_dist, next_dist)
            {
                clipped.push(intersection);
            }
            clipped.push(next);
        }
    }

    clipped
}

fn line_plane_intersection(
    start: Vec3,
    end: Vec3,
    start_dist: f32,
    end_dist: f32,
) -> Option<Vec3> {
    let denom = start_dist - end_dist;
    if denom.abs() <= EPSILON {
        return None;
    }
    let t = start_dist / denom;
    Some(start + (end - start) * t)
}

/// Convenience helper for constructing rectangle clipping planes given tangents and half-extents.
pub fn rectangle_planes(center: Vec3, tangent_u: Vec3, tangent_v: Vec3, half_u: f32, half_v: f32) -> [Plane; 4] {
    [
        Plane::from_point_normal(center + tangent_u * half_u,  tangent_u),
        Plane::from_point_normal(center - tangent_u * half_u, -tangent_u),
        Plane::from_point_normal(center + tangent_v * half_v,  tangent_v),
        Plane::from_point_normal(center - tangent_v * half_v, -tangent_v),
    ]
}
