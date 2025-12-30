type ManifoldDebugHook = dyn Fn(&ManifoldDebugInfo) + Send + Sync;

use std::{cmp::Ordering, collections::HashMap, fmt};

use glam::Vec3;

use crate::{
    collision::{
        clipping::{clip_polygon, rectangle_planes},
        narrowphase::NarrowPhase,
    },
    core::{
        collider::{Collider, ColliderShape},
        rigidbody::RigidBody,
        types::{Material, MaterialPairProperties},
    },
    dynamics::solver::Contact,
    utils::allocator::EntityId,
};

const MAX_MANIFOLD_POINTS: usize = 4;
const MANIFOLD_MAX_AGE: u32 = 12;
const FEATURE_QUANTIZATION_SCALE: f32 = 1000.0;
const FEATURE_QUANTIZATION_MAX: i32 = 1 << 20;

/// Lightweight contact point produced by the narrow phase before persistence.
#[derive(Debug, Clone)]
pub struct RawContactPoint {
    pub point: Vec3,
    pub depth: f32,
    pub feature_id: u64,
}

/// Raw manifold description generated per frame by the narrow phase.
#[derive(Debug, Clone)]
pub struct ContactManifold {
    pub normal: Vec3,
    pub points: Vec<RawContactPoint>,
    pub simplex: Option<Vec<Vec3>>,
}

impl ContactManifold {
    pub fn generate(
        collider_a: &Collider,
        body_a: &RigidBody,
        collider_b: &Collider,
        body_b: &RigidBody,
    ) -> Option<Self> {
        use crate::collision::narrowphase::NarrowPhase;

        if let (
            ColliderShape::Box { half_extents: he_a },
            ColliderShape::Box { half_extents: he_b },
        ) = (&collider_a.shape, &collider_b.shape)
        {
            if let Some(manifold) =
                generate_box_box_manifold(collider_a, body_a, *he_a, collider_b, body_b, *he_b)
            {
                return Some(manifold);
            }
        }

        let (contact, simplex) =
            NarrowPhase::collide(collider_a, body_a, collider_b, body_b, None)?;

        Some(Self {
            normal: contact.normal,
            points: vec![RawContactPoint {
                point: contact.point,
                depth: contact.depth,
                feature_id: contact.feature_id,
            }],
            simplex: Some(simplex),
        })
    }
}

#[derive(Debug, Clone)]
pub struct ContactPoint {
    pub feature_id: u64,
    pub world_point: Vec3,
    pub local_a: Vec3,
    pub local_b: Vec3,
    pub depth: f32,
    pub normal_impulse: f32,
    pub tangent_impulse: Vec3,
    pub rolling_impulse: Vec3,
    pub torsional_impulse: f32,
}

/// Snapshot-friendly view of a persisted manifold point for debugging/inspection.
#[derive(Debug, Clone)]
pub struct ManifoldPointDebugInfo {
    pub feature_id: u64,
    pub world_point: Vec3,
    pub local_a: Vec3,
    pub local_b: Vec3,
    pub depth: f32,
    pub normal_impulse: f32,
    pub tangent_impulse: Vec3,
    pub rolling_impulse: Vec3,
    pub torsional_impulse: f32,
}

/// Snapshot of a persistent manifold containing bodies, points, and material data.
#[derive(Debug, Clone)]
pub struct ManifoldDebugInfo {
    pub body_a: EntityId,
    pub body_b: EntityId,
    pub normal: Vec3,
    pub frame: u32,
    pub points: Vec<ManifoldPointDebugInfo>,
    pub material: MaterialPairProperties,
}

#[derive(Debug)]
pub struct PersistentManifold {
    body_a: EntityId,
    body_b: EntityId,
    pub normal: Vec3,
    pub points: Vec<ContactPoint>,
    pub last_frame: u32,
    material: MaterialPairProperties,
    pub simplex: Option<Vec<Vec3>>,
}

impl PersistentManifold {
    fn new(body_a: EntityId, body_b: EntityId) -> Self {
        Self {
            body_a,
            body_b,
            normal: Vec3::Y,
            points: Vec::new(),
            last_frame: 0,
            material: MaterialPairProperties::default(),
            simplex: None,
        }
    }

    fn update(
        &mut self,
        normal: Vec3,
        raw_points: &[RawContactPoint],
        body_a: &RigidBody,
        body_b: &RigidBody,
        frame: u32,
        simplex: Option<Vec<Vec3>>,
    ) {
        self.normal = normal;
        self.last_frame = frame;
        self.simplex = simplex;
        self.material = Material::combine_pair(&body_a.material, &body_b.material);

        let mut updated_points = Vec::with_capacity(raw_points.len());
        for raw in raw_points {
            if let Some(existing) = self.points.iter().find(|p| p.feature_id == raw.feature_id) {
                let mut point = existing.clone();
                point.world_point = raw.point;
                point.local_a = world_to_local(body_a, raw.point);
                point.local_b = world_to_local(body_b, raw.point);
                point.depth = raw.depth;
                updated_points.push(point);
            } else {
                updated_points.push(ContactPoint {
                    feature_id: raw.feature_id,
                    world_point: raw.point,
                    local_a: world_to_local(body_a, raw.point),
                    local_b: world_to_local(body_b, raw.point),
                    depth: raw.depth,
                    normal_impulse: 0.0,
                    tangent_impulse: Vec3::ZERO,
                    rolling_impulse: Vec3::ZERO,
                    torsional_impulse: 0.0,
                });
            }
        }

        if updated_points.len() > MAX_MANIFOLD_POINTS {
            self.points = select_best_points(updated_points, MAX_MANIFOLD_POINTS, self.normal);
        } else {
            self.points = updated_points;
        }
    }

    fn to_contacts(&self) -> Vec<Contact> {
        self.points
            .iter()
            .map(|point| Contact {
                body_a: self.body_a,
                body_b: self.body_b,
                point: point.world_point,
                normal: self.normal,
                depth: point.depth,
                relative_velocity: 0.0,
                feature_id: point.feature_id,
                accumulated_normal_impulse: point.normal_impulse,
                accumulated_tangent_impulse: point.tangent_impulse,
                accumulated_rolling_impulse: point.rolling_impulse,
                accumulated_torsional_impulse: point.torsional_impulse,
                material: self.material,
            })
            .collect()
    }

    fn apply_impulses(&mut self, contact: &Contact) {
        if let Some(point) = self
            .points
            .iter_mut()
            .find(|p| p.feature_id == contact.feature_id)
        {
            point.normal_impulse = contact.accumulated_normal_impulse;
            point.tangent_impulse = contact.accumulated_tangent_impulse;
            point.rolling_impulse = contact.accumulated_rolling_impulse;
            point.torsional_impulse = contact.accumulated_torsional_impulse;
            point.world_point = contact.point;
        }
    }

    fn debug_snapshot(&self, frame: u32) -> ManifoldDebugInfo {
        ManifoldDebugInfo {
            body_a: self.body_a,
            body_b: self.body_b,
            normal: self.normal,
            frame,
            material: self.material,
            points: self
                .points
                .iter()
                .map(|p| ManifoldPointDebugInfo {
                    feature_id: p.feature_id,
                    world_point: p.world_point,
                    local_a: p.local_a,
                    local_b: p.local_b,
                    depth: p.depth,
                    normal_impulse: p.normal_impulse,
                    tangent_impulse: p.tangent_impulse,
                    rolling_impulse: p.rolling_impulse,
                    torsional_impulse: p.torsional_impulse,
                })
                .collect(),
        }
    }
}

fn world_to_local(body: &RigidBody, point: Vec3) -> Vec3 {
    let relative = point - body.transform.position;
    body.transform.rotation.conjugate() * relative
}

fn generate_box_box_manifold(
    collider_a: &Collider,
    body_a: &RigidBody,
    half_extents_a: Vec3,
    collider_b: &Collider,
    body_b: &RigidBody,
    half_extents_b: Vec3,
) -> Option<ContactManifold> {
    let box_a = BoxGeometry::new(collider_a, body_a, half_extents_a);
    let box_b = BoxGeometry::new(collider_b, body_b, half_extents_b);

    let descriptor = find_face_axis(&box_a, &box_b)?;

    let (reference, incident, reference_is_a) = if descriptor.reference_is_a {
        (box_a, box_b, true)
    } else {
        (box_b, box_a, false)
    };

    let reference_axis = descriptor.axis_index;
    let reference_sign = descriptor.face_sign;
    let reference_normal = reference.axes[reference_axis] * reference_sign;
    let reference_center = reference.center
        + reference.axes[reference_axis] * reference.half_extents[reference_axis] * reference_sign;

    let normal = if reference_is_a {
        reference_normal
    } else {
        -reference_normal
    };

    let (incident_axis, incident_sign) = find_incident_face(&incident, normal);
    let incident_vertices = face_vertices(&incident, incident_axis, incident_sign);
    let incident_poly: Vec<Vec3> = incident_vertices.to_vec();

    let (tan_indices_u, tan_indices_v) = face_tangent_indices(reference_axis);
    let tangent_u = reference.axes[tan_indices_u];
    let tangent_v = reference.axes[tan_indices_v];
    let half_u = reference.half_extents[tan_indices_u];
    let half_v = reference.half_extents[tan_indices_v];
    let planes = rectangle_planes(reference_center, tangent_u, tangent_v, half_u, half_v);

    let mut clipped = clip_polygon(&incident_poly, &planes);
    if clipped.is_empty() {
        return None;
    }
    clipped = dedup_points(clipped);
    if clipped.is_empty() {
        return None;
    }

    if clipped.len() > MAX_MANIFOLD_POINTS {
        let centroid = centroid(&clipped);
        clipped.sort_by(|a, b| {
            let da = (*a - centroid).length_squared();
            let db = (*b - centroid).length_squared();
            db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal)
        });
        clipped.truncate(MAX_MANIFOLD_POINTS);
    }

    let incident_body = if reference_is_a { body_b } else { body_a };

    let mut points = Vec::new();
    for point in clipped {
        let depth = (reference_center - point).dot(reference_normal);
        if depth <= 0.0 {
            continue;
        }
        let local = world_to_local(incident_body, point);
        let feature_id = feature_id_from_local(incident_body.id, local);
        points.push(RawContactPoint {
            point,
            depth,
            feature_id,
        });
    }

    if points.is_empty() {
        return None;
    }

    Some(ContactManifold {
        normal,
        points,
        simplex: None,
    })
}

fn feature_id_from_local(body: EntityId, local: Vec3) -> u64 {
    fn quantize(value: f32) -> u64 {
        let scaled = (value * FEATURE_QUANTIZATION_SCALE).round();
        let max = (FEATURE_QUANTIZATION_MAX - 1) as f32;
        let min = -max;
        let clamped = scaled.clamp(min, max);
        let shifted = clamped as i32 + FEATURE_QUANTIZATION_MAX;
        (shifted as u32 & 0x1F_FFFF) as u64
    }

    let qx = quantize(local.x);
    let qy = quantize(local.y);
    let qz = quantize(local.z);
    let packed = qx | (qy << 21) | (qz << 42);
    packed ^ ((body.index() as u64) << 1)
}

#[derive(Clone, Copy)]
struct BoxGeometry {
    center: Vec3,
    axes: [Vec3; 3],
    half_extents: Vec3,
}

impl BoxGeometry {
    fn new(collider: &Collider, body: &RigidBody, half_extents: Vec3) -> Self {
        let world = collider.world_transform(&body.transform);
        let axes = [
            world.rotation * Vec3::X,
            world.rotation * Vec3::Y,
            world.rotation * Vec3::Z,
        ];
        let scaled_half_extents = half_extents * world.scale.abs();
        Self {
            center: world.position,
            axes,
            half_extents: scaled_half_extents,
        }
    }

    fn project_radius(&self, axis: Vec3) -> f32 {
        self.half_extents.x * self.axes[0].dot(axis).abs()
            + self.half_extents.y * self.axes[1].dot(axis).abs()
            + self.half_extents.z * self.axes[2].dot(axis).abs()
    }
}

struct FaceDescriptor {
    reference_is_a: bool,
    axis_index: usize,
    face_sign: f32,
}

fn find_face_axis(box_a: &BoxGeometry, box_b: &BoxGeometry) -> Option<FaceDescriptor> {
    let mut best_overlap = f32::MAX;
    let mut descriptor = None;
    let center_diff = box_b.center - box_a.center;

    for axis_index in 0..3 {
        let axis = box_a.axes[axis_index];
        let overlap = overlap_on_axis(axis, box_a.half_extents[axis_index], center_diff, box_b);
        if overlap < 0.0 {
            return None;
        }
        if overlap < best_overlap {
            best_overlap = overlap;
            let separation = center_diff.dot(axis);
            let face_sign = if separation >= 0.0 { 1.0 } else { -1.0 };
            descriptor = Some(FaceDescriptor {
                reference_is_a: true,
                axis_index,
                face_sign,
            });
        }
    }

    let reverse_diff = -center_diff;
    for axis_index in 0..3 {
        let axis = box_b.axes[axis_index];
        let overlap = overlap_on_axis(axis, box_b.half_extents[axis_index], reverse_diff, box_a);
        if overlap < 0.0 {
            return None;
        }
        if overlap < best_overlap {
            best_overlap = overlap;
            let separation = reverse_diff.dot(axis);
            let face_sign = if separation >= 0.0 { 1.0 } else { -1.0 };
            descriptor = Some(FaceDescriptor {
                reference_is_a: false,
                axis_index,
                face_sign,
            });
        }
    }

    descriptor
}

fn overlap_on_axis(
    axis: Vec3,
    reference_half_extent: f32,
    center_diff: Vec3,
    other: &BoxGeometry,
) -> f32 {
    let separation = center_diff.dot(axis).abs();
    let other_radius = other.project_radius(axis);
    reference_half_extent + other_radius - separation
}

fn find_incident_face(box_geom: &BoxGeometry, normal: Vec3) -> (usize, f32) {
    let mut min_dot = f32::MAX;
    let mut face_index = 0;
    for i in 0..3 {
        let dot = box_geom.axes[i].dot(normal);
        if dot < min_dot {
            min_dot = dot;
            face_index = i;
        }
    }
    let sign = if min_dot <= 0.0 { 1.0 } else { -1.0 };
    (face_index, sign)
}

fn face_vertices(box_geom: &BoxGeometry, face_index: usize, sign: f32) -> [Vec3; 4] {
    let face_center =
        box_geom.center + box_geom.axes[face_index] * box_geom.half_extents[face_index] * sign;
    let (u_idx, v_idx) = face_tangent_indices(face_index);
    let u = box_geom.axes[u_idx] * box_geom.half_extents[u_idx];
    let v = box_geom.axes[v_idx] * box_geom.half_extents[v_idx];
    [
        face_center + u + v,
        face_center + u - v,
        face_center - u + v,
        face_center - u - v,
    ]
}

fn face_tangent_indices(face_index: usize) -> (usize, usize) {
    match face_index {
        0 => (1, 2),
        1 => (0, 2),
        _ => (0, 1),
    }
}

fn dedup_points(points: Vec<Vec3>) -> Vec<Vec3> {
    let mut unique: Vec<Vec3> = Vec::new();
    const EPS: f32 = 1e-4;
    for point in points {
        if unique
            .iter()
            .all(|existing| (*existing - point).length_squared() > EPS * EPS)
        {
            unique.push(point);
        }
    }
    unique
}

fn centroid(points: &[Vec3]) -> Vec3 {
    if points.is_empty() {
        return Vec3::ZERO;
    }
    let sum: Vec3 = points.iter().copied().sum();
    sum / points.len() as f32
}

fn select_best_points(
    mut points: Vec<ContactPoint>,
    max_points: usize,
    normal: Vec3,
) -> Vec<ContactPoint> {
    if max_points == 0 {
        return Vec::new();
    }
    if points.len() <= max_points {
        return points;
    }

    points.sort_unstable_by(depth_then_feature);
    let mut selected: Vec<ContactPoint> = Vec::with_capacity(max_points);
    selected.push(points[0].clone());

    while selected.len() < max_points {
        let mut best_idx = None;
        let mut best_score = f32::NEG_INFINITY;

        for (idx, candidate) in points.iter().enumerate() {
            if selected
                .iter()
                .any(|existing| existing.feature_id == candidate.feature_id)
            {
                continue;
            }

            let spread = tangent_distance(candidate, &selected, normal);
            let score = spread + candidate.depth;

            match score.partial_cmp(&best_score) {
                Some(Ordering::Greater) => {
                    best_score = score;
                    best_idx = Some(idx);
                }
                Some(Ordering::Equal) => {
                    if let Some(prev_idx) = best_idx {
                        let current_best = &points[prev_idx];
                        if depth_then_feature(candidate, current_best) == Ordering::Less {
                            best_idx = Some(idx);
                        }
                    }
                }
                _ => {}
            }
        }

        if let Some(idx) = best_idx {
            selected.push(points[idx].clone());
        } else {
            break;
        }
    }

    selected.sort_unstable_by(depth_then_feature);
    selected.truncate(max_points);
    selected
}

fn depth_then_feature(a: &ContactPoint, b: &ContactPoint) -> Ordering {
    b.depth
        .partial_cmp(&a.depth)
        .unwrap_or(Ordering::Equal)
        .then_with(|| a.feature_id.cmp(&b.feature_id))
}

fn tangent_distance(candidate: &ContactPoint, selected: &[ContactPoint], normal: Vec3) -> f32 {
    if selected.is_empty() {
        return 0.0;
    }
    let mut min_dist = f32::MAX;
    for other in selected {
        let diff = candidate.world_point - other.world_point;
        let tangent = diff - normal * diff.dot(normal);
        let dist = tangent.length();
        if dist < min_dist {
            min_dist = dist;
        }
    }
    if min_dist.is_finite() {
        min_dist
    } else {
        0.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ManifoldKey {
    a: EntityId,
    b: EntityId,
}

impl ManifoldKey {
    fn new(a: EntityId, b: EntityId) -> Self {
        if a.index() < b.index() {
            Self { a, b }
        } else {
            Self { a: b, b: a }
        }
    }
}

pub struct ManifoldCache {
    manifolds: HashMap<ManifoldKey, PersistentManifold>,
    frame: u32,
    debug_hook: Option<Box<ManifoldDebugHook>>,
}

impl ManifoldCache {
    pub fn new() -> Self {
        Self {
            manifolds: HashMap::new(),
            frame: 0,
            debug_hook: None,
        }
    }

    pub fn begin_frame(&mut self, frame: u32) {
        self.frame = frame;
    }

    pub fn update_pair(
        &mut self,
        mut manifold: ContactManifold,
        collider_a: &Collider,
        collider_b: &Collider,
        rigid_a: &RigidBody,
        rigid_b: &RigidBody,
    ) -> Vec<Contact> {
        let (body_a, body_b) = (rigid_a.id, rigid_b.id);
        let key = ManifoldKey::new(body_a, body_b);
        let wants_debug = self.debug_hook.is_some();

        // If this manifold doesn't have a simplex yet (e.g. it was just created by generate()),
        // try to get one from the existing persistent manifold.
        if manifold.simplex.is_none() {
            if let Some(entry) = self.manifolds.get(&key) {
                if let Some((contact, simplex)) = NarrowPhase::collide(
                    collider_a,
                    rigid_a,
                    collider_b,
                    rigid_b,
                    entry.simplex.as_deref(),
                ) {
                    manifold.normal = contact.normal;
                    manifold.points = vec![RawContactPoint {
                        point: contact.point,
                        depth: contact.depth,
                        feature_id: contact.feature_id,
                    }];
                    manifold.simplex = Some(simplex);
                }
            }
        }
        let (contacts, debug_snapshot) = {
            let entry = self
                .manifolds
                .entry(key)
                .or_insert_with(|| PersistentManifold::new(body_a, body_b));
            entry.update(
                manifold.normal,
                &manifold.points,
                rigid_a,
                rigid_b,
                self.frame,
                manifold.simplex,
            );
            let contacts = entry.to_contacts();
            let snapshot = if wants_debug {
                Some(entry.debug_snapshot(self.frame))
            } else {
                None
            };
            (contacts, snapshot)
        };

        if let (Some(hook), Some(snapshot)) = (self.debug_hook.as_mut(), debug_snapshot) {
            hook(&snapshot);
        }

        contacts
    }

    pub fn record_contact(&mut self, contact: &Contact) {
        let key = ManifoldKey::new(contact.body_a, contact.body_b);
        let entry = self
            .manifolds
            .entry(key)
            .or_insert_with(|| PersistentManifold::new(contact.body_a, contact.body_b));
        // Local coordinates are omitted for CCD-generated contacts due to the absence of body references.
        // Warm-starting is minimized by assuming zero local contact offsets for these cases.
        entry.last_frame = self.frame;
        if entry.points.len() >= MAX_MANIFOLD_POINTS {
            entry.points.clear();
        }
        entry.material = contact.material;
        entry.points.push(ContactPoint {
            feature_id: contact.feature_id,
            world_point: contact.point,
            local_a: Vec3::ZERO,
            local_b: Vec3::ZERO,
            depth: contact.depth,
            normal_impulse: contact.accumulated_normal_impulse,
            tangent_impulse: contact.accumulated_tangent_impulse,
            rolling_impulse: contact.accumulated_rolling_impulse,
            torsional_impulse: contact.accumulated_torsional_impulse,
        });

        if let Some(snapshot) = self.debug_hook.as_ref().and_then(|_| {
            self.manifolds
                .get(&key)
                .map(|m| m.debug_snapshot(self.frame))
        }) {
            if let Some(hook) = self.debug_hook.as_mut() {
                hook(&snapshot);
            }
        }
    }

    pub fn apply_impulses(&mut self, contacts: &[Contact]) {
        if contacts.is_empty() {
            return;
        }
        let wants_debug = self.debug_hook.is_some();
        let mut updated_keys = Vec::new();
        for contact in contacts {
            let key = ManifoldKey::new(contact.body_a, contact.body_b);
            if let Some(manifold) = self.manifolds.get_mut(&key) {
                manifold.apply_impulses(contact);
                if wants_debug {
                    updated_keys.push(key);
                }
            }
        }

        if wants_debug {
            updated_keys.sort_unstable_by_key(|k| (k.a.index(), k.b.index()));
            updated_keys.dedup_by(|a, b| a.a == b.a && a.b == b.b);
            for key in updated_keys {
                let snapshot = self
                    .manifolds
                    .get(&key)
                    .map(|m| m.debug_snapshot(self.frame));
                if let (Some(hook), Some(snapshot)) = (self.debug_hook.as_mut(), snapshot) {
                    hook(&snapshot);
                }
            }
        }
    }

    pub fn prune_stale(&mut self) {
        let frame = self.frame;
        self.manifolds
            .retain(|_, manifold| frame.saturating_sub(manifold.last_frame) <= MANIFOLD_MAX_AGE);
    }

    pub fn set_debug_hook<F>(&mut self, hook: Option<F>)
    where
        F: Fn(&ManifoldDebugInfo) + Send + Sync + 'static,
    {
        self.debug_hook = hook.map(|f| Box::new(f) as Box<_>);
    }

    pub fn debug_snapshots(&self) -> Vec<ManifoldDebugInfo> {
        self.manifolds
            .values()
            .map(|manifold| manifold.debug_snapshot(self.frame))
            .collect()
    }
}

impl Default for ManifoldCache {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for ManifoldCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ManifoldCache")
            .field("manifolds", &self.manifolds.len())
            .field("frame", &self.frame)
            .finish()
    }
}
