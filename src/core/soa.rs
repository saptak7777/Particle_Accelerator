use crate::core::rigidbody::RigidBody;
use crate::core::types::{MassProperties, Material, Transform, Velocity};
use crate::utils::allocator::EntityId;
use glam::{Mat3, Vec3};
use std::collections::VecDeque;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BodyFlags {
    pub is_static: bool,
    pub is_kinematic: bool,
    pub is_awake: bool,
    pub is_enabled: bool,
}

impl Default for BodyFlags {
    fn default() -> Self {
        Self {
            is_static: false,
            is_kinematic: false,
            is_awake: true,
            is_enabled: true,
        }
    }
}

/// Structure-of-Arrays storage for Rigid Bodies.
/// Replaces Arena<RigidBody> for better cache locality.
#[derive(Default)]
pub struct BodiesSoA {
    // Generational memory management
    pub generations: Vec<u32>,
    pub free_list: VecDeque<usize>,

    // Core state (Dense vectors, but indexed sparsely by ID)
    // Valid data is maintained even in allocated "free" slots to avoid Option wrappers.
    // Validity is checked via generations.
    pub ids: Vec<EntityId>,
    pub transforms: Vec<Transform>,
    pub velocities: Vec<Velocity>,
    pub accelerations: Vec<Vec3>,

    // Mass properties
    pub inverse_masses: Vec<f32>,
    pub inverse_inertias: Vec<Mat3>,
    pub mass_properties: Vec<MassProperties>,

    // Material & Config
    pub materials: Vec<Material>,
    pub flags: Vec<BodyFlags>,
    pub gravity_scales: Vec<f32>,
    pub linear_dampings: Vec<f32>,
    pub angular_dampings: Vec<f32>,
}

impl BodiesSoA {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, body: RigidBody) -> EntityId {
        if let Some(index) = self.free_list.pop_front() {
            let generation = self.generations[index];
            let id = EntityId::new(index, generation);
            self.write_at(index, id, body);
            id
        } else {
            let index = self.generations.len();
            self.generations.push(0);
            let id = EntityId::new(index, 0);
            self.push(id, body);
            id
        }
    }

    pub fn remove(&mut self, id: EntityId) -> Option<()> {
        if !self.is_valid(id) {
            return None;
        }
        let index = id.index();
        // Increment generation to invalidate old IDs
        self.generations[index] = self.generations[index].wrapping_add(1);
        self.free_list.push_back(index);
        // Active status is not verified as the layout does not utilize Option wrappers.
        Some(())
    }

    pub fn get(&self, id: EntityId) -> Option<BodyRef<'_>> {
        if self.is_valid(id) {
            let idx = id.index();
            Some(BodyRef {
                soa: self,
                index: idx,
            })
        } else {
            None
        }
    }

    pub fn get_mut(&mut self, id: EntityId) -> Option<BodyMut<'_>> {
        if !self.is_valid(id) {
            return None;
        }
        let i = id.index();
        // SAFE: A single mutable reference is returned to the caller.
        Some(BodyMut {
            id,
            transform: &mut self.transforms[i],
            velocity: &mut self.velocities[i],
            acceleration: &mut self.accelerations[i],
            inverse_mass: &mut self.inverse_masses[i],
            inverse_inertia: &mut self.inverse_inertias[i],
            mass_properties: &mut self.mass_properties[i],
            material: &mut self.materials[i],
            flags: &mut self.flags[i],
            gravity_scale: &mut self.gravity_scales[i],
            linear_damping: &mut self.linear_dampings[i],
            angular_damping: &mut self.angular_dampings[i],
        })
    }

    pub fn get2_mut(
        &mut self,
        id_a: EntityId,
        id_b: EntityId,
    ) -> Option<(BodyMut<'_>, BodyMut<'_>)> {
        if id_a == id_b || !self.is_valid(id_a) || !self.is_valid(id_b) {
            return None;
        }

        let i_a = id_a.index();
        let i_b = id_b.index();

        // Use raw pointers to form two mutable refs
        unsafe {
            let ptr_trans = self.transforms.as_mut_ptr();
            let ptr_vel = self.velocities.as_mut_ptr();
            let ptr_acc = self.accelerations.as_mut_ptr();
            let ptr_im = self.inverse_masses.as_mut_ptr();
            let ptr_ii = self.inverse_inertias.as_mut_ptr();
            let ptr_mp = self.mass_properties.as_mut_ptr();
            let ptr_mat = self.materials.as_mut_ptr();
            let ptr_flags = self.flags.as_mut_ptr();
            let ptr_grav = self.gravity_scales.as_mut_ptr();
            let ptr_ld = self.linear_dampings.as_mut_ptr();
            let ptr_ad = self.angular_dampings.as_mut_ptr();

            let a = BodyMut {
                id: id_a,
                transform: &mut *ptr_trans.add(i_a),
                velocity: &mut *ptr_vel.add(i_a),
                acceleration: &mut *ptr_acc.add(i_a),
                inverse_mass: &mut *ptr_im.add(i_a),
                inverse_inertia: &mut *ptr_ii.add(i_a),
                mass_properties: &mut *ptr_mp.add(i_a),
                material: &mut *ptr_mat.add(i_a),
                flags: &mut *ptr_flags.add(i_a),
                gravity_scale: &mut *ptr_grav.add(i_a),
                linear_damping: &mut *ptr_ld.add(i_a),
                angular_damping: &mut *ptr_ad.add(i_a),
            };

            let b = BodyMut {
                id: id_b,
                transform: &mut *ptr_trans.add(i_b),
                velocity: &mut *ptr_vel.add(i_b),
                acceleration: &mut *ptr_acc.add(i_b),
                inverse_mass: &mut *ptr_im.add(i_b),
                inverse_inertia: &mut *ptr_ii.add(i_b),
                mass_properties: &mut *ptr_mp.add(i_b),
                material: &mut *ptr_mat.add(i_b),
                flags: &mut *ptr_flags.add(i_b),
                gravity_scale: &mut *ptr_grav.add(i_b),
                linear_damping: &mut *ptr_ld.add(i_b),
                angular_damping: &mut *ptr_ad.add(i_b),
            };

            Some((a, b))
        }
    }

    pub fn len(&self) -> usize {
        self.generations.len() - self.free_list.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn iter(&self) -> impl Iterator<Item = BodyRef<'_>> {
        self.generations
            .iter()
            .enumerate()
            .filter_map(move |(idx, &gen)| {
                if self.ids[idx].generation() == gen {
                    Some(BodyRef {
                        soa: self,
                        index: idx,
                    })
                } else {
                    None
                }
            })
    }

    // Safe internal split-borrow helper?
    // Validation is omitted as the SoA layout does not use Option wrappers.

    // A custom iterator utilizing raw pointers is used for SoA fields
    // to verify safety dynamically and ensure index uniqueness (valid for 0..len).

    pub fn iter_mut(&mut self) -> SoAIterMut<'_> {
        SoAIterMut::new(self)
    }
}

/// Mutable proxy that behaves like `RigidBody`.
/// Holds disjoint mutable borrows to the SoA vectors.
pub struct BodyProxyMut<'a> {
    pub id: EntityId,
    pub transform: &'a mut Transform,
    pub velocity: &'a mut Velocity,
    pub acceleration: &'a mut Vec3,
    pub inverse_mass: &'a mut f32,
    pub inverse_inertia: &'a mut Mat3,
    pub mass_properties: &'a mut MassProperties,
    pub material: &'a mut Material,
    pub flags: &'a mut BodyFlags,
    pub gravity_scale: &'a mut f32,
    pub linear_damping: &'a mut f32,
    pub angular_damping: &'a mut f32,
}

impl<'a> BodyProxyMut<'a> {
    pub fn apply_impulse(&mut self, impulse: Vec3, position: Vec3) {
        if self.flags.is_static {
            return;
        }
        self.velocity.linear += impulse * (*self.inverse_mass);
        let torque = (position - self.transform.position).cross(impulse);
        self.velocity.angular += (*self.inverse_inertia) * torque;
        self.flags.is_awake = true;
    }

    pub fn apply_force(&mut self, force: Vec3) {
        if self.flags.is_static {
            return;
        }
        *self.acceleration += force * (*self.inverse_mass);
        self.flags.is_awake = true;
    }

    pub fn set_velocity(&mut self, linear: Vec3, angular: Vec3) {
        self.velocity.linear = linear;
        self.velocity.angular = angular;
    }

    // Accessors to match RigidBody API
    pub fn is_static(&self) -> bool {
        self.flags.is_static
    }
    pub fn is_kinematic(&self) -> bool {
        self.flags.is_kinematic
    }
    pub fn is_awake(&self) -> bool {
        self.flags.is_awake
    }
    pub fn is_enabled(&self) -> bool {
        self.flags.is_enabled
    }

    pub fn set_awake(&mut self, awake: bool) {
        self.flags.is_awake = awake;
    }

    pub fn to_rigid_body(&self) -> RigidBody {
        let mut body = RigidBody::new(self.id);
        body.transform = *self.transform;
        body.velocity = *self.velocity;
        body.acceleration = *self.acceleration;
        body.inverse_mass = *self.inverse_mass;
        body.inverse_inertia = *self.inverse_inertia;
        body.mass_properties = *self.mass_properties;
        body.material = *self.material;

        body.is_static = self.flags.is_static;
        body.is_kinematic = self.flags.is_kinematic;
        body.is_awake = self.flags.is_awake;
        body.is_enabled = self.flags.is_enabled;

        body.gravity_scale = *self.gravity_scale;
        body.linear_velocity_damping = *self.linear_damping;
        body.angular_velocity_damping = *self.angular_damping;

        body
    }
}

pub struct SoAIterMut<'a> {
    // Raw pointers and length are maintained to facilitate split borrows.
    // SAFETY: The iterator implementation guarantees the yielding of unique indices.
    len: usize,
    pos: usize,
    generations: &'a [u32],
    ids: &'a [EntityId],

    ptr_transforms: *mut Transform,
    ptr_velocities: *mut Velocity,
    ptr_accelerations: *mut Vec3,
    ptr_inv_mass: *mut f32,
    ptr_inv_inertia: *mut Mat3,
    ptr_mass_props: *mut MassProperties,
    ptr_materials: *mut Material,
    ptr_flags: *mut BodyFlags,
    ptr_gravity: *mut f32,
    ptr_linear_damping: *mut f32,
    ptr_angular_damping: *mut f32,

    _marker: std::marker::PhantomData<&'a mut BodiesSoA>,
}

impl<'a> SoAIterMut<'a> {
    fn new(soa: &'a mut BodiesSoA) -> Self {
        Self {
            len: soa.generations.len(),
            pos: 0,
            generations: &soa.generations,
            ids: &soa.ids,

            ptr_transforms: soa.transforms.as_mut_ptr(),
            ptr_velocities: soa.velocities.as_mut_ptr(),
            ptr_accelerations: soa.accelerations.as_mut_ptr(),
            ptr_inv_mass: soa.inverse_masses.as_mut_ptr(),
            ptr_inv_inertia: soa.inverse_inertias.as_mut_ptr(),
            ptr_mass_props: soa.mass_properties.as_mut_ptr(),
            ptr_materials: soa.materials.as_mut_ptr(),
            ptr_flags: soa.flags.as_mut_ptr(),
            ptr_gravity: soa.gravity_scales.as_mut_ptr(),
            ptr_linear_damping: soa.linear_dampings.as_mut_ptr(),
            ptr_angular_damping: soa.angular_dampings.as_mut_ptr(),

            _marker: std::marker::PhantomData,
        }
    }
}

impl<'a> Iterator for SoAIterMut<'a> {
    type Item = BodyProxyMut<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        while self.pos < self.len {
            let i = self.pos;
            self.pos += 1;

            // Check generation validity
            // NOTE: The 'ids' vector is assumed to be synchronized, storing IDs with valid generations.
            // at the time of insertion/update.
            if self.generations[i] != self.ids[i].generation() {
                continue;
            }

            unsafe {
                return Some(BodyProxyMut {
                    id: self.ids[i],
                    transform: &mut *self.ptr_transforms.add(i),
                    velocity: &mut *self.ptr_velocities.add(i),
                    acceleration: &mut *self.ptr_accelerations.add(i),
                    inverse_mass: &mut *self.ptr_inv_mass.add(i),
                    inverse_inertia: &mut *self.ptr_inv_inertia.add(i),
                    mass_properties: &mut *self.ptr_mass_props.add(i),
                    material: &mut *self.ptr_materials.add(i),
                    flags: &mut *self.ptr_flags.add(i),
                    gravity_scale: &mut *self.ptr_gravity.add(i),
                    linear_damping: &mut *self.ptr_linear_damping.add(i),
                    angular_damping: &mut *self.ptr_angular_damping.add(i),
                });
            }
        }
        None
    }
}

// Needed to replace BodyMut
pub type BodyMut<'a> = BodyProxyMut<'a>;

impl BodiesSoA {
    pub fn is_valid(&self, id: EntityId) -> bool {
        if id.index() >= self.generations.len() {
            return false;
        }
        self.generations[id.index()] == id.generation()
    }

    pub fn write_at(&mut self, index: usize, id: EntityId, body: RigidBody) {
        self.ids[index] = id;
        self.transforms[index] = body.transform;
        self.velocities[index] = body.velocity;
        self.accelerations[index] = body.acceleration;
        self.inverse_masses[index] = body.inverse_mass;
        self.inverse_inertias[index] = body.inverse_inertia;
        self.mass_properties[index] = body.mass_properties;
        self.materials[index] = body.material;
        self.flags[index] = BodyFlags {
            is_static: body.is_static,
            is_kinematic: body.is_kinematic,
            is_awake: body.is_awake,
            is_enabled: body.is_enabled,
        };
        self.gravity_scales[index] = body.gravity_scale;
        self.linear_dampings[index] = body.linear_velocity_damping;
        self.angular_dampings[index] = body.angular_velocity_damping;
    }

    pub fn push(&mut self, id: EntityId, body: RigidBody) {
        self.ids.push(id);
        self.transforms.push(body.transform);
        self.velocities.push(body.velocity);
        self.accelerations.push(body.acceleration);
        self.inverse_masses.push(body.inverse_mass);
        self.inverse_inertias.push(body.inverse_inertia);
        self.mass_properties.push(body.mass_properties);
        self.materials.push(body.material);
        self.flags.push(BodyFlags {
            is_static: body.is_static,
            is_kinematic: body.is_kinematic,
            is_awake: body.is_awake,
            is_enabled: body.is_enabled,
        });
        self.gravity_scales.push(body.gravity_scale);
        self.linear_dampings.push(body.linear_velocity_damping);
        self.angular_dampings.push(body.angular_velocity_damping);
    }
}

pub struct BodyRef<'a> {
    soa: &'a BodiesSoA,
    index: usize,
}

impl<'a> BodyRef<'a> {
    pub fn id(&self) -> EntityId {
        self.soa.ids[self.index]
    }
    pub fn transform(&self) -> &Transform {
        &self.soa.transforms[self.index]
    }
    pub fn velocity(&self) -> &Velocity {
        &self.soa.velocities[self.index]
    }
    pub fn inverse_mass(&self) -> f32 {
        self.soa.inverse_masses[self.index]
    }
    pub fn inverse_inertia(&self) -> Mat3 {
        self.soa.inverse_inertias[self.index]
    }
    pub fn is_static(&self) -> bool {
        self.soa.flags[self.index].is_static
    }
    pub fn is_kinematic(&self) -> bool {
        self.soa.flags[self.index].is_kinematic
    }
    pub fn is_awake(&self) -> bool {
        self.soa.flags[self.index].is_awake
    }
    pub fn is_enabled(&self) -> bool {
        self.soa.flags[self.index].is_enabled
    }
    pub fn material(&self) -> &Material {
        &self.soa.materials[self.index]
    }

    pub fn to_rigid_body(&self) -> RigidBody {
        let mut body = RigidBody::new(self.id());
        body.transform = *self.transform();
        body.velocity = *self.velocity();
        body.acceleration = self.soa.accelerations[self.index];
        body.inverse_mass = self.inverse_mass();
        body.inverse_inertia = self.inverse_inertia();
        body.mass_properties = self.soa.mass_properties[self.index];
        body.material = *self.material();

        let flags = self.soa.flags[self.index];
        body.is_static = flags.is_static;
        body.is_kinematic = flags.is_kinematic;
        body.is_awake = flags.is_awake;
        body.is_enabled = flags.is_enabled;

        body.gravity_scale = self.soa.gravity_scales[self.index];
        body.linear_velocity_damping = self.soa.linear_dampings[self.index];
        body.angular_velocity_damping = self.soa.angular_dampings[self.index];

        body
    }
}
