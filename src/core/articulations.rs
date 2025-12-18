use super::types::Transform;
use crate::utils::allocator::EntityId;
use glam::{Mat3, Vec3};

/// Type of joint connecting a link to its parent in reduced coordinates.
#[derive(Debug, Clone, Copy)]
pub enum JointType {
    /// 1-DOF Rotational joint.
    Revolute { axis: Vec3 },
    /// 1-DOF Translational joint.
    Prismatic { axis: Vec3 },
    /// 3-DOF Rotational joint.
    Spherical,
    /// 0-DOF Rigid connection.
    Fixed,
}

impl JointType {
    /// Returns the number of degrees of freedom for this joint type.
    pub fn dofs(&self) -> usize {
        match self {
            JointType::Revolute { .. } => 1,
            JointType::Prismatic { .. } => 1,
            JointType::Spherical => 3,
            JointType::Fixed => 0,
        }
    }

    /// Calculates the local transform across the joint given its coordinates.
    pub fn transform(&self, q: &[f32]) -> Transform {
        match self {
            JointType::Revolute { axis } => Transform {
                rotation: glam::Quat::from_axis_angle(*axis, q[0]),
                ..Transform::default()
            },
            JointType::Prismatic { axis } => Transform {
                position: *axis * q[0],
                ..Transform::default()
            },
            JointType::Spherical => {
                // q[0..3] as exponential coordinates or euler? Let's use Euler for simplicity in MVP.
                Transform {
                    rotation: glam::Quat::from_euler(glam::EulerRot::XYZ, q[0], q[1], q[2]),
                    ..Transform::default()
                }
            }
            JointType::Fixed => Transform::default(),
        }
    }

    /// Returns the motion subspace matrix S. For 1-DOF, this is a 6x1 spatial vector.
    /// For multi-DOF, it would be a matrix. We'll return a Vec of SpatialVecs.
    pub fn spatial_subspace(&self) -> Vec<crate::utils::spatial::SpatialVec> {
        use crate::utils::spatial::SpatialVec;
        match self {
            JointType::Revolute { axis } => vec![SpatialVec::new(*axis, Vec3::ZERO)],
            JointType::Prismatic { axis } => vec![SpatialVec::new(Vec3::ZERO, *axis)],
            JointType::Spherical => vec![
                SpatialVec::new(Vec3::X, Vec3::ZERO),
                SpatialVec::new(Vec3::Y, Vec3::ZERO),
                SpatialVec::new(Vec3::Z, Vec3::ZERO),
            ],
            JointType::Fixed => vec![],
        }
    }
}

/// A single node in the articulated body tree.
#[derive(Debug, Clone)]
pub struct Link {
    pub name: String,
    /// Index of the parent link. None if this is the root link.
    pub parent_idx: Option<usize>,
    /// The joint connecting this link to its parent.
    pub joint_type: JointType,
    /// The offset of the joint (q) indices in the multibody state vectors.
    pub q_offset: usize,

    /// Static transform from parent link frame to this link's joint frame (at q=0).
    pub parent_to_joint: Transform,

    /// Physical properties of the link.
    pub mass: f32,
    /// Center of mass offset from the link frame (usually joint frame).
    pub com_offset: Vec3,
    /// Rotational inertia tensor about the center of mass.
    pub inertia: Mat3,

    /// Optional mapping to a maximal coordinate body for collision detection.
    pub collision_body: Option<EntityId>,
}

impl Link {
    pub fn new(name: &str, parent: Option<usize>, joint: JointType) -> Self {
        Self {
            name: name.into(),
            parent_idx: parent,
            joint_type: joint,
            q_offset: 0,
            parent_to_joint: Transform::default(),
            mass: 1.0,
            com_offset: Vec3::ZERO,
            inertia: Mat3::IDENTITY,
            collision_body: None,
        }
    }
}

/// A collection of links forming a tree structure for reduced-coordinate dynamics.
#[derive(Debug, Clone)]
pub struct Multibody {
    pub id: EntityId,
    /// Links ordered such that a parent always appears before its children.
    pub links: Vec<Link>,

    /// Total degrees of freedom across all joints.
    pub total_dofs: usize,

    /// Generalized positions (q).
    pub q: Vec<f32>,
    /// Generalized velocities (u or dq).
    pub dq: Vec<f32>,
    /// Generalized accelerations (ddq).
    pub ddq: Vec<f32>,
    /// Generalized forces (torques applied at joints).
    pub tau: Vec<f32>,

    // Computed states
    pub world_transforms: Vec<Transform>,
    pub spatial_velocities: Vec<crate::utils::spatial::SpatialVec>,
}

impl Multibody {
    pub fn new(id: EntityId) -> Self {
        Self {
            id,
            links: Vec::new(),
            total_dofs: 0,
            q: Vec::new(),
            dq: Vec::new(),
            ddq: Vec::new(),
            tau: Vec::new(),
            world_transforms: Vec::new(),
            spatial_velocities: Vec::new(),
        }
    }

    /// Adds a link to the multibody and allocates space for its DOFs.
    pub fn add_link(&mut self, mut link: Link) -> usize {
        let idx = self.links.len();
        link.q_offset = self.total_dofs;
        let dofs = link.joint_type.dofs();
        self.total_dofs += dofs;

        // Resize state vectors
        self.q.resize(self.total_dofs, 0.0);
        self.dq.resize(self.total_dofs, 0.0);
        self.ddq.resize(self.total_dofs, 0.0);
        self.tau.resize(self.total_dofs, 0.0);

        self.world_transforms.push(Transform::default());
        self.spatial_velocities
            .push(crate::utils::spatial::SpatialVec::default());

        self.links.push(link);
        idx
    }

    /// Updates world transforms and spatial velocities based on current q and dq.
    pub fn update_kinematics(&mut self) {
        use crate::utils::spatial::SpatialVec;

        for i in 0..self.links.len() {
            let link = &self.links[i];
            let q_slice = &self.q[link.q_offset..link.q_offset + link.joint_type.dofs()];
            let dq_slice = &self.dq[link.q_offset..link.q_offset + link.joint_type.dofs()];

            let local_x = link.joint_type.transform(q_slice);
            let rel_x = link.parent_to_joint.combine(&local_x);

            if let Some(p_idx) = link.parent_idx {
                let parent_world_x = self.world_transforms[p_idx];
                self.world_transforms[i] = parent_world_x.combine(&rel_x);

                // v_i = v_parent + s_i * dq_i
                let s_vectors = link.joint_type.spatial_subspace();
                let mut v_joint = SpatialVec::default();
                for (j, s) in s_vectors.iter().enumerate() {
                    v_joint = v_joint + (*s * dq_slice[j]);
                }

                // Transform parent velocity to current link frame?
                // Featherstone usually works in link frames for simplicity.
                // We'll keep it simple for now and assume world space or handle xforms later.
                // For MVP, let's just add velocities if aligned.
                self.spatial_velocities[i] = self.spatial_velocities[p_idx] + v_joint;
            } else {
                self.world_transforms[i] = rel_x;
                let s_vectors = link.joint_type.spatial_subspace();
                let mut v_joint = SpatialVec::default();
                for (j, s) in s_vectors.iter().enumerate() {
                    v_joint = v_joint + (*s * dq_slice[j]);
                }
                self.spatial_velocities[i] = v_joint;
            }
        }
    }
}
