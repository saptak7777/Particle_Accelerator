use crate::core::articulations::Multibody;
use crate::utils::spatial::{transform_force, transform_motion, SpatialMat, SpatialVec};

pub struct ABASolver;

impl ABASolver {
    /// Solves for generalized accelerations (ddq) using Featherstone's ABA.
    pub fn solve(mb: &mut Multibody, gravity: glam::Vec3) {
        let n = mb.links.len();
        if n == 0 {
            return;
        }

        // State storage for the 3 passes
        let mut v = vec![SpatialVec::default(); n];
        let mut c = vec![SpatialVec::default(); n];
        let mut i_a = vec![SpatialMat::default(); n];
        let mut p_a = vec![SpatialVec::default(); n];

        let mut u_vec = vec![SpatialVec::default(); n]; // U_i = I_i^A * S_i
        let mut d_inv = vec![0.0; n]; // For 1-DOF
        let mut force_u = vec![0.0; n]; // u_i = tau_i - S_i^T * p_i^A

        // --- Pass 1: Outward ---
        // Calculate velocities and bias forces
        for i in 0..n {
            let link = &mb.links[i];
            let q_slice = &mb.q[link.q_offset..link.q_offset + link.joint_type.dofs()];
            let dq_slice = &mb.dq[link.q_offset..link.q_offset + link.joint_type.dofs()];

            let s_vectors = link.joint_type.spatial_subspace();
            let mut v_joint = SpatialVec::default();
            for (j, s) in s_vectors.iter().enumerate() {
                v_joint = v_joint + (*s * dq_slice[j]);
            }

            if let Some(p_idx) = link.parent_idx {
                // X_i_parent = parent_to_joint * joint(q)
                // v_i = X_i_parent * v_parent + v_joint
                let local_x = link.joint_type.transform(q_slice);
                let x_rel = link.parent_to_joint.combine(&local_x);

                v[i] = transform_motion(v[p_idx], x_rel.rotation, x_rel.position) + v_joint;
                c[i] = v[i].cross_motion(&v_joint);
            } else {
                v[i] = v_joint;
                c[i] = SpatialVec::default();
            }

            let i_link = crate::utils::spatial::SpatialInertia::new(
                link.mass,
                link.com_offset,
                link.inertia,
            )
            .to_mat();
            i_a[i] = i_link;
            p_a[i] = v[i].cross_force(&i_link.mul_vec(v[i]));
        }

        // --- Pass 2: Inward ---
        for i in (0..n).rev() {
            let link = &mb.links[i];
            let s_vectors = link.joint_type.spatial_subspace();

            if s_vectors.len() == 1 {
                let s = s_vectors[0];
                let u_i = i_a[i].mul_vec(s);
                u_vec[i] = u_i;

                let d = s.dot(&u_i);
                let di = if d.abs() > 1e-6 { 1.0 / d } else { 0.0 };
                d_inv[i] = di;

                let q_offset = link.q_offset;
                force_u[i] = mb.tau[q_offset] - s.dot(&p_a[i]);

                if let Some(p_idx) = link.parent_idx {
                    // Propagate to parent
                    let i_reduced = i_a[i] - SpatialMat::outer_product(u_i) * di;
                    let p_reduced = p_a[i] + i_a[i].mul_vec(c[i]) + u_i * (di * force_u[i]);

                    let q_offset = mb.links[i].q_offset;
                    let local_x = mb.links[i]
                        .joint_type
                        .transform(&mb.q[q_offset..q_offset + 1]);
                    let x_rel = mb.links[i].parent_to_joint.combine(&local_x);

                    let inv_rot = x_rel.rotation.inverse();
                    let inv_pos = -(inv_rot * x_rel.position);

                    p_a[p_idx] = p_a[p_idx] + transform_force(p_reduced, inv_rot, inv_pos);
                    i_a[p_idx] = i_a[p_idx] + i_reduced;
                }
            }
        }

        // --- Pass 3: Outward ---
        let mut a = vec![SpatialVec::default(); n];
        for i in 0..n {
            let link = &mb.links[i];
            let q_offset = link.q_offset;

            let mut a_parent = SpatialVec::new(glam::Vec3::ZERO, -gravity);
            if let Some(p_idx) = link.parent_idx {
                let local_x = link
                    .joint_type
                    .transform(&mb.q[q_offset..q_offset + link.joint_type.dofs()]);
                let x_rel = link.parent_to_joint.combine(&local_x);
                a_parent = transform_motion(a[p_idx], x_rel.rotation, x_rel.position);
            }

            let a_hat = a_parent + c[i];

            let s_vectors = link.joint_type.spatial_subspace();
            if s_vectors.len() == 1 {
                let ddq = d_inv[i] * (force_u[i] - u_vec[i].dot(&a_hat));
                mb.ddq[q_offset] = ddq;
                a[i] = a_hat + s_vectors[0] * ddq;
            } else if s_vectors.is_empty() {
                // Fixed joint: acceleration is just propagated bias
                a[i] = a_hat;
            }
        }
    }
}
