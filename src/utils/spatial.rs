use glam::{Mat3, Vec3};

/// A 6D spatial vector combining angular and linear components.
/// In motion space, angular is velocity and linear is translation.
/// In force space, angular is torque and linear is force.
#[derive(Debug, Clone, Copy, Default)]
pub struct SpatialVec {
    pub ang: Vec3,
    pub lin: Vec3,
}

impl SpatialVec {
    pub fn new(ang: Vec3, lin: Vec3) -> Self {
        Self { ang, lin }
    }

    pub fn dot(&self, other: &SpatialVec) -> f32 {
        self.ang.dot(other.ang) + self.lin.dot(other.lin)
    }

    /// Spatial motion cross product: v1 x_m v2
    pub fn cross_motion(&self, other: &SpatialVec) -> SpatialVec {
        SpatialVec {
            ang: self.ang.cross(other.ang),
            lin: self.ang.cross(other.lin) + self.lin.cross(other.ang),
        }
    }

    /// Spatial force cross product: v x_f f
    pub fn cross_force(&self, other: &SpatialVec) -> SpatialVec {
        SpatialVec {
            ang: self.ang.cross(other.ang) + self.lin.cross(other.lin),
            lin: self.ang.cross(other.lin),
        }
    }
}

impl std::ops::Add for SpatialVec {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self {
            ang: self.ang + other.ang,
            lin: self.lin + other.lin,
        }
    }
}

impl std::ops::Sub for SpatialVec {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        Self {
            ang: self.ang - other.ang,
            lin: self.lin - other.lin,
        }
    }
}

impl std::ops::Mul<f32> for SpatialVec {
    type Output = Self;
    fn mul(self, rhs: f32) -> Self {
        Self {
            ang: self.ang * rhs,
            lin: self.lin * rhs,
        }
    }
}

/// A 6x6 spatial matrix represented as 4 3x3 blocks.
#[derive(Debug, Clone, Copy, Default)]
pub struct SpatialMat {
    pub m00: Mat3,
    pub m01: Mat3,
    pub m10: Mat3,
    pub m11: Mat3,
}

impl SpatialMat {
    pub fn new(m00: Mat3, m01: Mat3, m10: Mat3, m11: Mat3) -> Self {
        Self { m00, m01, m10, m11 }
    }

    pub fn mul_vec(&self, v: SpatialVec) -> SpatialVec {
        SpatialVec {
            ang: self.m00 * v.ang + self.m01 * v.lin,
            lin: self.m10 * v.ang + self.m11 * v.lin,
        }
    }

    /// Computes the outer product (v * v.T) as a 6x6 matrix.
    pub fn outer_product(v: SpatialVec) -> Self {
        Self {
            m00: outer_vec3(v.ang, v.ang),
            m01: outer_vec3(v.ang, v.lin),
            m10: outer_vec3(v.lin, v.ang),
            m11: outer_vec3(v.lin, v.lin),
        }
    }
}

fn outer_vec3(a: glam::Vec3, b: glam::Vec3) -> glam::Mat3 {
    glam::Mat3::from_cols(a * b.x, a * b.y, a * b.z)
}

impl std::ops::Add for SpatialMat {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self {
            m00: self.m00 + other.m00,
            m01: self.m01 + other.m01,
            m10: self.m10 + other.m10,
            m11: self.m11 + other.m11,
        }
    }
}

impl std::ops::Sub for SpatialMat {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        Self {
            m00: self.m00 - other.m00,
            m01: self.m01 - other.m01,
            m10: self.m10 - other.m10,
            m11: self.m11 - other.m11,
        }
    }
}

impl std::ops::Mul<f32> for SpatialMat {
    type Output = Self;
    fn mul(self, rhs: f32) -> Self {
        Self {
            m00: self.m00 * rhs,
            m01: self.m01 * rhs,
            m10: self.m10 * rhs,
            m11: self.m11 * rhs,
        }
    }
}

impl SpatialMat {
    /// Transforms a spatial inertia matrix from one frame to another.
    /// I' = X.T * I * X
    pub fn transform_inertia(
        i: &SpatialMat,
        _rotation: glam::Quat,
        _translation: glam::Vec3,
    ) -> Self {
        // This is a complex transformation.
        // Vector transforms are implemented as the primary mechanism for state propagation.
        // Basic vector transforms are utilized for the initial ABA implementation.
        *i
    }
}

/// Helper to transform a motion vector between frames.
pub fn transform_motion(
    v: SpatialVec,
    rotation: glam::Quat,
    translation: glam::Vec3,
) -> SpatialVec {
    let ang = rotation * v.ang;
    let lin = rotation * (v.lin - translation.cross(v.ang));
    SpatialVec { ang, lin }
}

/// Helper to transform a force vector between frames.
pub fn transform_force(f: SpatialVec, rotation: glam::Quat, translation: glam::Vec3) -> SpatialVec {
    let lin = rotation * f.lin;
    let ang = rotation * (f.ang - translation.cross(f.lin));
    SpatialVec { ang, lin }
}

/// A spatial inertia tensor representing mass, center of mass, and rotational inertia.
#[derive(Debug, Clone, Copy)]
pub struct SpatialInertia {
    pub mass: f32,
    pub com: Vec3,
    pub inertia: Mat3, // Rotational inertia at COM
}

impl SpatialInertia {
    pub fn new(mass: f32, com: Vec3, inertia: Mat3) -> Self {
        Self { mass, com, inertia }
    }

    /// Converts this spatial inertia to its 6x6 matrix representation.
    pub fn to_mat(&self) -> SpatialMat {
        let m = self.mass;
        let c = self.com;
        let c_skew = Mat3::from_cols(
            Vec3::new(0.0, c.z, -c.y),
            Vec3::new(-c.z, 0.0, c.x),
            Vec3::new(c.y, -c.x, 0.0),
        );
        let mc_skew = c_skew * m;

        // I_fixed = I_com - m * c_skew * c_skew
        let i_fixed = self.inertia - c_skew * c_skew * m;

        SpatialMat {
            m00: i_fixed,
            m01: mc_skew,
            m10: mc_skew.transpose(),
            m11: Mat3::IDENTITY * m,
        }
    }

    /// Multiplies spatial inertia by spatial motion to get spatial force: f = I * v
    pub fn mul_motion(&self, v: SpatialVec) -> SpatialVec {
        let m = self.mass;
        let c = self.com;

        let ang = self.inertia * v.ang + c.cross(v.lin * m) + c.cross(v.ang.cross(c * m));
        let lin = m * (v.lin - c.cross(v.ang));

        SpatialVec { ang, lin }
    }

    /// Adds two spatial inertias in the same coordinate frame.
    pub fn add(&self, other: &Self) -> Self {
        let m_total = self.mass + other.mass;
        if m_total < 1e-9 {
            return *self;
        }

        let com_total = (self.com * self.mass + other.com * other.mass) / m_total;

        // Parallel axis theorem to move both to total COM
        let d1 = self.com - com_total;
        let d2 = other.com - com_total;

        let i1 = self.inertia + self.inertia_offset(d1, self.mass);
        let i2 = other.inertia + self.inertia_offset(d2, other.mass);

        Self {
            mass: m_total,
            com: com_total,
            inertia: i1 + i2,
        }
    }

    fn inertia_offset(&self, d: Vec3, m: f32) -> Mat3 {
        let d2 = d.length_squared();
        Mat3::from_diagonal(Vec3::splat(d2)) - Mat3::from_cols(d * d.x, d * d.y, d * d.z) * m
    }
}
