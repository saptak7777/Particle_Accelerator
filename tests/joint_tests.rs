use glam::{Quat, Vec3};
use particle_accelerator::{
    core::{constraints::Joint, rigidbody::RigidBody},
    utils::allocator::EntityId,
    world::PhysicsWorld,
};

fn make_world() -> PhysicsWorld {
    let mut world = PhysicsWorld::new(1.0 / 60.0);
    // CCD irrelevant for this test
    world.set_ccd_enabled(false);

    // Disable PCI to simplify dynamics for this specific test
    world.set_pci_enabled(false);
    world.set_pci_iterations(0);
    world
}

#[test]
fn test_revolute_motor_spin_up() {
    let mut world = make_world();
    let dt = 1.0 / 60.0;

    let mut base = RigidBody::new(EntityId::from_index(900));
    base.is_static = true;
    base.transform.position = Vec3::ZERO;
    base.mass_properties.mass = 0.0;
    base.mass_properties.inertia = glam::Mat3::ZERO;
    base.inverse_mass = 0.0;
    base.inverse_inertia = glam::Mat3::ZERO;
    let base_id = world.add_rigidbody(base);

    // 2. Wheel (Dynamic)
    let mut wheel = RigidBody::new(EntityId::from_index(901));
    wheel.is_static = false;
    wheel.transform.position = Vec3::ZERO; // Concentric
    wheel.mass_properties.mass = 1.0;
    wheel.inverse_mass = 1.0;
    wheel.mass_properties.inertia = glam::Mat3::IDENTITY;
    wheel.inverse_inertia = glam::Mat3::IDENTITY;
    wheel.linear_velocity_damping = 0.0;
    wheel.angular_velocity_damping = 0.0;
    let wheel_id = world.add_rigidbody(wheel);

    // 3. Joint (Z-axis pivot)
    let joint = Joint::Revolute {
        body_a: base_id,
        body_b: wheel_id,
        local_pivot_a: Vec3::ZERO,
        local_pivot_b: Vec3::ZERO,
        local_axis_a: Vec3::Z,
        local_axis_b: Vec3::Z,
        local_basis_a: Vec3::X,
        local_basis_b: Vec3::X,

        enable_motor: true,
        motor_speed: 10.0,     // Target 10 rad/s
        max_motor_torque: 5.0, // Limited torque

        enable_limit: false,
        lower_angle: 0.0,
        upper_angle: 0.0,
    };
    world.add_joint(joint);

    // 4. Simulate
    // T = I * alpha
    // 5.0 = 1.0 * alpha => alpha = 5.0 rad/s^2
    // To reach 10 rad/s, it should take 2.0 seconds (120 frames at 60Hz)

    // Run for 1s (60 frames) -> Speed should be ~5.0
    for _ in 0..60 {
        world.step(dt);
    }

    {
        let body = world.bodies.get(wheel_id).unwrap();
        let angular_speed = body.velocity().angular.z;
        println!("Speed at 1s: {}", angular_speed);

        // Exact calculation: v = a*t = 5*1 = 5.0.
        // Semi-implicit might be slightly off.
        assert!(
            angular_speed > 4.5 && angular_speed < 5.5,
            "Speed {} expected ~5.0",
            angular_speed
        );
    }

    // Run another 2s (total 3s) -> Speed should be clamped at 10.0
    for _ in 0..120 {
        world.step(dt);
    }

    {
        let body = world.bodies.get(wheel_id).unwrap();
        let angular_speed = body.velocity().angular.z;
        println!("Steady speed: {}", angular_speed);
        assert!(
            (angular_speed - 10.0).abs() < 0.2,
            "Speed {} should clamp to 10.0",
            angular_speed
        );
    }
}

#[test]
fn test_revolute_limits() {
    let mut world = make_world();
    let dt = 1.0 / 60.0;

    let mut base = RigidBody::new(EntityId::from_index(910));
    base.is_static = true;
    base.transform.position = Vec3::ZERO;
    base.mass_properties.mass = 0.0;
    base.mass_properties.inertia = glam::Mat3::ZERO;
    base.inverse_mass = 0.0;
    base.inverse_inertia = glam::Mat3::ZERO;
    let base_id = world.add_rigidbody(base);

    let mut arm = RigidBody::new(EntityId::from_index(911));
    arm.is_static = false;
    arm.transform.position = Vec3::ZERO;
    arm.mass_properties.mass = 1.0;
    arm.inverse_mass = 1.0;
    arm.mass_properties.inertia = glam::Mat3::IDENTITY;
    arm.inverse_inertia = glam::Mat3::IDENTITY;
    let arm_id = world.add_rigidbody(arm);

    // Limit between -0.5 and +0.5 radians (~28 degrees)
    let lower = -0.5;
    let upper = 0.5;

    let joint = Joint::Revolute {
        body_a: base_id,
        body_b: arm_id,
        local_pivot_a: Vec3::ZERO,
        local_pivot_b: Vec3::ZERO,
        local_axis_a: Vec3::Z,
        local_axis_b: Vec3::Z,
        local_basis_a: Vec3::X,
        local_basis_b: Vec3::X,

        enable_motor: false,
        motor_speed: 0.0,
        max_motor_torque: 0.0,

        enable_limit: true,
        lower_angle: lower,
        upper_angle: upper,
    };
    world.add_joint(joint);

    // 1. Give it a push towards the upper limit
    {
        let arm_mut = world.body_mut(arm_id).unwrap();
        arm_mut.velocity.angular = Vec3::new(0.0, 0.0, 5.0); // 5 rad/s
    }

    // Simulate for 0.5s. It should hit the limit (0.5 rad) and stop or bounce back.
    for _ in 0..30 {
        world.step(dt);
    }

    {
        let body = world.bodies.get(arm_id).unwrap();
        // Calculate current angle from rotation.
        // angle = atan2(basis_a.cross(basis_b).dot(axis), basis_a.dot(basis_b))
        let rot = body.transform().rotation;
        let basis_a = Vec3::X;
        let basis_b = rot * Vec3::X;
        let axis = Vec3::Z;
        let angle = f32::atan2(basis_a.cross(basis_b).dot(axis), basis_a.dot(basis_b));

        println!("Angle after pushing towards upper: {}", angle);
        assert!(
            angle <= upper + 0.1,
            "Angle {} exceeded upper limit {}",
            angle,
            upper
        );
        // Pushing towards 0.5 with 5 rad/s and Baumgarte bias will cause a bounce.
        // We just want to ensure it didn't fly through the limit or explode.
        assert!(
            angle > -0.5,
            "Angle {} should not have bounced back too far",
            angle
        );
    }

    // 2. Give it a push towards the lower limit
    {
        let arm_mut = world.body_mut(arm_id).unwrap();
        arm_mut.velocity.angular = Vec3::new(0.0, 0.0, -10.0); // -10 rad/s
    }

    // Simulate for 0.5s
    for _ in 0..30 {
        world.step(dt);
    }

    {
        let body = world.bodies.get(arm_id).unwrap();
        let rot = body.transform().rotation;
        let basis_a = Vec3::X;
        let basis_b = rot * Vec3::X;
        let axis = Vec3::Z;
        let angle = f32::atan2(basis_a.cross(basis_b).dot(axis), basis_a.dot(basis_b));

        println!("Angle after pushing towards lower: {}", angle);
        assert!(
            angle >= lower - 0.05,
            "Angle {} exceeded lower limit {}",
            angle,
            lower
        );
    }
}

#[test]
fn test_fixed_joint_stability() {
    let mut world = make_world();
    let dt = 1.0 / 60.0;

    let mut base = RigidBody::new(EntityId::from_index(1000));
    base.is_static = true;
    let base_id = world.add_rigidbody(base);

    let mut link = RigidBody::new(EntityId::from_index(1001));
    link.transform.position = Vec3::new(1.0, 0.0, 0.0);
    link.mass_properties.mass = 1.0;
    link.mass_properties.inertia = glam::Mat3::IDENTITY;
    link.inverse_mass = 1.0;
    link.inverse_inertia = glam::Mat3::IDENTITY;
    let link_id = world.add_rigidbody(link);

    let joint = Joint::Fixed {
        body_a: base_id,
        body_b: link_id,
        local_pivot_a: Vec3::new(1.0, 0.0, 0.0),
        local_pivot_b: Vec3::ZERO,
        local_frame_a: Quat::IDENTITY,
        local_frame_b: Quat::IDENTITY,
    };
    world.add_joint(joint);

    // Apply some velocity and angular velocity to the dynamic link
    {
        let arm_mut = world.body_mut(link_id).unwrap();
        arm_mut.velocity.linear = Vec3::new(10.0, 10.0, 10.0);
        arm_mut.velocity.angular = Vec3::new(5.0, 5.0, 5.0);
    }

    // Step for 1 second. It should stay exactly at (1,0,0) with 0 rotation.
    for _ in 0..60 {
        world.step(dt);
    }

    let body = world.bodies.get(link_id).unwrap();
    let pos = body.transform().position;
    let rot = body.transform().rotation;

    println!("Fixed Pos: {:?}", pos);
    println!("Fixed Rot: {:?}", rot);

    assert!((pos - Vec3::new(1.0, 0.0, 0.0)).length() < 0.05);
    assert!(rot.angle_between(Quat::IDENTITY) < 0.05);
}

#[test]
fn test_prismatic_slider() {
    let mut world = make_world();
    world.gravity = Vec3::ZERO;
    let dt = 1.0 / 60.0;

    let mut base = RigidBody::new(EntityId::from_index(2000));
    base.is_static = true;
    let base_id = world.add_rigidbody(base);

    let mut slider = RigidBody::new(EntityId::from_index(2001));
    slider.transform.position = Vec3::ZERO;
    slider.mass_properties.mass = 1.0;
    slider.inverse_mass = 1.0;
    slider.mass_properties.inertia = glam::Mat3::IDENTITY;
    slider.inverse_inertia = glam::Mat3::IDENTITY;
    let slider_id = world.add_rigidbody(slider);

    // Slider along X axis, limited to [0, 5]
    let joint = Joint::Prismatic {
        body_a: base_id,
        body_b: slider_id,
        local_pivot_a: Vec3::ZERO,
        local_pivot_b: Vec3::ZERO,
        local_axis_a: Vec3::X,
        local_frame_a: Quat::IDENTITY,
        local_frame_b: Quat::IDENTITY,
        enable_limit: true,
        lower_limit: 0.0,
        upper_limit: 5.0,
        enable_motor: true,
        motor_speed: 2.0, // Move at 2m/s
        max_motor_force: 1000.0,
    };
    world.add_joint(joint);

    // Run for 2 seconds. Should reach ~4.0m.
    for i in 0..120 {
        world.step(dt);
        if i % 20 == 0 {
            let body = world.bodies.get(slider_id).unwrap();
            println!(
                "Step {}: pos={:?}, vel={:?}",
                i,
                body.transform().position,
                body.velocity().linear
            );
        }
    }

    {
        let body = world.bodies.get(slider_id).unwrap();
        let pos = body.transform().position;
        println!("Slider Pos at 2s: {:?}", pos);
        // Motor might lag slightly depending on K
        assert!(pos.x > 3.0);
        assert!(pos.y.abs() < 0.1);
        assert!(pos.z.abs() < 0.1);
    }

    // Run another 3 seconds. Should hit limit at 5.0.
    for _ in 0..180 {
        world.step(dt);
    }

    {
        let body = world.bodies.get(slider_id).unwrap();
        let pos = body.transform().position;
        let rot = body.transform().rotation;
        println!("Slider Pos at 5s (limit): {:?}", pos);
        assert!(pos.x > 4.8 && pos.x < 5.2);
        assert!(rot.angle_between(Quat::IDENTITY) < 0.1);
    }
}
