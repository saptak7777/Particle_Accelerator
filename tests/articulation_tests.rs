use glam::{Mat3, Vec3};
use particle_accelerator::core::articulations::{
    JointType as ArticulatedJointType, Link, Multibody,
};
use particle_accelerator::core::types::Transform;
use particle_accelerator::utils::allocator::EntityId;
use particle_accelerator::world::PhysicsWorld;

#[test]
fn test_articulation_chain_stability() {
    let mut world = PhysicsWorld::new(1.0 / 60.0);
    let mut mb = Multibody::new(EntityId::default());

    // Link 1 (Fixed to root)
    let link1 = Link::new("root", None, ArticulatedJointType::Fixed);
    mb.add_link(link1);

    // Link 2 (Revolute Z)
    let mut link2 = Link::new(
        "link2",
        Some(0),
        ArticulatedJointType::Revolute { axis: Vec3::Z },
    );
    link2.parent_to_joint = Transform {
        position: Vec3::X,
        ..Transform::default()
    };
    link2.mass = 1.0;
    link2.com_offset = Vec3::X * 0.5;
    link2.inertia = Mat3::IDENTITY;
    mb.add_link(link2);

    // Apply an initial velocity
    mb.dq[0] = 1.0; // 1 rad/s

    world.add_multibody(mb);

    // Step and check stability
    for _ in 0..60 {
        world.step(1.0 / 60.0);
    }

    let mb_ref = world.articulated_bodies.iter().next().unwrap();
    println!("Final q[0]: {}", mb_ref.q[0]);
    // It should have moved significantly from 0
    assert!(mb_ref.q[0].abs() > 0.01);
}

#[test]
fn test_double_pendulum_under_gravity() {
    let mut world = PhysicsWorld::new(1.0 / 60.0);
    let mut mb = Multibody::new(EntityId::default());

    // Link 1: Fixed to world at origin
    let link1 = Link::new("root", None, ArticulatedJointType::Fixed);
    mb.add_link(link1);

    // Link 2: Revolute Z at (1,0,0) rel to parent
    let mut link2 = Link::new(
        "link2",
        Some(0),
        ArticulatedJointType::Revolute { axis: Vec3::Z },
    );
    link2.parent_to_joint = Transform {
        position: Vec3::X,
        ..Transform::default()
    };
    link2.mass = 1.0;
    link2.com_offset = Vec3::X * 0.5;
    mb.add_link(link2);

    // Link 3: Revolute Z at (1,0,0) rel to link 2
    let mut link3 = Link::new(
        "link3",
        Some(1),
        ArticulatedJointType::Revolute { axis: Vec3::Z },
    );
    link3.parent_to_joint = Transform {
        position: Vec3::X,
        ..Transform::default()
    };
    link3.mass = 1.0;
    link3.com_offset = Vec3::X * 0.5;
    mb.add_link(link3);

    world.add_multibody(mb);

    // Step for 1 second
    for _ in 0..60 {
        world.step(1.0 / 60.0);
    }

    let mb_ref = world.articulated_bodies.iter().next().unwrap();
    println!("Double Pendulum q: {:?}", mb_ref.q);
    // Final check: should have moved due to gravity
    assert!(mb_ref.q[0].abs() > 0.01);
    assert!(mb_ref.q[1].abs() > 0.01);
}
