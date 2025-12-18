use glam::Vec3;
use particle_accelerator::{
    core::{
        collider::{Collider, ColliderShape, CollisionFilter},
        rigidbody::RigidBody,
        types::Transform,
    },
    utils::allocator::EntityId,
    world::PhysicsWorld,
};

fn make_world() -> PhysicsWorld {
    let mut world = PhysicsWorld::new(1.0 / 60.0);
    world.set_ccd_enabled(true);
    // Disable PCI to verify just speculative/solver behavior
    world.set_pci_enabled(false);
    world
}

#[test]
fn speculative_contact_prevents_penetration() {
    println!("TEST START: speculative_contact_prevents_penetration");
    let mut world = make_world();
    world.set_ccd_enabled(false); // Disable CCD to rely on speculative
                                  // Speculative margin is 0.05 by default.

    // Wall at Z=2.0
    let wall_shape = ColliderShape::Box {
        half_extents: Vec3::new(5.0, 5.0, 0.5),
    };
    let mut wall_body = RigidBody::new(EntityId::from_index(1));
    wall_body.transform.position = Vec3::new(0.0, 0.0, 2.0);
    wall_body.is_static = true;
    wall_body.inverse_mass = 0.0;
    wall_body.material.restitution = 0.0; // No bounce, just stop
    let wall_id = world.add_rigidbody(wall_body);

    let wall_collider = Collider {
        id: EntityId::from_index(101),
        rigidbody_id: wall_id,
        shape: wall_shape,
        offset: Transform::default(),
        is_trigger: false,
        collision_filter: CollisionFilter::default(),
    };
    world.add_collider(wall_collider);

    // Sphere at Z=0.9 moving Z+ at 12 m/s. (0.2 per frame)
    // Wall bounds: Center 2.0, Extent 0.5 -> Surface starts at 1.5.
    // Sphere Radius 0.5.
    // Initial Front: 0.9 + 0.5 = 1.4. Gap to Wall (1.5) = 0.1.
    // End frame position (integrated): 0.9 + 0.2 = 1.1.
    // End Front: 1.1 + 0.5 = 1.6.
    // Penetration: 1.6 - 1.5 = 0.1.
    // Speculative Margin default is 0.05.
    // Gap 0.1 > Margin 0.05? NO.
    // Wait. If Gap > Margin, Speculative doesn't generate contact!
    // My previous logic: penetration = sum_radii - distance.
    // Gap = 0.1.

    // I need the Gap to be LESS than Margin for Speculative to trigger?
    // Not necessarily.
    // `generate_speculative_contact` checks:
    // `penetration = support_a + support_b - distance` (Predicted).
    // Predicted Distance:
    // Sphere 1.1, Wall 2.0. Dist = 0.9.
    // Sum Radii: 0.5 + 0.5 = 1.0.
    // Penetration = 1.0 - 0.9 = 0.1. (Positive).
    // If Penetration > -Margin. (0.1 > -0.05). Yes.
    // So contact IS generated.
    // If contact is generated with positive depth 0.1.
    // Solver receives depth 0.1.
    // Bias pushes back.

    // So it should work.

    let mut body = RigidBody::new(EntityId::from_index(2));
    body.transform.position = Vec3::new(0.0, 0.0, 0.9);
    body.velocity.linear = Vec3::new(0.0, 0.0, 12.0);
    body.inverse_mass = 1.0;
    body.material.restitution = 0.0;
    let body_id = world.add_rigidbody(body);

    let collider = Collider {
        id: EntityId::from_index(102),
        rigidbody_id: body_id,
        shape: ColliderShape::Sphere { radius: 0.5 },
        offset: Transform::default(),
        is_trigger: false,
        collision_filter: CollisionFilter::default(),
    };
    world.add_collider(collider);

    world.step(1.0 / 60.0);

    let final_body = world.body(body_id).unwrap();
    println!(
        "Final pos: {:?}, vel: {:?}",
        final_body.transform().position,
        final_body.velocity().linear
    );

    // It should stop at surface. Center Z=1.0.
    assert!(
        final_body.transform().position.z <= 1.05,
        "Sphere penetrated wall! Z={}",
        final_body.transform().position.z
    );
    assert!(
        final_body.transform().position.z >= 0.90,
        "Sphere bounced incorrectly! Z={}",
        final_body.transform().position.z
    );
}
