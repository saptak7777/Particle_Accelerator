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
    // Disable PCI to verify restitution bounce without interference
    world.set_pci_enabled(false);
    world.set_pci_iterations(0);
    world
}

#[test]
fn fast_sphere_hits_thin_wall() {
    println!("TEST START: fast_sphere_hits_thin_wall");
    let mut world = make_world();

    // Wall at Z=5, thickness 0.2
    let wall_shape = ColliderShape::Box {
        half_extents: Vec3::new(10.0, 10.0, 0.1),
    };
    let mut wall_body = RigidBody::new(EntityId::from_index(1));
    wall_body.transform.position = Vec3::new(0.0, 0.0, 5.0);
    wall_body.is_static = true;
    wall_body.inverse_mass = 0.0; // Infinite mass for perfect bounce calculation
    wall_body.material.restitution = 1.0; // Perfect bounce
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

    // Bullet at Z=0, moving Z+ at 600 m/s (10m per frame!)
    // Wall is at 5.0. Frame 1 end: 10.0.
    // It should pass through completely without CCD.
    let mut bullet_body = RigidBody::new(EntityId::from_index(2));
    bullet_body.transform.position = Vec3::new(0.0, 0.0, 0.0);
    bullet_body.velocity.linear = Vec3::new(0.0, 0.0, 600.0);
    bullet_body.inverse_mass = 1.0;
    bullet_body.material.restitution = 1.0; // Perfect bounce
    let bullet_id = world.add_rigidbody(bullet_body);

    let bullet_collider = Collider {
        id: EntityId::from_index(102),
        rigidbody_id: bullet_id,
        shape: ColliderShape::Sphere { radius: 0.2 },
        offset: Transform::default(),
        is_trigger: false,
        collision_filter: CollisionFilter::default(),
    };
    world.add_collider(bullet_collider);

    // Step
    println!("Step start");
    world.step(1.0 / 60.0);
    println!("Step end");

    let final_body = world.body(bullet_id).unwrap();

    println!(
        "Final pos: {:?}, vel: {:?}",
        final_body.transform().position,
        final_body.velocity().linear
    );

    assert!(
        final_body.transform().position.z < 6.0,
        "Bullet tunneled through wall!"
    );
    // With 1.0 restitution, it should bounce back with negative velocity
    assert!(
        final_body.velocity().linear.z < 0.0,
        "Bullet did not bounce!"
    );
}

#[test]
fn fast_box_hits_wall_ccd() {
    println!("TEST START: fast_box_hits_wall_ccd");
    let mut world = make_world();

    // Wall at Z=10
    let wall_shape = ColliderShape::Box {
        half_extents: Vec3::new(5.0, 5.0, 0.5),
    };
    let mut wall_body = RigidBody::new(EntityId::from_index(1));
    wall_body.transform.position = Vec3::new(0.0, 0.0, 10.0);
    wall_body.is_static = true;
    wall_body.inverse_mass = 0.0;
    wall_body.material.restitution = 1.0;
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

    // Box projectile
    let mut box_body = RigidBody::new(EntityId::from_index(2));
    box_body.transform.position = Vec3::new(0.0, 0.0, 0.0);
    box_body.velocity.linear = Vec3::new(0.0, 0.0, 600.0); // 10m/frame
    box_body.inverse_mass = 1.0;
    box_body.material.restitution = 1.0;
    let box_id = world.add_rigidbody(box_body);

    let box_collider = Collider {
        id: EntityId::from_index(102),
        rigidbody_id: box_id,
        shape: ColliderShape::Box {
            half_extents: Vec3::splat(0.2),
        },
        offset: Transform::default(),
        is_trigger: false,
        collision_filter: CollisionFilter::default(),
    };
    world.add_collider(box_collider);

    world.step(1.0 / 60.0);

    let final_body = world.body(box_id).unwrap();
    println!(
        "Final box pos: {:?}, vel: {:?}",
        final_body.transform().position,
        final_body.velocity().linear
    );

    assert!(final_body.transform().position.z < 11.0, "Box tunneled!");
    assert!(final_body.velocity().linear.z < 0.0, "Box did not bounce!");
}
