use particle_accelerator::*;

#[test]
fn bodies_fall_under_gravity() {
    let mut world = PhysicsWorld::new(1.0 / 60.0);
    let mut body = RigidBody::new(EntityId::from_index(0));
    body.transform.position = Vec3::new(0.0, 10.0, 0.0);
    body.mass_properties.mass = 1.0;
    let body_id = world.add_rigidbody(body);

    world.step(1.0 / 60.0);

    let position_y = world
        .body(body_id)
        .expect("body should exist")
        .transform
        .position
        .y;
    assert!(position_y < 10.0, "body should start falling, y = {}", position_y);
}
