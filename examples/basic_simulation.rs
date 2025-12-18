use particle_accelerator::*;

fn main() {
    let mut engine = PhysicsEngine::new(1.0 / 60.0);
    engine.set_parallel_enabled(true);

    let mut ground = RigidBody::new(EntityId::from_index(0));
    ground.is_static = true;
    engine.add_body(ground);

    let mut body = RigidBody::new(EntityId::from_index(1));
    body.transform.position = Vec3::new(0.0, 1.0, 0.0);
    body.mass_properties.mass = 1.0;
    let body_id = engine.add_body(body);

    let collider = Collider {
        id: EntityId::from_index(2),
        rigidbody_id: body_id,
        shape: ColliderShape::Sphere { radius: 0.5 },
        offset: Transform::default(),
        is_trigger: false,
        collision_filter: CollisionFilter::default(),
    };
    engine.add_collider(collider);

    engine.step(1.0 / 60.0);
    if let Some(body) = engine.get_body(body_id) {
        println!(
            "Body position after one step: {:?}",
            body.transform().position
        );
    }
}
