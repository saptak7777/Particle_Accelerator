use particle_accelerator::*;

fn main() {
    let mut world = PhysicsWorld::new(1.0 / 60.0);
    let body_id = world.add_rigidbody(RigidBody::new(EntityId::from_index(0)));
    let collider = Collider {
        id: EntityId::from_index(1),
        rigidbody_id: body_id,
        shape: ColliderShape::Sphere { radius: 1.0 },
        offset: Transform::default(),
        is_trigger: false,
        collision_filter: CollisionFilter::default(),
    };
    world.add_collider(collider);

    let query = RaycastQuery {
        origin: Vec3::new(0.0, 0.0, -5.0),
        direction: Vec3::Z,
        max_distance: 10.0,
    };

    let hits = Raycast::cast(&query, &world.colliders, &world.bodies);
    println!("Ray hits: {}", hits.len());
}
