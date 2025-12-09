use particle_accelerator::*;

fn main() {
    let mut world = PhysicsWorld::new(1.0 / 60.0);

    let mut ground = RigidBody::new(EntityId::from_index(0));
    ground.is_static = true;
    world.add_rigidbody(ground);

    for i in 0..5 {
        let mut body = RigidBody::new(EntityId::from_index((i + 1) as u32));
        body.transform.position = Vec3::new(0.0, i as f32 + 0.5, 0.0);
        world.add_rigidbody(body);
    }

    for _ in 0..120 {
        world.step(1.0 / 60.0);
    }

    println!("Simulated stack of boxes for 2 seconds");
}
