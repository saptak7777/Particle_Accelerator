use criterion::{black_box, criterion_group, criterion_main, Criterion};
use particle_accelerator::*;

fn bench_broad_phase(c: &mut Criterion) {
    c.bench_function("broad_phase_100_bodies", |b| {
        b.iter(|| {
            let mut world = PhysicsWorld::new(1.0 / 60.0);
            for i in 0..100 {
                let mut body = RigidBody::new(EntityId::from_index(i));
                body.transform.position = Vec3::new(i as f32 * 0.1, 0.0, 0.0);
                world.add_rigidbody(body);
            }
            world.step(black_box(1.0 / 60.0));
        })
    });
}

criterion_group!(benches, bench_broad_phase);
criterion_main!(benches);
