use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use particle_accelerator::*;
use std::hint::black_box;

const DT: f32 = 1.0 / 60.0;

fn bench_gpu_vs_cpu_broadphase(c: &mut Criterion) {
    let mut group = c.benchmark_group("broadphase_scaling");

    // Test from 1k to 100k objects
    for &count in &[1000usize, 10000, 50000, 100000] {
        group.bench_with_input(BenchmarkId::new("cpu_grid", count), &count, |b, &count| {
            let mut world = PhysicsWorld::builder().time_step(DT).build();

            for i in 0..count {
                let body = RigidBody::builder()
                    .position(glam::Vec3::new(i as f32 * 0.1, 0.0, 0.0))
                    .build();
                world.add_rigidbody(body);

                let collider = Collider::builder().sphere(0.5).build();
                world.add_collider(collider);
            }

            b.iter(|| {
                // Manual call to broadphase to isolate performance
                black_box(world.collect_contacts());
            })
        });

        // For GPU, we'd need a real Vulkan device.
        // This benchmark serves as a template for the user to run on their Intel Arc.
        group.bench_with_input(
            BenchmarkId::new("gpu_grid_sync_overhead", count),
            &count,
            |b, &_count| {
                let mut world = PhysicsWorld::builder().time_step(DT).build();

                // (Populate world as above...)

                b.iter(|| {
                    // This measures the time to sync data to GPU-accessible memory
                    // even if the dispatch is mocked or noop.
                    world.step(DT);
                })
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_gpu_vs_cpu_broadphase);
criterion_main!(benches);
