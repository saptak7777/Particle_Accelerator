use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use particle_accelerator::{core::mesh::TriangleMesh, *};
use std::hint::black_box;

const DT: f32 = 1.0 / 60.0;

fn prepare_world(body_count: usize) -> PhysicsEngine {
    let mut engine = PhysicsEngine::new(DT);
    for i in 0..body_count {
        let mut body = RigidBody::new(EntityId::from_index(i as u32));
        body.transform.position = Vec3::new(i as f32 * 0.1, 0.0, 0.0);
        engine.add_body(body);
    }
    engine
}

fn bench_world_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("world_step");
    for &count in &[128usize, 512, 2048] {
        group.bench_with_input(
            BenchmarkId::new("sequential", count),
            &count,
            |b, &count| {
                b.iter(|| {
                    let mut engine = prepare_world(count);
                    engine.set_parallel_enabled(false);
                    engine.step(black_box(DT));
                })
            },
        );
        group.bench_with_input(BenchmarkId::new("parallel", count), &count, |b, &count| {
            b.iter(|| {
                let mut engine = prepare_world(count);
                engine.set_parallel_enabled(true);
                engine.step(black_box(DT));
            })
        });
    }
    group.finish();
}

fn generate_grid_mesh(resolution: usize) -> (Vec<Vec3>, Vec<[u32; 3]>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    for y in 0..=resolution {
        for x in 0..=resolution {
            vertices.push(Vec3::new(x as f32, 0.0, y as f32));
        }
    }
    let width = resolution + 1;
    for y in 0..resolution {
        for x in 0..resolution {
            let i = y * width + x;
            let a = i as u32;
            let b = (i + 1) as u32;
            let c = (i + width) as u32;
            let d = (i + width + 1) as u32;
            indices.push([a, b, c]);
            indices.push([b, d, c]);
        }
    }
    (vertices, indices)
}

fn bench_mesh_builder(c: &mut Criterion) {
    let mut group = c.benchmark_group("mesh_builder");
    for &res in &[16usize, 32, 64] {
        group.bench_with_input(BenchmarkId::new("build", res), &res, |b, &res| {
            let (vertices, indices) = generate_grid_mesh(res);
            b.iter(|| {
                let mesh = TriangleMesh::builder(vertices.clone(), indices.clone())
                    .weld_vertices(0.001)
                    .recenter()
                    .build();
                black_box(mesh)
            })
        });
    }
    group.finish();
}

fn bench_gjk(c: &mut Criterion) {
    let mut group = c.benchmark_group("gjk_narrowphase");
    let count = 1000;

    // Prepare data
    use particle_accelerator::collision::narrowphase::GJKAlgorithm;
    use particle_accelerator::core::{collider::ColliderShape, types::Transform};
    use particle_accelerator::utils::allocator::EntityId;
    use particle_accelerator::utils::simd::gjk_batch;

    let shape = ColliderShape::Box {
        half_extents: Vec3::splat(0.5),
    };
    let shapes = vec![&shape; count];
    let mut transforms_a = Vec::with_capacity(count);
    let mut transforms_b = Vec::with_capacity(count);

    for i in 0..count {
        transforms_a.push(Transform {
            position: Vec3::new(i as f32 * 2.0, 0.0, 0.0),
            rotation: glam::Quat::IDENTITY,
            scale: Vec3::ONE,
        });
        transforms_b.push(Transform {
            position: Vec3::new(i as f32 * 2.0 + 0.8, 0.0, 0.0), // Intersecting
            rotation: glam::Quat::IDENTITY,
            scale: Vec3::ONE,
        });
    }

    group.bench_function("scalar_loop", |b| {
        b.iter(|| {
            for i in 0..count {
                let _ = black_box(GJKAlgorithm::intersect(
                    shapes[i],
                    &transforms_a[i],
                    shapes[i],
                    &transforms_b[i],
                    EntityId::default(),
                    EntityId::default(),
                    None,
                ));
            }
        })
    });

    group.bench_function("batch_simd", |b| {
        b.iter(|| {
            let _ = black_box(gjk_batch(&shapes, &transforms_a, &shapes, &transforms_b));
        })
    });

    group.finish();
}

criterion_group!(benches, bench_world_step, bench_mesh_builder, bench_gjk);
criterion_main!(benches);
