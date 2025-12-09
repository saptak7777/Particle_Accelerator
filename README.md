# Particle Accelerator

A modular, ECS-ready physics engine prototype written in Rust. Particle Accelerator targets learning and experimentation: it focuses on readable architecture, explicit subsystems, and a clear roadmap for future work.

## Features

- **Generational arena storage** for rigid bodies and colliders, enabling stable `EntityId`s and safe references.
- **Core dynamics pipeline** with force registry, semi-implicit Euler integrator, and contact solver placeholder.
- **Collision detection** including spatial-grid broad phase, GJK/SAT-based narrow phase, and continuous collision detection sweeps feeding time-of-impact contacts into the solver.
- **Query utilities** with ray casting across multiple shapes (including mesh triangles), layer/mask filtering, trigger toggles, and user-defined filters exposed through `PhysicsWorld::raycast_with_filter`.
- **Extensible module layout** (`core`, `dynamics`, `collision`, `utils`, `world`, `gpu`) aligned with ECS-inspired architectures.
- **SIMD + parallel performance path** via `utils::simd` helpers, Rayon-backed integrator & solver toggles, and ScopedTimer instrumentation.
- **Mesh collider pipeline** with welding/recentering builders, CCD/broad-phase/raycast support, and mass/inertia approximation helpers.
- **GPU-ready scaffolding** exposing a `ComputeBackend` trait and `GpuWorldState` snapshot for future wgpu/cuda accelerators (ships with a CPU `NoopBackend`).
- **Examples and benchmarks** demonstrating basic simulation, stacking, ray casting, mesh authoring, and Criterion world-step / mesh-builder microbenches.

## Getting Started

### Prerequisites
- Rust 1.75+ (2021 edition)
- `cargo` for building, testing, and running examples

### Build & Test
```bash
cargo build
cargo test
cargo clippy
```

### Run Examples
```bash
cargo run --example basic_simulation
cargo run --example stacking
cargo run --example ray_casting
```
Try enabling parallel mode or mesh colliders inside the examples to see the new systems in action.

### Run Mesh Example (snippet)
```rust
use particle_accelerator::*;
use glam::Vec3;

fn main() {
    let mut engine = PhysicsEngine::new(1.0 / 60.0);
    engine.set_parallel_enabled(true);
    // Optional: swap to a custom GPU backend once implemented
    // engine.set_gpu_backend(MyWgpuBackend::new()?);

    let mesh = TriangleMesh::builder(
        vec![
            Vec3::new(-1.0, 0.0, -1.0),
            Vec3::new(1.0, 0.0, -1.0),
            Vec3::new(1.0, 0.0, 1.0),
            Vec3::new(-1.0, 0.0, 1.0),
        ],
        vec![[0, 1, 2], [0, 2, 3]],
    )
    .weld_vertices(0.001)
    .recenter()
    .build();

    let mut body = RigidBody::new(EntityId::from_index(0));
    body.transform.position = Vec3::new(0.0, 2.0, 0.0);
    let body_id = engine.add_body(body);

    let collider = Collider {
        id: EntityId::from_index(1),
        rigidbody_id: body_id,
        shape: ColliderShape::Mesh { mesh },
        offset: Transform::default(),
        is_trigger: false,
        collision_filter: CollisionFilter::default(),
    };
    engine.add_collider(collider);

    engine.step(1.0 / 60.0);
}
```

### Benchmarks
```bash
cargo bench
```
The Criterion suite reports sequential vs. parallel world-step timings at multiple body counts so you can quantify the impact of `PhysicsEngine::set_parallel_enabled(true)`.

## Architecture Highlights
- `PhysicsWorld` orchestrates arenas, forces, integration, broad phase, narrow phase, and constraint solving.
- `utils::allocator::Arena` provides generational IDs and ergonomic iteration helpers.
- `collision` module now wires broad-phase pairs through narrow-phase manifolds and CCD sweeps, with ray queries offering filtering callbacks.
- `dynamics` module defines the force registry, integrator, solver, and island manager for sleeping logic.

For deeper explanations, see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Roadmap Snapshot
1. ✅ **Phase 4 – Constraint & Island Stability**: PGSSolver velocity/position iterations, joint scaffolding, island-based sleeping now integrated in the world step.
2. ✅ **Phase 5 – Advanced Collision Features**  
   - CCD sweeps emit TOI-aware contacts with shape-aware radii and angular padding.  
   - Ray casting exposes layer masks, trigger filtering, and API-level callbacks backed by examples/tests.  
   - Regression tests cover filtered raycasts and high-speed tunneling scenarios.  
3. ✅ **Phase 6 – Optimization & Polish**  
   - SIMD helper suite (`utils::simd`) powers convex support, broad-phase radii, and CCD math.  
   - Optional Rayon-backed integrator & solver paths with ScopedTimer profiling and Criterion coverage.  
   - Documentation/examples refreshed to highlight perf tuning knobs.  
4. ✅ **Phase 7 – Advanced Shapes & GPU Scaffolding**  
   - Mesh colliders with weld/recenter builders, mass/inertia approximations, ray/CCD/broad-phase support.  
   - GPU-ready SoA snapshot (`GpuWorldState`) plus pluggable `ComputeBackend` hooks and a default CPU backend.  
   - New mesh tests/benches validating the authoring flow.  
5. 📌 Future: Higher-level ECS bindings, rendering hooks, soft bodies, and full GPU acceleration.

## Phase 6 Highlights & What's Next
1. **SIMD Math Helpers** – `batch_transform_points`, `max_dot(_point)`, and `max_length` unlock shared hot-path acceleration (broad-phase, CCD, convex support).
2. **Parallel Integrator & Solver Hooks** – `PhysicsEngine::set_parallel_enabled(true)` flips both integrator and solver to Rayon-driven slices with per-island job batching.
3. **Benchmarks & Instrumentation** – Criterion’s `world_step` group compares sequential vs. parallel execution (128–2048 bodies) while `ScopedTimer` labels surface perf traces during normal runs.
4. **Documentation & Examples** – examples now show how to toggle parallel mode, and roadmap notes include prep for Phase 7 (GPU + advanced shapes).

**Next up:** finalize GPU/advanced-shape planning, add a dedicated CCD + raycast perf demo, and explore exposing SIMD job modes to users that want explicit control over Rayon in helper routines.

## Phase 7 Highlights
1. **Mesh Collider Pipeline** – `TriangleMesh::builder` now welds/recenters geometry, feeds CCD/broad-phase/raycast, and approximates mass/inertia so complex shapes are drop-in ready.
2. **GPU-Ready Architecture** – `GpuWorldState` snapshots bodies/colliders into SoA buffers, while the `ComputeBackend` trait (defaulting to `NoopBackend`) lets future wgpu/cuda backends plug into the step loop.
3. **Expanded Tooling** – new `mesh_tests` ensure authoring correctness; Criterion gained `mesh_builder` benchmarks for cook times alongside the existing world-step suite.

## Benchmark Snapshot
_Machine: Windows, release build, `cargo bench`_

### World Step (lower is faster)

| Bodies | Sequential | Parallel |
|--------|-----------:|---------:|
| 128    | 61 µs      | 388 µs   |
| 512    | 227 µs     | 790 µs   |
| 2048   | 877 µs     | 2.21 ms  |

### Mesh Builder (grid resolution)

| Resolution | Build Time |
|------------|-----------:|
| 16         | 25.8 µs    |
| 32         | 94.4 µs    |
| 64         | 365 µs     |

Use `cargo bench -- world_step` (or `cargo bench -- mesh_builder`) to reproduce these numbers and compare future optimizations.

## Performance Tuning Cheat Sheet
- **Enable parallel execution:** call `engine.set_parallel_enabled(true)` (or `world.set_parallel_enabled(true)`) before stepping to process integrator + solver in parallel.
- **Inspect timers:** run with `RUST_LOG=trace` (or another logger configuration) to see `ScopedTimer` spans such as `contacts::generate`, `islands::build`, and `solver::parallel`.
- **Benchmark changes:** `cargo bench -- world_step` compares sequential and parallel configurations for multiple scene sizes, letting you validate regressions before/after tweaks.

## Author & License
- **Author:** Saptak Santra
- **License:** [Apache License 2.0](LICENSE)

Contributions and issue reports are welcome!
