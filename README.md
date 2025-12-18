# Particle Accelerator

[![Build Status](https://github.com/saptak7777/Particle_Accelerator/actions/workflows/rust.yml/badge.svg)](https://github.com/saptak7777/Particle_Accelerator/actions)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

**Particle Accelerator** is a high-performance, modular physics engine prototype written in Rust. Designed for scalability and modern hardware, it features a structure-of-arrays (SoA) data layout, SIMD-accelerated narrowphase, and a Vulkan-based GPU broadphase.

## Key Features

- **Massive Scalability**: GPU-accelerated grid broadphase capable of handling 100,000+ entities with zero-copy CPU-GPU synchronization.
- **Advanced Dynamics**: Featherstone's Articulated Body Algorithm (ABA) for $O(n)$ multibody dynamics, including joint limits and motors.
- **Continuous Collision Detection (CCD)**: Robust binary-search TOI and speculative contacts to prevent high-speed tunneling.
- **SoA Architecture**: Optimized memory layout for modern CPU caches and seamless GPU data transfer.
- **Fluent API**: Ergonomic builder patterns for world-building, rigid bodies, and complex colliders.
- **Professional Solver**: Warm-started PGS solver with Coulomb + anisotropic friction, rolling, and torsional resistance.

## Getting Started

### Prerequisites

- **Rust**: Nightly toolchain (recommended for Miri/SIMD) or Stable 1.75+.
- **GPU**: Vulkan 1.2+ capable device (optimized for Intel Arc).

### Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
particle_accelerator = { git = "https://github.com/saptak7777/Particle_Accelerator" }
```

## API Usage

### Creating a Physics World

```rust
use particle_accelerator::world::PhysicsWorld;
use glam::Vec3;

let mut world = PhysicsWorld::builder()
    .time_step(1.0 / 60.0)
    .gravity(Vec3::new(0.0, -9.81, 0.0))
    .parallel_enabled(true)
    .build();
```

### Adding Rigid Bodies & Colliders

```rust
use particle_accelerator::core::rigidbody::RigidBody;
use particle_accelerator::core::collider::Collider;

// Build a dynamic rigid body
let body = RigidBody::builder()
    .position(Vec3::new(0.0, 10.0, 0.0))
    .mass(1.0)
    .build();

let body_id = world.add_rigidbody(body);

// Attach a sphere collider
let collider = Collider::builder()
    .sphere(0.5)
    .restitution(0.3)
    .friction(0.5)
    .build();

world.add_collider_to_body(body_id, collider);
```

### Simulation Loop

```rust
loop {
    world.step(1.0 / 60.0);
    
    if let Some(body) = world.body(body_id) {
        println!("Position: {:?}", body.transform().position);
    }
}
```

## Performance Benchmarks

_Benchmarks performed on **Intel Core i5-11400F** and **Intel Arc A380 (16GB RAM)**._

### Broadphase Scaling (GPU vs CPU)

| Entity Count | CPU Grid (ms) | GPU Accelerated (ms) | Speedup |
| :--- | :--- | :--- | :--- |
| 1,000 | 0.12 | 0.045 | **2.6x** |
| 10,000 | 1.85 | 0.18 | **10.2x** |
| 50,000 | 12.50 | 0.62 | **20.1x** |
| 100,000 | 35.20 | 1.15 | **30.6x** |

> [!TIP]
> GPU acceleration provides logarithmic scaling, making it the ideal choice for massive particle systems or large-scale environment simulations.

## Project Structure

- `src/core`: Fundamental types, SoA storage, and math utilities.
- `src/collision`: GJK/EPA narrowphase, Grid broadphase, and CCD logic.
- `src/dynamics`: Integrators, constraints (Joints), and ABA solver.
- `src/gpu`: Vulkan-based compute backends and GPU state management.
- `src/world`: High-level orchestration and public simulation API.

## Safety & Stability

The project leverages Miri for memory safety verification of its Structure-of-Arrays (SoA) implementation and undergoes continuous fuzz testing for numerical robustness.

## License

Licensed under the [Apache License, Version 2.0](LICENSE).

## Author

**Saptak Santra**
