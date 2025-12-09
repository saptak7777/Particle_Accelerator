# Particle Accelerator

A modular, ECS-ready physics engine prototype written in Rust. Particle Accelerator targets learning and experimentation: it focuses on readable architecture, explicit subsystems, and a clear roadmap for future work.

## Features

- **Generational arena storage** for rigid bodies and colliders, enabling stable `EntityId`s and safe references.
- **Core dynamics pipeline** with force registry, semi-implicit Euler integrator, and contact solver placeholder.
- **Collision detection** including spatial-grid broad phase, GJK/SAT-based narrow phase, and early CCD scaffold.
- **Extensible module layout** (`core`, `dynamics`, `collision`, `utils`, `world`) aligned with ECS-inspired architectures.
- **Examples and benchmarks** demonstrating basic simulation, stacking, and ray casting scenarios.

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

### Benchmarks
```bash
cargo bench
```

## Architecture Highlights
- `PhysicsWorld` orchestrates arenas, forces, integration, broad phase, narrow phase, and constraint solving.
- `utils::allocator::Arena` provides generational IDs and ergonomic iteration helpers.
- `collision` module includes contact manifolds, ray queries, and CCD scaffolding for future expansion.
- `dynamics` module defines the force registry, integrator, solver, and island manager for sleeping logic.

For deeper explanations, see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Roadmap Snapshot
1. Flesh out constraint solving (PGS, joints, friction).
2. Implement advanced collision features (manifold persistence, CCD sweeps, ray casting filters).
3. Add island sleeping optimizations, joint types, and parallel integration via `rayon`.
4. Provide higher-level ECS bindings and rendering hooks.

## Author & License
- **Author:** Saptak Santra
- **License:** [Apache License 2.0](LICENSE)

Contributions and issue reports are welcome!
