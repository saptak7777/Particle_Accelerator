# Particle Accelerator – Architecture

## Overview
Particle Accelerator is a modular physics engine targeting ECS-friendly game and simulation workloads. The crate is organized by responsibility rather than feature size, following a six-phase roadmap that grows the engine from core math primitives to advanced CCD and SIMD pipelines. Each phase expands the same top-level crate, keeping the public API stable while internal subsystems mature.

## Module Layout
```
src/
├── core/        # Data components (RigidBody, Collider, math types)
├── dynamics/    # Integration, forces, solvers, sleeping/islands
├── collision/   # Broad-phase, narrow-phase, queries, CCD
├── utils/       # Allocators, math helpers, logging, SIMD stubs
├── world.rs     # PhysicsWorld orchestrator
└── lib.rs       # Public API + PhysicsEngine facade
```
Key traits:
- `core::types`: wraps glam math, ensuring consistent Transform/Velocity handling.
- `core::rigidbody` / `core::collider`: ECS-ready components referencing `EntityId`s from the generational arena.
- `dynamics::*`: Integrator (semi-implicit Euler), ForceRegistry, ConstraintSolver + PGS stubs, and the Island system for Phase 4 sleeping.
- `collision::*`: Uniform-grid broad-phase, simplified GJK/SAT narrow-phase, contact manifold generation, ray queries, and CCD placeholder.
- `utils::allocator`: Generational arena & IDs to protect against use-after-free as the engine scales.

## Data Flow per Step
1. **Forces & Gravity**: `PhysicsWorld::apply_gravity` and `ForceRegistry` accumulate accelerations.
2. **Broad-phase**: `BroadPhase::get_potential_pairs` bins colliders into a spatial grid.
3. **Narrow-phase**: `ContactManifold::generate` dispatches to SAT or GJK for specific shape pairs.
4. **Constraint Solving**: `ConstraintSolver` (Phase 2) resolves impulses; `PGSSolver` extends this in Phase 4.
5. **Integration**: `Integrator::step` performs velocity + position updates with damping.
6. **Sleeping/Islands (Phase 4)**: `IslandManager` groups connected bodies and toggles awake states.

## Roadmap Alignment
- **Phase 1 (Core)**: Complete – project setup, math types, rigid bodies, colliders, allocator.
- **Phase 2 (Dynamics)**: Integrator, forces, solver skeleton, `PhysicsWorld` glue – implemented.
- **Phase 3 (Collision)**: Broad/narrow-phase, contacts, queries are stubbed/placeholder but wired in.
- **Phase 4 (Stability)**: PGSSolver + islands exist as scaffolding awaiting full integration.
- **Phase 5 (Advanced)**: CCD + ray casting modules stubbed for future expansion.
- **Phase 6 (Optimization)**: SIMD/parallel files created with descriptive placeholders.

## Extending the Engine
Future work hooks should follow the phase order:
1. Replace placeholder IDs with arena lookups in solver/collision.
2. Implement EPA penetration depth + manifold warm-starting.
3. Integrate CCD sweeps and ray queries into `PhysicsWorld::step`.
4. Fill benches/examples/docs for regression tracking.

This document should be updated every time a phase graduates from placeholders to production code.
