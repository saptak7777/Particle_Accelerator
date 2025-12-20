# Particle Accelerator – Architecture

## Overview
Particle Accelerator is a modular physics engine built for massively parallel simulation. It uses a Structure-of-Arrays (SoA) layout for high cache residency and zero-copy CPU-GPU data transfers. The engine architecture partitions responsibilities into dedicated subsystems for dynamics, collision, and acceleration.

## Module Layout
```
src/
├── core/        # Core SoA storage (BodiesSoA), EntityId system, and math types.
├── dynamics/    # ABA (Articulated Body Algorithm), PGS Solver, Integrators, and Islands.
├── collision/   # GPU Grid Broadphase, GJK/EPA Narrowphase, CCD (TOI), and ray queries.
├── gpu/         # Vulkan (ash) compute backends for hardware acceleration.
├── utils/       # SIMD math helpers, generational allocators, and profiling tools.
├── world.rs     # PhysicsWorld orchestration and state management.
└── lib.rs       # High-level PhysicsEngine facade and public API.
```

## Core Systems

### 1. Data Residency (SoA)
Rigid bodies are stored in a structure-of-arrays (`BodiesSoA`) which allows the integrator and solver to process millions of entities using contiguous memory access. This layout is directly mirrored to GPU buffers for broadphase acceleration.

### 2. Dynamics Pipeline (ABA & Solver)
The engine supports both high-level rigid bodies and complex articulated multibodies using Featherstone's Articulated Body Algorithm ($O(n)$ complexity). A Projected Gauss-Seidel (PGS) solver handles constraints (joints, motors, limits) with warm-starting for temporal stability.

### 3. Collision Pipeline (CCD & GpuGrid)
- **Broadphase**: A parallel grid-based approach implemented in both CPU (multithreaded) and GPU (Vulkan Compute) variants.
- **Narrowphase**: SIMD-optimized GJK and EPA algorithms for precise penetration depth and manifold generation.
- **CCD**: Continuous Collision Detection using binary-search Time-of-Impact (TOI) and speculative contacts to ensure stability at high velocities.

### 4. Utilities & Profiling
- **SIMD**: Hand-optimized math kernels for common physics operations (`dot`, `cross`, `transform`).
- **Profiling**: Embedded `PhysicsProfiler` tracks per-system execution time, entity counts, and solver metrics.

## Professional Invariants
- **Memory Safety**: No raw pointer aliasing in `get2_mut` (SoA). Validated via Miri.
- **Numerical Stability**: Adaptive sub-stepping and predictive-corrective integration (PCI) minimize energy drift.
- **Deterministic**: Use of fixed-timestep integration and ordered solver passes ensures reproductive simulations.

## Extension Guide
For further development, the `ComputeBackend` trait provides a standardized interface for adding new hardware targets (e.g., CUDA, Metal). The `PhysicsWorldBuilder` serves as the primary entry point for configuring engine features.
