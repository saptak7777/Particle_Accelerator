# Particle Accelerator

> A modular physics engine that (hopefully) won't explode your game

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Build Status](https://github.com/saptak7777/Particle_Accelerator/actions/workflows/rust.yml/badge.svg)](https://github.com/saptak7777/Particle_Accelerator/actions)

## What This Is (Probably)

Particle Accelerator is a prototype physics engine for Rust. It's built with a Structure-of-Arrays (SoA) architecture and is designed to be "ECS-ready" from the ground up. It handles rigid body dynamics, joint constraints, and collision detection with an optional GPU-accelerated broadphase for when you have too many things on screen.

**Fair Warning**: This is a physics library, not a complete solution. You'll need to handle your own rendering, scene management, and asset loading. Think of it as the suspension for your car—it handles the bumps and collisions, but you still need a chassis and an engine to actually go anywhere.

## Installation

[dependencies]
particle_accelerator = "0.2.0"
glam = "0.28"  # Math types

## Core Concepts

### What It Does

- ✅ **PGS Solver**: Standard Sequential Impulse solver for stable constraints.
- ✅ **Pre-Integration CCD**: Continuous Collision Detection that actually works (fixed tunneling in v0.2.0).
- ✅ **GPU Broadphase**: Vulkan compute shaders for handling 100k+ entities.
- ✅ **Joint Hierarchy**: Supports Fixed, Revolute, and Prismatic joints with motors and limits.
- ✅ **SoA Layout**: Built for cache performance and efficient GPU staging.
- ✅ **Articulated Bodies**: Featherstone's ABA for complex multi-body systems.

### What It Doesn't Do

- ❌ **Rendering** (Use `ash`, `wgpu`, or your favorite renderer).
- ❌ **Soft Bodies** (Everything is solid as a rock).
- ❌ **Fluid Sim** (It's not called Fluid Accelerator for a reason).
- ❌ **Built-in Character Controller** (You'll have to write your own `move_and_slide`).

## Quick Start

### 1. Basic World Setup

```rust
use particle_accelerator::PhysicsWorld;
use glam::Vec3;

fn main() {
    // Timestep is usually 1/60s. 
    let mut world = PhysicsWorld::new(1.0 / 60.0);
    
    // Gravity is off by default.
    world.set_gravity(Vec3::new(0.0, -9.81, 0.0));
}
```

### 2. Spawning a Dynamic Body

```rust
use particle_accelerator::{RigidBody, Collider, ColliderShape};
use particle_accelerator::utils::allocator::EntityId;

fn setup_body(world: &mut PhysicsWorld) {
    let mut body = RigidBody::new(EntityId::from_index(0));
    body.transform.position = Vec3::new(0.0, 10.0, 0.0);
    body.mass_properties.mass = 1.0;
    
    let body_id = world.add_rigidbody(body);

    let collider = Collider::builder()
        .shape(ColliderShape::Sphere { radius: 0.5 })
        .build();

    world.add_collider(body_id, collider);
}
```

### 3. Adding a Prismatic Joint (Slider)

```rust
use particle_accelerator::core::constraints::Joint;

fn add_slider(world: &mut PhysicsWorld, body_a: EntityId, body_b: EntityId) {
    let joint = Joint::Prismatic {
        local_pivot_a: Vec3::ZERO,
        local_pivot_b: Vec3::ZERO,
        local_axis_a: Vec3::X,
        enable_limit: true,
        lower_limit: 0.0,
        upper_limit: 5.0,
        enable_motor: true,
        motor_speed: 2.0,
        max_motor_force: 100.0,
    };
    
    world.add_joint(body_a, body_b, joint);
}
```

## API Reference

### Core Types

**PhysicsWorld**
The main entry point. Orchestrates CCD, collision generation, and constraint solving.

**RigidBody**
Stores physical state (velocity, mass, damping). Uses `inverse_mass = 0.0` for static objects.

**Collider**
The geometric representation. Supports Spheres, Boxes, Capsules, and Compound shapes.

### Systems

**world.step(dt)**
The heart of the engine. In `v0.2.0`, it follows a strict **Solve -> Integrate** order:
1. **Solve CCD**: Detect prospective collisions and clamp velocities.
2. **Solve Constraints**: PGS solver runs for multiple iterations.
3. **Integrate**: Move bodies based on their final velocities.

## Known Limitations & "Features"

- **High-Speed CCD**: While vastly improved in v0.2.0, if you move at 500m/s with a 1hz timestep, things will still break. Physics is a game of numbers, and numbers have limits.
- **GPU Requirements**: The GPU broadphase needs Vulkan 1.2+. If you're on a toaster, the CPU grid fallback works just fine.
- **Compound Shapes**: They work, but keep them reasonable. A compound shape with 100 children is a great way to turn your simulation into a slideshow.

## Performance Notes

What We've Tested:
- ✅ **100k Bodies**: GPU broadphase handles this in ~1.15ms on an Arc A380.
- ✅ **Single-threaded stability**: Solid convergence at 4-8 iterations.
- ✅ **Memory Safety**: Verified with Miri for the SoA core.

What We Haven't Tested:
- ❌ **Massive Joint Chains**: 1000+ linked joints might oscillate without higher iteration counts.
- ❌ **Mobile Hardware**: It's desktop-first for now.

## Troubleshooting

**"My object tunneled through the floor!"**
Check if you're using high velocities and if CCD is enabled for that body. If it is and it still tunnels, your floor might be "too thin" for the timestep.

**"The simulation is oscillating/shaking!"**
Increase your solver iterations (`velocity_iterations` in `PGSSolver`) or decrease your timestep. Physics is sensitive.

## Version History

### v0.2.0 (2025-12-30) - The "Pre-Integration" Edition
- ✅ **Pre-Integration CCD**: Restructured the loop to solve CCD *before* integration. No more "time travel" bugs.
- ✅ **Joint Stability**: Fixed a regression where motors would accelerate 2x faster than requested.
- ✅ **Clippy Compliance**: 100% clean check with zero warnings.

### v0.1.8 - The Initial Release
- Basic PGS Solver and SoA architecture.
- Initial Vulkan compute broadphase implementation.

## Acknowledgments

- **glam**: For making math in Rust less painful.
- **archetype_ecs**: The intended home for this library.

---

Questions? Issues? Open a PR on GitHub. I'll probably look at it eventually. 
Using this in production? You're braver than I am. Let me know how it goes!
