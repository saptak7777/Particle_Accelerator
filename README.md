# Particle Accelerator

[![Build Status](https://github.com/saptak7777/Particle_Accelerator/actions/workflows/rust.yml/badge.svg)](https://github.com/saptak7777/Particle_Accelerator/actions)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

A physics engine built in Rust that I've been working on. Uses Structure-of-Arrays (SoA) layout and has some GPU acceleration via Vulkan for the broadphase stuff.

## What's in here

- **GPU broadphase** - Grid-based collision detection running on Vulkan compute shaders. Works pretty well for lots of objects (tested up to 100k entities)
- **Articulated bodies** - Featherstone's ABA algorithm for robot/ragdoll simulation. Joint limits and motors are in there too
- **CCD** - Continuous collision detection to catch fast-moving objects. Uses binary search for TOI
- **SoA memory layout** - Better cache performance and makes GPU sync easier
- **PGS solver** - Sequential impulse solver with friction (Coulomb + rolling/torsional)

## Requirements

- Rust 1.75+ (I use nightly for some SIMD stuff but stable should work)
- Vulkan 1.2+ if you want GPU acceleration (tested on Intel Arc A380)

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
particle_accelerator = { git = "https://github.com/saptak7777/Particle_Accelerator" }
```

## How to use it

### Basic setup

```rust
use particle_accelerator::PhysicsWorld;
use particle_accelerator::{RigidBody, Collider, ColliderShape};
use glam::Vec3;

// Create world with 60 FPS timestep
let mut world = PhysicsWorld::new(1.0 / 60.0);

// Gravity is off by default, enable it if you need
world.set_gravity(Vec3::new(0.0, -9.81, 0.0));
```

### Adding bodies

The API is pretty straightforward - create a body, add it to the world, then attach colliders:

```rust
use particle_accelerator::utils::allocator::EntityId;

// Create a dynamic body
let mut body = RigidBody::new(EntityId::from_index(0));
body.transform.position = Vec3::new(0.0, 10.0, 0.0);
body.is_static = false;
body.set_mass(1.0);  // Sets inv_mass internally

let body_id = world.add_rigidbody(body);

// Add a sphere collider
let collider = Collider {
    id: EntityId::from_index(100),
    rigidbody_id: body_id,
    shape: ColliderShape::Sphere { radius: 0.5 },
    offset: Transform::default(),
    is_trigger: false,
    collision_filter: CollisionFilter::default(),
};

world.add_collider(collider);
```

### Running the simulation

```rust
// Main loop
loop {
    world.step(1.0 / 60.0);
    
    // Read body state
    if let Some(body_ref) = world.body(body_id) {
        let pos = body_ref.transform().position;
        println!("Body at: {}, {}, {}", pos.x, pos.y, pos.z);
    }
}
```

### Different collider shapes

```rust
// Box
let box_collider = Collider {
    shape: ColliderShape::Box {
        half_extents: Vec3::new(1.0, 0.5, 0.25)
    },
    // ... other fields
};

// Capsule
let capsule = Collider {
    shape: ColliderShape::Capsule {
        radius: 0.3,
        height: 2.0
    },
    // ...
};

// Compound (multiple shapes)
let compound = Collider {
    shape: ColliderShape::Compound {
        shapes: vec![
            (Transform::from_position(Vec3::ZERO), 
             ColliderShape::Sphere { radius: 0.5 }),
            (Transform::from_position(Vec3::new(0.0, 1.0, 0.0)),
             ColliderShape::Box { half_extents: Vec3::splat(0.3) }),
        ]
    },
    // ...
};
```

### Material properties

```rust
use particle_accelerator::Material;

let mut body = RigidBody::new(EntityId::from_index(0));
body.material = Material {
    restitution: 0.3,      // Bounciness (0 = no bounce, 1 = perfect bounce)
    static_friction: 0.6,
    dynamic_friction: 0.4,
    rolling_friction: 0.01,
    torsional_friction: 0.01,
};
```

## Performance

Tested on my machine (i5-11400F + Arc A380, 16GB RAM). Your mileage may vary:

### Broadphase comparison

| Bodies | CPU Grid | GPU Compute | Speedup |
|--------|----------|-------------|---------|
| 1,000 | 0.12ms | 0.045ms | 2.6x |
| 10,000 | 1.85ms | 0.18ms | 10x |
| 50,000 | 12.5ms | 0.62ms | 20x |
| 100,000 | 35.2ms | 1.15ms | 30x |

GPU really shines when you have lots of objects. For small scenes (< 1000 bodies) the CPU version is actually faster due to dispatch overhead.

## Project layout

```
src/
├── core/          # Basic types, SoA storage, math stuff
├── collision/     # GJK/EPA, broadphase, CCD
├── dynamics/      # Integrator, solver, constraints
├── gpu/           # Vulkan compute shaders
└── world/         # Main API
```

## Known issues

- CCD can miss collisions if objects are moving *really* fast (> 100 m/s)
- GPU broadphase requires Vulkan 1.2+ (won't work on older hardware)
- Compound shapes with lots of children (> 50) can be slow
- No heightfield or triangle mesh support yet

## Testing

I've tested this with Miri for memory safety and run fuzz tests on the GJK/EPA code. Should be pretty stable but if you find bugs please open an issue.

## License

Apache 2.0 - see [LICENSE](LICENSE)

## Author

Saptak Santra
