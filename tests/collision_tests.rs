use particle_accelerator::core::soa::BodiesSoA;
use particle_accelerator::utils::allocator::Arena;
use particle_accelerator::*;

fn make_box_body(id: u32, position: Vec3) -> (RigidBody, Collider) {
    let mut body = RigidBody::new(EntityId::from_index(id));
    body.transform.position = position;

    let collider = Collider {
        id: EntityId::from_index(id + 100),
        rigidbody_id: body.id,
        shape: ColliderShape::Box {
            half_extents: Vec3::splat(0.5),
        },
        offset: Transform::default(),
        is_trigger: false,
        collision_filter: CollisionFilter::default(),
    };

    (body, collider)
}

#[test]
fn contact_manifold_detects_box_overlap() {
    let (body_a, collider_a) = make_box_body(0, Vec3::ZERO);
    let (mut body_b, collider_b) = make_box_body(1, Vec3::new(0.4, 0.0, 0.0));
    body_b.is_static = false;

    let manifold = ContactManifold::generate(&collider_a, &body_a, &collider_b, &body_b)
        .expect("overlapping boxes should generate contact");

    assert!(!manifold.points.is_empty());
    assert!(manifold.points[0].depth > 0.0);
}

#[test]
fn box_box_manifold_produces_multiple_points() {
    let (body_a, collider_a) = make_box_body(10, Vec3::ZERO);
    let (mut body_b, collider_b) = make_box_body(11, Vec3::new(0.3, 0.0, 0.2));
    body_b.is_static = false;

    let manifold = ContactManifold::generate(&collider_a, &body_a, &collider_b, &body_b)
        .expect("deep overlap should produce manifold");

    assert!(
        manifold.points.len() >= 2,
        "manifold should contain multiple clipped points, got {}",
        manifold.points.len()
    );
    assert!(
        manifold.points.len() <= 4,
        "manifold should be capped to 4 points"
    );
}

#[test]
fn broadphase_returns_overlapping_pair() {
    let (body_a, mut collider_a) = make_box_body(2, Vec3::ZERO);
    let (body_b, mut collider_b) = make_box_body(3, Vec3::new(0.2, 0.0, 0.0));
    let mut broadphase = BroadPhase::new(1.0);
    let mut bodies = BodiesSoA::new();
    let mut colliders = Arena::new();

    let body_a_id = bodies.insert(body_a);
    let body_b_id = bodies.insert(body_b);

    collider_a.rigidbody_id = body_a_id;
    let collider_a_id = colliders.insert(collider_a);
    colliders.get_mut(collider_a_id).unwrap().id = collider_a_id;
    collider_b.rigidbody_id = body_b_id;
    let collider_b_id = colliders.insert(collider_b);
    colliders.get_mut(collider_b_id).unwrap().id = collider_b_id;

    let expected = (collider_a_id.index(), collider_b_id.index());
    let pairs = broadphase.get_potential_pairs(&colliders, &bodies);
    assert!(
        pairs
            .iter()
            .any(|(a, b)| (a.index(), b.index()) == expected || (b.index(), a.index()) == expected),
        "broadphase missed overlapping colliders"
    );
}
