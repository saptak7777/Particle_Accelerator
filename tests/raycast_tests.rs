use particle_accelerator::{
    collision::queries::{Raycast, RaycastQuery},
    core::{
        collider::{Collider, ColliderShape, CollisionFilter},
        rigidbody::RigidBody,
        types::Transform,
    },
    utils::allocator::{Arena, EntityId},
    Vec3,
};

fn add_body(arena: &mut Arena<RigidBody>, position: Vec3) -> EntityId {
    let mut body = RigidBody::new(EntityId::default());
    body.transform.position = position;
    let id = arena.insert(body);
    arena.get_mut(id).unwrap().id = id;
    id
}

fn add_collider(
    arena: &mut Arena<Collider>,
    body_id: EntityId,
    shape: ColliderShape,
    filter: CollisionFilter,
    is_trigger: bool,
) -> EntityId {
    let collider = Collider {
        id: EntityId::default(),
        rigidbody_id: body_id,
        shape,
        offset: Transform::default(),
        is_trigger,
        collision_filter: filter,
    };

    let id = arena.insert(collider);
    arena.get_mut(id).unwrap().id = id;
    id
}

#[test]
fn raycast_filters_layers_and_triggers() {
    let mut bodies = Arena::new();
    let mut colliders = Arena::new();

    let near_body = add_body(&mut bodies, Vec3::new(0.0, 0.0, 5.0));
    let near_collider = add_collider(
        &mut colliders,
        near_body,
        ColliderShape::Sphere { radius: 0.5 },
        CollisionFilter {
            layer: 0b01,
            mask: u32::MAX,
        },
        false,
    );

    let far_body = add_body(&mut bodies, Vec3::new(0.0, 0.0, 10.0));
    add_collider(
        &mut colliders,
        far_body,
        ColliderShape::Sphere { radius: 0.5 },
        CollisionFilter {
            layer: 0b10,
            mask: u32::MAX,
        },
        false,
    );

    let trigger_body = add_body(&mut bodies, Vec3::new(0.0, 0.0, 3.0));
    add_collider(
        &mut colliders,
        trigger_body,
        ColliderShape::Sphere { radius: 0.5 },
        CollisionFilter {
            layer: 0b01,
            mask: u32::MAX,
        },
        true,
    );

    let mut query = RaycastQuery::new(Vec3::ZERO, Vec3::Z, 20.0);
    query.layer_mask = 0b01;
    query.ignore_triggers = true;
    query.closest_only = false;

    let hits = Raycast::cast(&query, &colliders, &bodies);
    assert_eq!(hits.len(), 1, "only near collider should pass filters");
    assert_eq!(
        hits[0].collider_id, near_collider,
        "unexpected collider returned"
    );
}

#[test]
fn raycast_returns_closest_when_requested() {
    let mut bodies = Arena::new();
    let mut colliders = Arena::new();

    let near_body = add_body(&mut bodies, Vec3::new(0.0, 0.0, 4.0));
    add_collider(
        &mut colliders,
        near_body,
        ColliderShape::Sphere { radius: 0.5 },
        CollisionFilter::default(),
        false,
    );

    let far_body = add_body(&mut bodies, Vec3::new(0.0, 0.0, 8.0));
    add_collider(
        &mut colliders,
        far_body,
        ColliderShape::Sphere { radius: 0.5 },
        CollisionFilter::default(),
        false,
    );

    let mut full_query = RaycastQuery::new(Vec3::ZERO, Vec3::Z, 20.0);
    full_query.closest_only = false;
    let mut hits = Raycast::cast(&full_query, &colliders, &bodies);
    assert_eq!(hits.len(), 2);
    assert!(
        hits[0].distance < hits[1].distance,
        "hits should be sorted by distance"
    );

    let mut closest_query = full_query;
    closest_query.closest_only = true;
    hits = Raycast::cast(&closest_query, &colliders, &bodies);
    assert_eq!(hits.len(), 1, "closest_only should collapse to nearest hit");
    assert!(
        hits[0].distance < 5.0,
        "returned hit should be the nearer collider"
    );
}
