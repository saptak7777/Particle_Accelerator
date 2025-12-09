use particle_accelerator::core::{
    collider::ColliderShape,
    mesh::TriangleMesh,
    types::Transform,
};
use particle_accelerator::{Collider, CollisionFilter, EntityId, RigidBody};

#[test]
fn weld_vertices_reduces_duplicates() {
    let vertices = vec![
        glam::Vec3::new(0.0, 0.0, 0.0),
        glam::Vec3::new(0.0, 0.0, 0.0),
        glam::Vec3::new(1.0, 0.0, 0.0),
        glam::Vec3::new(1.0, 0.0, 0.0),
        glam::Vec3::new(0.0, 1.0, 0.0),
    ];
    let indices = vec![[0, 2, 4]];

    let mesh = TriangleMesh::builder(vertices, indices)
        .weld_vertices(0.01)
        .build();

    // Expect duplicate vertices to be merged.
    assert_eq!(mesh.vertices.len(), 3);
}

#[test]
fn mesh_mass_properties_are_positive() {
    let vertices = vec![
        glam::Vec3::new(-1.0, 0.0, -1.0),
        glam::Vec3::new(1.0, 0.0, -1.0),
        glam::Vec3::new(1.0, 0.0, 1.0),
        glam::Vec3::new(-1.0, 0.0, 1.0),
    ];
    let indices = vec![[0, 1, 2], [0, 2, 3]];

    let mesh = TriangleMesh::builder(vertices, indices).build();
    let props = mesh.approximate_mass_properties(2.0);

    assert!(props.mass > 0.0);
    assert!(props.inertia.determinant() > 0.0);
}

#[test]
fn mesh_collider_bounds_match_shape() {
    let vertices = vec![
        glam::Vec3::new(-2.0, -0.5, -2.0),
        glam::Vec3::new(2.0, -0.5, -2.0),
        glam::Vec3::new(2.0, -0.5, 2.0),
        glam::Vec3::new(-2.0, -0.5, 2.0),
    ];
    let indices = vec![[0, 1, 2], [0, 2, 3]];
    let mesh = TriangleMesh::builder(vertices, indices).build();

    let collider = Collider {
        id: EntityId::from_index(0),
        rigidbody_id: EntityId::from_index(0),
        shape: ColliderShape::Mesh { mesh },
        offset: Transform::default(),
        is_trigger: false,
        collision_filter: CollisionFilter::default(),
    };

    let mut body = RigidBody::new(EntityId::from_index(0));
    body.is_static = true;

    let world_transform = collider.world_transform(&body.transform);
    assert_eq!(world_transform.position, glam::Vec3::ZERO);
    assert!(collider.bounding_radius() >= 2.0);
}
