use particle_accelerator::PhysicsWorld;
use std::sync::{Arc, Mutex};
use std::thread;

#[test]
fn test_physics_world_is_sync_and_send() {
    fn assert_sync_send<T: Sync + Send>() {}
    assert_sync_send::<PhysicsWorld>();
}

#[test]
fn test_shared_physics_world_across_threads() {
    let world = Arc::new(Mutex::new(PhysicsWorld::new(1.0 / 60.0)));

    let mut handles = vec![];
    for _ in 0..4 {
        let world_clone = Arc::clone(&world);
        let handle = thread::spawn(move || {
            let mut world = world_clone.lock().unwrap();
            world.step(1.0 / 60.0);
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }
}
