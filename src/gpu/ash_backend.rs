use crate::gpu::{ComputeBackend, GpuWorldState};
use ash::vk;
use ash_renderer::vulkan::{Allocator, ComputePipeline, VulkanDevice};
use parking_lot::Mutex;
use std::sync::Arc;

pub struct GpuBuffer {
    pub buffer: vk::Buffer,
    pub allocation: vk_mem::Allocation,
    pub size: u64,
}

pub struct AshBackend {
    pub device: Arc<VulkanDevice>,
    pub allocator: Arc<Allocator>,

    // Buffers for world state (using interior mutability)
    pub position_buffer: Mutex<Option<GpuBuffer>>,
    pub velocity_buffer: Mutex<Option<GpuBuffer>>,
    pub mass_buffer: Mutex<Option<GpuBuffer>>,
    pub bounds_buffer: Mutex<Option<GpuBuffer>>,

    pub broadphase_pipeline: Option<Arc<ComputePipeline>>,
}

impl AshBackend {
    pub fn new(device: Arc<VulkanDevice>) -> Self {
        let allocator = unsafe { Allocator::new(&device).expect("Failed to create VMA allocator") };
        Self {
            device,
            allocator: Arc::new(allocator),
            position_buffer: Mutex::new(None),
            velocity_buffer: Mutex::new(None),
            mass_buffer: Mutex::new(None),
            bounds_buffer: Mutex::new(None),
            broadphase_pipeline: None,
        }
    }

    pub fn set_broadphase_pipeline(&mut self, pipeline: Arc<ComputePipeline>) {
        self.broadphase_pipeline = Some(pipeline);
    }

    fn ensure_buffer(
        &self,
        existing: &mut Option<GpuBuffer>,
        size: u64,
        usage: vk::BufferUsageFlags,
    ) {
        if let Some(buf) = existing {
            if buf.size >= size {
                return;
            }
            // Recreate if too small
            unsafe {
                self.allocator
                    .destroy_buffer(buf.buffer, &mut buf.allocation);
            }
        }

        let (buffer, allocation) = unsafe {
            self.allocator
                .create_buffer(
                    size,
                    usage | vk::BufferUsageFlags::STORAGE_BUFFER,
                    vk_mem::MemoryUsage::AutoPreferHost,
                )
                .expect("Failed to create GPU buffer")
        };
        *existing = Some(GpuBuffer {
            buffer,
            allocation,
            size,
        });
    }
}

impl ComputeBackend for AshBackend {
    fn name(&self) -> &str {
        "vulkan-ash"
    }

    fn prepare_step(&self, state: &GpuWorldState) {
        let count = state.body_count();
        if count == 0 {
            return;
        }

        let pos_size = (state.positions.len() * std::mem::size_of::<glam::Vec3>()) as u64;
        let vel_size = (state.velocities.len() * std::mem::size_of::<glam::Vec3>()) as u64;
        let mass_size = (state.inverse_masses.len() * std::mem::size_of::<f32>()) as u64;
        let bounds_size = (state.collider_bounds.len() * std::mem::size_of::<f32>()) as u64;

        unsafe {
            // Position Sync
            {
                let mut pos_lock = self.position_buffer.lock();
                self.ensure_buffer(&mut pos_lock, pos_size, vk::BufferUsageFlags::empty());
                if let Some(buf) = pos_lock.as_mut() {
                    let ptr = self
                        .allocator
                        .vma
                        .map_memory(&mut buf.allocation)
                        .expect("Failed to map position buffer");
                    std::ptr::copy_nonoverlapping(
                        state.positions.as_ptr(),
                        ptr as *mut glam::Vec3,
                        state.positions.len(),
                    );
                    self.allocator.vma.unmap_memory(&mut buf.allocation);
                }
            }

            // Velocity Sync
            {
                let mut vel_lock = self.velocity_buffer.lock();
                self.ensure_buffer(&mut vel_lock, vel_size, vk::BufferUsageFlags::empty());
                if let Some(buf) = vel_lock.as_mut() {
                    let ptr = self
                        .allocator
                        .vma
                        .map_memory(&mut buf.allocation)
                        .expect("Failed to map velocity buffer");
                    std::ptr::copy_nonoverlapping(
                        state.velocities.as_ptr(),
                        ptr as *mut glam::Vec3,
                        state.velocities.len(),
                    );
                    self.allocator.vma.unmap_memory(&mut buf.allocation);
                }
            }

            // Mass Sync
            {
                let mut mass_lock = self.mass_buffer.lock();
                self.ensure_buffer(&mut mass_lock, mass_size, vk::BufferUsageFlags::empty());
                if let Some(buf) = mass_lock.as_mut() {
                    let ptr = self
                        .allocator
                        .vma
                        .map_memory(&mut buf.allocation)
                        .expect("Failed to map mass buffer");
                    std::ptr::copy_nonoverlapping(
                        state.inverse_masses.as_ptr(),
                        ptr as *mut f32,
                        state.inverse_masses.len(),
                    );
                    self.allocator.vma.unmap_memory(&mut buf.allocation);
                }
            }

            // Bounds Sync
            {
                let mut bounds_lock = self.bounds_buffer.lock();
                self.ensure_buffer(&mut bounds_lock, bounds_size, vk::BufferUsageFlags::empty());
                if let Some(buf) = bounds_lock.as_mut() {
                    let ptr = self
                        .allocator
                        .vma
                        .map_memory(&mut buf.allocation)
                        .expect("Failed to map bounds buffer");
                    std::ptr::copy_nonoverlapping(
                        state.collider_bounds.as_ptr(),
                        ptr as *mut f32,
                        state.collider_bounds.len(),
                    );
                    self.allocator.vma.unmap_memory(&mut buf.allocation);
                }
            }
        }
    }

    fn dispatch_broadphase(&self, _state: &GpuWorldState) {
        // Broadphase shader dispatch logic
    }
}

impl Drop for AshBackend {
    fn drop(&mut self) {
        unsafe {
            if let Some(mut buf) = self.position_buffer.lock().take() {
                self.allocator
                    .destroy_buffer(buf.buffer, &mut buf.allocation);
            }
            if let Some(mut buf) = self.velocity_buffer.lock().take() {
                self.allocator
                    .destroy_buffer(buf.buffer, &mut buf.allocation);
            }
            if let Some(mut buf) = self.mass_buffer.lock().take() {
                self.allocator
                    .destroy_buffer(buf.buffer, &mut buf.allocation);
            }
            if let Some(mut buf) = self.bounds_buffer.lock().take() {
                self.allocator
                    .destroy_buffer(buf.buffer, &mut buf.allocation);
            }
        }
    }
}
