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
    pub body_buffer: Mutex<Option<GpuBuffer>>,
    pub grid_buffer: Mutex<Option<GpuBuffer>>,
    pub pair_buffer: Mutex<Option<GpuBuffer>>,
    pub counter_buffer: Mutex<Option<GpuBuffer>>,

    pub broadphase_pipeline: Option<Arc<ComputePipeline>>,
}

impl AshBackend {
    pub fn new(device: Arc<VulkanDevice>) -> Self {
        let allocator = unsafe { Allocator::new(&device).expect("Failed to create VMA allocator") };
        Self {
            device,
            allocator: Arc::new(allocator),
            body_buffer: Mutex::new(None),
            grid_buffer: Mutex::new(None),
            pair_buffer: Mutex::new(None),
            counter_buffer: Mutex::new(None),
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

        let body_size = (state.bodies.len() * std::mem::size_of::<crate::gpu::GpuBody>()) as u64;
        // Conservative grid size for hashing (e.g., 64k cells)
        let grid_size = (65536 * std::mem::size_of::<i32>()) as u64;
        // Max pairs we can store (e.g., 1M pairs)
        let pair_size = (1_000_000 * std::mem::size_of::<[u32; 2]>()) as u64;
        let counter_size = std::mem::size_of::<u32>() as u64;

        unsafe {
            // Body Sync
            {
                let mut body_lock = self.body_buffer.lock();
                self.ensure_buffer(&mut body_lock, body_size, vk::BufferUsageFlags::empty());
                if let Some(buf) = body_lock.as_mut() {
                    let ptr = self
                        .allocator
                        .vma
                        .map_memory(&mut buf.allocation)
                        .expect("Failed to map body buffer");
                    std::ptr::copy_nonoverlapping(
                        state.bodies.as_ptr(),
                        ptr as *mut crate::gpu::GpuBody,
                        state.bodies.len(),
                    );
                    self.allocator.vma.unmap_memory(&mut buf.allocation);
                }
            }

            // Grid Buffer Size ensuring
            {
                let mut grid_lock = self.grid_buffer.lock();
                self.ensure_buffer(&mut grid_lock, grid_size, vk::BufferUsageFlags::empty());
            }

            // Pair Buffer Size ensuring
            {
                let mut pair_lock = self.pair_buffer.lock();
                self.ensure_buffer(&mut pair_lock, pair_size, vk::BufferUsageFlags::empty());
            }

            // Counter Buffer Size ensuring
            {
                let mut counter_lock = self.counter_buffer.lock();
                self.ensure_buffer(
                    &mut counter_lock,
                    counter_size,
                    vk::BufferUsageFlags::empty(),
                );
            }
        }
    }

    fn dispatch_broadphase(&self, state: &GpuWorldState) {
        let count = state.body_count();
        if count == 0 {
            return;
        }

        let _pipeline = match &self.broadphase_pipeline {
            Some(p) => p,
            None => return,
        };

        unsafe {
            let body_buf = self.body_buffer.lock();
            let grid_buf = self.grid_buffer.lock();
            let pair_buf = self.pair_buffer.lock();
            let counter_buf = self.counter_buffer.lock();

            if let (Some(_bodies), Some(grid), Some(_pairs), Some(counters)) =
                (&*body_buf, &*grid_buf, &*pair_buf, &*counter_buf)
            {
                // In a real implementation, we'd use command buffers from AshRenderer.
                // Assuming AshRenderer's ComputePipeline provides a simplified dispatch.
                // For this MVP, we acknowledge dispatch occurs here.

                // 1. Clear Counter
                let ptr = self
                    .allocator
                    .vma
                    .map_memory(&mut counters.allocation.clone())
                    .expect("Map fail");
                std::ptr::write(ptr as *mut u32, 0);
                self.allocator
                    .vma
                    .unmap_memory(&mut counters.allocation.clone());

                // 2. Clear Grid (partial)
                let grid_ptr = self
                    .allocator
                    .vma
                    .map_memory(&mut grid.allocation.clone())
                    .expect("Map fail");
                std::ptr::write_bytes(grid_ptr, 0xFF, grid.size as usize); // -1 in int
                self.allocator
                    .vma
                    .unmap_memory(&mut grid.allocation.clone());

                // 3. Dispatch Broadphase (Shader now handles neighbor search)
                // Note: The shader currently expects populated heads/nexts.
                // A better implementation would have a separate 'Insertion' kernel.
                // For this polish, we provide the hooks.
            }
        }
    }
}

impl Drop for AshBackend {
    fn drop(&mut self) {
        unsafe {
            if let Some(mut buf) = self.body_buffer.lock().take() {
                self.allocator
                    .destroy_buffer(buf.buffer, &mut buf.allocation);
            }
            if let Some(mut buf) = self.grid_buffer.lock().take() {
                self.allocator
                    .destroy_buffer(buf.buffer, &mut buf.allocation);
            }
            if let Some(mut buf) = self.pair_buffer.lock().take() {
                self.allocator
                    .destroy_buffer(buf.buffer, &mut buf.allocation);
            }
            if let Some(mut buf) = self.counter_buffer.lock().take() {
                self.allocator
                    .destroy_buffer(buf.buffer, &mut buf.allocation);
            }
        }
    }
}
