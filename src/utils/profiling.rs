use std::time::{Duration, Instant};

/// Global or Thread-Local profiler state would be ideal, but for now
/// we'll attach it to the PhysicsWorld and pass it down or access it via a static if needed.
/// Simple structure to hold frame timing data.
#[derive(Debug, Default, Clone, Copy)]
pub struct PhysicsProfiler {
    pub broad_phase_time: Duration,
    pub narrow_phase_time: Duration,
    pub solver_time: Duration,
    pub integrator_time: Duration,
    pub total_frame_time: Duration,

    pub body_count: usize,
    pub contact_count: usize,
    pub active_island_count: usize,
}

impl PhysicsProfiler {
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    pub fn report(&self) {
        let total_us = self.total_frame_time.as_micros() as f32;
        if total_us < 1.0 {
            return;
        }

        println!("--- Physics Profile ---");
        println!(
            "Bodies: {}, Contacts: {}, Islands: {}",
            self.body_count, self.contact_count, self.active_island_count
        );

        println!(
            "Total Frame: {:.2} ms",
            self.total_frame_time.as_secs_f32() * 1000.0
        );

        println!(
            "  Broad Phase:  {:.2} ms ({:.1}%)",
            self.broad_phase_time.as_secs_f32() * 1000.0,
            (self.broad_phase_time.as_micros() as f32 / total_us) * 100.0
        );

        println!(
            "  Narrow Phase: {:.2} ms ({:.1}%)",
            self.narrow_phase_time.as_secs_f32() * 1000.0,
            (self.narrow_phase_time.as_micros() as f32 / total_us) * 100.0
        );

        println!(
            "  Solver:       {:.2} ms ({:.1}%)",
            self.solver_time.as_secs_f32() * 1000.0,
            (self.solver_time.as_micros() as f32 / total_us) * 100.0
        );

        println!(
            "  Integrator:   {:.2} ms ({:.1}%)",
            self.integrator_time.as_secs_f32() * 1000.0,
            (self.integrator_time.as_micros() as f32 / total_us) * 100.0
        );
        println!("-----------------------");
    }
}

pub struct ScopedTimer<'a> {
    start: Instant,
    output: &'a mut Duration,
}

impl<'a> ScopedTimer<'a> {
    pub fn new(output: &'a mut Duration) -> Self {
        Self {
            start: Instant::now(),
            output,
        }
    }
}

impl<'a> Drop for ScopedTimer<'a> {
    fn drop(&mut self) {
        *self.output += self.start.elapsed();
    }
}
