use log::{Level, log_enabled, warn};
use std::time::{Duration, Instant};

/// Simple scoped timer for profiling critical sections.
pub struct ScopedTimer<'a> {
    label: &'a str,
    start: Instant,
}

impl<'a> ScopedTimer<'a> {
    pub fn new(label: &'a str) -> Self {
        if log_enabled!(Level::Trace) {
            log::trace!("⏱️ start {label}");
        }
        Self {
            label,
            start: Instant::now(),
        }
    }
}

impl<'a> Drop for ScopedTimer<'a> {
    fn drop(&mut self) {
        if log_enabled!(Level::Trace) {
            let elapsed = self.start.elapsed();
            log::trace!("⏱️ end {} ({} µs)", self.label, elapsed.as_micros());
        }
    }
}

/// Registers a warning when frame budget is exceeded.
pub fn warn_if_frame_budget_exceeded(duration: Duration, budget_ms: f32) {
    if duration.as_secs_f32() * 1000.0 > budget_ms {
        warn!(
            "Frame exceeded budget: {:.2} ms > {:.2} ms",
            duration.as_secs_f32() * 1000.0,
            budget_ms
        );
    }
}
