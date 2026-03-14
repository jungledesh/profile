//! Collectors: GPU util, power draw, token stats. High-freq sampler now.

// Global singleton sampler state (lazy-init, thread-safe via Lazy + Mutex)
// Note: For a real daemon/agent, consider passing a Collectors struct instead of globals.

mod gpu;
mod sampler;
mod tokens;

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use once_cell::sync::Lazy;

use sampler::{start_sampler, GpuSample, SamplerHandle};

#[derive(Debug, Default, Clone)]
pub struct Snapshot {
    pub gpu_name: Option<String>,
    pub gpu_util: Option<f32>,
    pub power_w: Option<f32>,
    pub memory_used_mb: Option<u64>,
    pub tokens_per_sec: Option<f32>,
}

// Global sampler state (lazy init)
static SAMPLER_BUFFER: Lazy<Arc<Mutex<VecDeque<GpuSample>>>> =
    Lazy::new(|| Arc::new(Mutex::new(VecDeque::with_capacity(1200))));

static SAMPLER_HANDLE: Lazy<Mutex<Option<SamplerHandle>>> = Lazy::new(|| Mutex::new(None));

pub fn snapshot() -> Snapshot {
    // Lazy start sampler on first call
    {
        let mut handle_guard = SAMPLER_HANDLE.lock().unwrap();
        if handle_guard.is_none() {
            let handle = start_sampler();
            *handle_guard = Some(handle);
        }
    }

    // Small warmup if no samples yet
    let util = latest_gpu_util()
        .or_else(|| {
            std::thread::sleep(Duration::from_millis(60));
            latest_gpu_util()
        })
        .unwrap_or(0.0);

    let power = latest_power_w()
        .or_else(|| {
            std::thread::sleep(Duration::from_millis(60));
            latest_power_w()
        })
        .unwrap_or(0.0);

    let mem_mb = latest_memory_used()
        .or_else(|| {
            std::thread::sleep(Duration::from_millis(60));
            latest_memory_used()
        })
        .unwrap_or(0);

    Snapshot {
        gpu_name: gpu::gpu_name(),
        gpu_util: Some(util),
        power_w: Some(power),
        memory_used_mb: Some(mem_mb),
        tokens_per_sec: tokens::token_stats(),
    }
}

// Optional: expose for future (e.g. energy calc)
pub fn get_buffer() -> Arc<Mutex<VecDeque<GpuSample>>> {
    SAMPLER_BUFFER.clone()
}

// Safer: read-only access to the most recent sample (if any)
// Returns None if buffer is empty or sampler not started
pub fn with_latest_sample<F, R>(f: F) -> Option<R>
where
    F: FnOnce(&GpuSample) -> R,
{
    let buf = SAMPLER_BUFFER.lock().unwrap();
    buf.back().map(f)
}

// Convenience wrappers (use these in snapshot() etc.)
pub fn latest_gpu_util() -> Option<f32> {
    with_latest_sample(|sample| sample.sm_util)
}

pub fn latest_power_w() -> Option<f32> {
    with_latest_sample(|sample| sample.power_w)
}

pub fn latest_memory_used() -> Option<u64> {
    with_latest_sample(|sample| sample.memory_used_mb)
}
