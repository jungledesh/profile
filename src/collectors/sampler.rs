//! High-frequency GPU sampler thread with graceful shutdown.
//! Samples every 50 ms, stores in fixed-size ring buffer.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use crossbeam_channel::{bounded, RecvTimeoutError, Sender};
use nvml_wrapper::Nvml;

const SAMPLE_INTERVAL_MS: u64 = 50;
const BUFFER_CAPACITY: usize = 1200; // 60 seconds @ 50 ms

#[derive(Clone, Debug)]
pub struct GpuSample {
    pub timestamp: Instant,
    pub sm_util: f32,
    pub power_w: f32,
    pub memory_used_mb: u64, // NEW
}

pub struct SamplerHandle {
    shutdown_tx: Sender<()>,
    join_handle: Option<thread::JoinHandle<()>>,
}

impl Drop for SamplerHandle {
    fn drop(&mut self) {
        let _ = self.shutdown_tx.send(()); // signal shutdown
        if let Some(handle) = self.join_handle.take() {
            let _ = handle.join(); // wait for clean exit
        }
    }
}

pub fn start_sampler() -> SamplerHandle {
    let buffer = Arc::new(Mutex::new(VecDeque::<GpuSample>::with_capacity(
        BUFFER_CAPACITY,
    )));
    let buffer_clone = buffer.clone();

    let (shutdown_tx, shutdown_rx) = bounded(1); // small bounded channel for shutdown

    let join_handle = thread::spawn(move || {
        let nvml = match Nvml::init() {
            Ok(n) => n,
            Err(_) => return,
        };
        let device = match nvml.device_by_index(0) {
            Ok(d) => d,
            Err(_) => return,
        };

        loop {
            match shutdown_rx.recv_timeout(Duration::from_millis(SAMPLE_INTERVAL_MS)) {
                Ok(_) | Err(RecvTimeoutError::Disconnected) => {
                    break;
                }
                Err(RecvTimeoutError::Timeout) => {
                    let ts = Instant::now();

                    let util = device
                        .utilization_rates()
                        .map(|r| r.gpu as f32)
                        .unwrap_or(0.0);

                    let power_w = device
                        .power_usage()
                        .map(|mw| mw as f32 / 1000.0)
                        .unwrap_or(0.0);

                    let memory_used_mb = device
                        .memory_info()
                        .map(|info| info.used / 1024 / 1024)
                        .unwrap_or(0);

                    let mut buf = buffer_clone.lock().unwrap();
                    if buf.len() >= BUFFER_CAPACITY {
                        buf.pop_front();
                    }
                    buf.push_back(GpuSample {
                        timestamp: ts,
                        sm_util: util,
                        power_w,
                        memory_used_mb,
                    });
                }
            }
        }
    });

    SamplerHandle {
        shutdown_tx,
        join_handle: Some(join_handle),
    }
}