use crate::collectors::RawSnapshot;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TrafficSource {
    VllmMetrics,
    EstimatedFromScheduler,
}

#[derive(Debug, Clone)]
pub struct TrafficState {
    pub active_requests: Option<f64>,
    pub waiting_requests: Option<f64>,
    /// Request completion rate from `request_success` Δ / scrape window when present.
    pub qps: Option<f64>,
    pub source: TrafficSource,
}

pub fn traffic_from_snapshot(s: &RawSnapshot) -> TrafficState {
    let active = s.vllm.num_requests_running;
    let waiting = s.vllm.num_requests_waiting;
    let source = if s.vllm.generation_tokens_per_sec.is_some() || active.is_some() {
        TrafficSource::VllmMetrics
    } else {
        TrafficSource::EstimatedFromScheduler
    };
    TrafficState {
        active_requests: active,
        waiting_requests: waiting,
        qps: s.vllm.request_success_per_sec,
        source,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collectors::{RawSnapshot, VllmRawMetrics};
    use std::time::SystemTime;

    fn mk_snap(run: Option<f64>, tps: Option<f64>) -> RawSnapshot {
        RawSnapshot {
            gpu_observed_at: SystemTime::UNIX_EPOCH,
            vllm_observed_at: SystemTime::UNIX_EPOCH,
            timestamp: SystemTime::UNIX_EPOCH,
            vllm: VllmRawMetrics {
                num_requests_running: run,
                generation_tokens_per_sec: tps,
                num_requests_waiting: Some(0.0),
                ..Default::default()
            },
            gpus: vec![],
            host_gpu_count: None,
        }
    }

    #[test]
    fn traffic_source_is_vllm_metrics_when_running_present() {
        let t = traffic_from_snapshot(&mk_snap(Some(3.0), None));
        assert_eq!(t.source, TrafficSource::VllmMetrics);
        assert_eq!(t.active_requests, Some(3.0));
    }

    #[test]
    fn traffic_source_is_estimated_when_no_metrics() {
        let t = traffic_from_snapshot(&mk_snap(None, None));
        assert_eq!(t.source, TrafficSource::EstimatedFromScheduler);
    }

    #[test]
    fn qps_follows_request_success_per_sec() {
        let mk = |qps: Option<f64>| RawSnapshot {
            gpu_observed_at: SystemTime::UNIX_EPOCH,
            vllm_observed_at: SystemTime::UNIX_EPOCH,
            timestamp: SystemTime::UNIX_EPOCH,
            vllm: VllmRawMetrics {
                num_requests_running: Some(5.0),
                generation_tokens_per_sec: Some(100.0),
                request_success_per_sec: qps,
                num_requests_waiting: Some(0.0),
                ..Default::default()
            },
            gpus: vec![],
            host_gpu_count: None,
        };
        assert_eq!(traffic_from_snapshot(&mk(Some(12.5))).qps, Some(12.5));
        assert!(traffic_from_snapshot(&mk(None)).qps.is_none());
    }
}
