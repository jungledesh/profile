//! Profiler: orchestrate collectors for `diagnose`.
//!
//! Multi-window aggregation rules: `docs/collection-policy.md`.

use crate::collectors::{self, build_config, window_is_evaluable, window_is_idle};
use crate::context::{RuntimeWindow, StaticContext};
use std::time::{Duration, SystemTime};

mod aggregate;
pub mod delta;
pub mod drift;
pub mod loop_runner;
pub mod poll;
pub mod prompt;
pub mod state;

pub use prompt::MaxNumSeqsPrompt;

#[derive(Debug, Clone)]
pub struct DiagnoseResult {
    pub snapshot: collectors::RawSnapshot,
    pub windows: Vec<RuntimeWindow>,
    pub static_ctx: StaticContext,
    pub duration: Duration,
    pub started_at: SystemTime,
    /// False when every collected window failed `window_is_evaluable` - not an under-load diagnosis.
    pub any_evaluable: bool,
    /// True when every evaluable window is idle (working telemetry, zero traffic).
    /// Non-evaluable windows are ignored for this determination.
    pub all_idle: bool,
    /// Metrics URL passed to `diagnose` (for display when `any_evaluable` is false).
    pub metrics_input: String,
}

pub fn run_diagnose(
    vllm_metrics_input: &str,
    max_num_seqs: Option<u32>,
    cost_per_hour: Option<f64>,
    tensor_parallel_size: u32,
    gpu_indices: Vec<u32>,
    duration: Duration,
) -> anyhow::Result<DiagnoseResult> {
    let started_at = SystemTime::now();
    let metrics_input = vllm_metrics_input.to_string();
    let window = logical_window_size(duration);
    let window_durations = build_window_durations(duration, window);
    let raw_windows = collect_windows(
        vllm_metrics_input,
        &window_durations,
        tensor_parallel_size,
        &gpu_indices,
    )?;
    let any_evaluable = raw_windows.iter().any(window_is_evaluable);
    let all_idle = any_evaluable
        && raw_windows
            .iter()
            .filter(|w| window_is_evaluable(w))
            .all(window_is_idle);
    let snapshot = if raw_windows.is_empty() {
        empty_snapshot(started_at)
    } else if any_evaluable {
        aggregate::aggregate_windows(&raw_windows, &window_durations, started_at)
    } else if let Some(last) = raw_windows.last() {
        context_only_diagnose_snapshot(last, started_at)
    } else {
        empty_snapshot(started_at)
    };
    let mut config = build_config(vllm_metrics_input, &snapshot, max_num_seqs);
    config.cost_per_hour = cost_per_hour;
    config.tensor_parallel_size = Some(tensor_parallel_size);
    let static_ctx = StaticContext::from_snapshot(&snapshot, config);
    let windows: Vec<RuntimeWindow> = raw_windows
        .into_iter()
        .map(RuntimeWindow::from_snapshot)
        .collect();

    Ok(DiagnoseResult {
        snapshot,
        windows,
        static_ctx,
        duration,
        started_at,
        any_evaluable,
        all_idle,
        metrics_input,
    })
}

fn logical_window_size(duration: Duration) -> Duration {
    if duration <= Duration::from_secs(30) {
        Duration::from_secs(2)
    } else {
        Duration::from_secs(10)
    }
}

fn build_window_durations(duration: Duration, logical_window: Duration) -> Vec<Duration> {
    let mut out = Vec::new();
    let total_ms = duration.as_millis();
    let win_ms = logical_window.as_millis();
    let mut elapsed_ms: u128 = 0;
    while elapsed_ms < total_ms {
        let remain = total_ms - elapsed_ms;
        let this_window = Duration::from_millis(remain.min(win_ms) as u64);
        out.push(this_window);
        elapsed_ms += this_window.as_millis();
    }
    out
}

fn track_window_topology(
    baseline: &mut Option<collectors::GpuFingerprint>,
    snap: &collectors::RawSnapshot,
) -> anyhow::Result<()> {
    let fp = snap.fingerprint();
    if let Some(base) = baseline.as_ref() {
        collectors::types::check_topology_drift(base, &fp)?;
    } else {
        *baseline = Some(fp);
    }
    Ok(())
}

fn collect_windows(
    vllm_metrics_input: &str,
    window_durations: &[Duration],
    tensor_parallel_size: u32,
    gpu_indices: &[u32],
) -> anyhow::Result<Vec<collectors::RawSnapshot>> {
    let mut out = Vec::new();
    let mut baseline_fingerprint: Option<collectors::GpuFingerprint> = None;

    for &this_window in window_durations {
        let snap = collectors::collect_snapshot_for_window(
            vllm_metrics_input,
            this_window,
            tensor_parallel_size,
            gpu_indices,
        )?;
        track_window_topology(&mut baseline_fingerprint, &snap)?;

        out.push(snap);
    }
    Ok(out)
}

fn empty_snapshot(at: SystemTime) -> collectors::RawSnapshot {
    collectors::RawSnapshot {
        gpu_observed_at: at,
        vllm_observed_at: at,
        timestamp: at,
        vllm: collectors::VllmRawMetrics::default(),
        gpus: vec![],
    }
}

/// Identity fields only - no runtime gauges. Used when no window was evaluable so we do not imply an under-load diagnosis.
fn context_only_diagnose_snapshot(
    source: &collectors::RawSnapshot,
    at: SystemTime,
) -> collectors::RawSnapshot {
    collectors::RawSnapshot {
        gpu_observed_at: at,
        vllm_observed_at: at,
        timestamp: at,
        vllm: collectors::VllmRawMetrics {
            model_name: source.vllm.model_name.clone(),
            max_num_seqs: source.vllm.max_num_seqs,
            max_num_batched_tokens: source.vllm.max_num_batched_tokens,
            cache_config: source.vllm.cache_config.clone(),
            ..Default::default()
        },
        gpus: source
            .gpus
            .iter()
            .map(|g| collectors::GpuRawMetrics {
                gpu_name: g.gpu_name.clone(),
                gpu_index: g.gpu_index,
                gpu_uuid: g.gpu_uuid.clone(),
                pcie_bus_id: g.pcie_bus_id.clone(),
                power_limit_watts: g.power_limit_watts,
                vram_total_mb: g.vram_total_mb,
                ..Default::default()
            })
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collectors::test_fixtures::snap_with_gpu_indices;
    use crate::collectors::{CacheConfigLabels, GpuRawMetrics, RawSnapshot, VllmRawMetrics};

    #[test]
    fn build_window_durations_includes_partial_tail() {
        let d = build_window_durations(Duration::from_secs(32), Duration::from_secs(10));
        assert_eq!(
            d,
            vec![
                Duration::from_secs(10),
                Duration::from_secs(10),
                Duration::from_secs(10),
                Duration::from_secs(2),
            ]
        );
    }

    #[test]
    fn context_only_diagnose_snapshot_strips_runtime_gauges() {
        let v = VllmRawMetrics {
            model_name: Some("llama".into()),
            max_num_seqs: Some(128),
            num_requests_running: Some(99.0),
            generation_tokens_total: Some(1e9),
            cache_config: CacheConfigLabels {
                block_size: Some(16),
                ..Default::default()
            },
            ..Default::default()
        };
        let g = GpuRawMetrics {
            gpu_name: Some("H100".into()),
            gpu_index: Some(0),
            vram_used_mb: Some(12345),
            vram_total_mb: Some(80000),
            ..Default::default()
        };
        let src = RawSnapshot {
            gpu_observed_at: std::time::SystemTime::UNIX_EPOCH,
            vllm_observed_at: std::time::SystemTime::UNIX_EPOCH,
            timestamp: std::time::SystemTime::UNIX_EPOCH,
            vllm: v,
            gpus: vec![g],
        };
        let out = context_only_diagnose_snapshot(&src, std::time::SystemTime::UNIX_EPOCH);
        assert_eq!(out.vllm.model_name.as_deref(), Some("llama"));
        assert_eq!(out.vllm.max_num_seqs, Some(128));
        assert_eq!(out.vllm.cache_config.block_size, Some(16));
        assert!(out.vllm.num_requests_running.is_none());
        assert!(out.vllm.generation_tokens_total.is_none());
        assert_eq!(
            out.gpus.first().and_then(|g| g.gpu_name.as_deref()),
            Some("H100")
        );
        assert_eq!(out.gpus.first().and_then(|g| g.vram_total_mb), Some(80000));
        assert!(out.gpus.first().and_then(|g| g.vram_used_mb).is_none());
    }

    #[test]
    fn track_window_topology_sets_baseline_on_first_window() {
        let mut baseline = None;
        let snap = snap_with_gpu_indices(&[0, 1]);
        track_window_topology(&mut baseline, &snap).unwrap();
        assert_eq!(baseline, Some(snap.fingerprint()));
    }

    #[test]
    fn track_window_topology_errors_when_gpu_set_changes_mid_pass() {
        let mut baseline = None;
        track_window_topology(&mut baseline, &snap_with_gpu_indices(&[0, 1])).unwrap();
        let err =
            track_window_topology(&mut baseline, &snap_with_gpu_indices(&[0, 2])).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("Topology drift detected"));
        assert!(msg.contains("missing [GPU-1]"));
        assert!(msg.contains("new [GPU-2]"));
    }

    #[test]
    fn all_idle_true_when_evaluable_windows_idle_and_one_non_evaluable() {
        use crate::collectors::{window_is_evaluable, window_is_idle};
        use std::time::SystemTime;

        let t = SystemTime::UNIX_EPOCH;

        let idle_snap = RawSnapshot {
            gpu_observed_at: t,
            vllm_observed_at: t,
            timestamp: t,
            vllm: VllmRawMetrics {
                num_requests_running: Some(0.0),
                generation_tokens_per_sec: Some(0.0),
                window_duration_secs: Some(2.0),
                ..Default::default()
            },
            gpus: vec![],
        };

        let non_eval_snap = RawSnapshot {
            gpu_observed_at: t,
            vllm_observed_at: t,
            timestamp: t,
            vllm: VllmRawMetrics::default(),
            gpus: vec![],
        };

        assert!(window_is_evaluable(&idle_snap));
        assert!(window_is_idle(&idle_snap));
        assert!(!window_is_evaluable(&non_eval_snap));
        assert!(!window_is_idle(&non_eval_snap));

        let windows = [idle_snap.clone(), idle_snap, non_eval_snap];
        let any_evaluable = windows.iter().any(window_is_evaluable);
        let all_idle = any_evaluable
            && windows
                .iter()
                .filter(|w| window_is_evaluable(w))
                .all(window_is_idle);

        assert!(any_evaluable);
        assert!(
            all_idle,
            "all_idle should be true when all evaluable windows are idle"
        );
    }
}
