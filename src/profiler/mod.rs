//! Profiler: orchestrate collectors for `diagnose`.
//!
//! Multi-window aggregation rules: **`docs/collection-policy.md`**.

use crate::collectors::{self, build_config, window_is_evaluable};
use crate::context::{RuntimeWindow, StaticContext};
use std::time::{Duration, SystemTime};

#[derive(Debug, Clone)]
pub struct DiagnoseResult {
    pub snapshot: collectors::RawSnapshot,
    pub windows: Vec<RuntimeWindow>,
    pub static_ctx: StaticContext,
    pub duration: Duration,
    pub started_at: SystemTime,
}

pub fn run_diagnose(
    vllm_metrics_input: &str,
    max_num_seqs: u32,
    duration: Duration,
) -> anyhow::Result<DiagnoseResult> {
    let started_at = SystemTime::now();
    let window = logical_window_size(duration);
    let window_durations = build_window_durations(duration, window);
    let raw_windows = collect_windows(vllm_metrics_input, max_num_seqs, &window_durations)?;
    let snapshot = aggregate_windows(&raw_windows, &window_durations, started_at);
    let config = build_config(vllm_metrics_input, &snapshot, max_num_seqs);
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

fn collect_windows(
    vllm_metrics_input: &str,
    max_num_seqs: u32,
    window_durations: &[Duration],
) -> anyhow::Result<Vec<collectors::RawSnapshot>> {
    let mut out = Vec::new();
    for &this_window in window_durations {
        let snap =
            collectors::collect_snapshot_for_window(vllm_metrics_input, max_num_seqs, this_window)?;
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
        gpu: collectors::GpuRawMetrics::default(),
    }
}

fn aggregate_windows(
    windows: &[collectors::RawSnapshot],
    window_durations: &[Duration],
    started_at: SystemTime,
) -> collectors::RawSnapshot {
    if windows.is_empty() {
        return empty_snapshot(started_at);
    }

    let pairs: Vec<(&collectors::RawSnapshot, Duration)> = windows
        .iter()
        .enumerate()
        .filter_map(|(i, w)| {
            if !window_is_evaluable(w) {
                return None;
            }
            let d = window_durations.get(i).copied()?;
            Some((w, d))
        })
        .collect();

    // Cumulative Prometheus counters: chronologically last collection (idle tail included).
    let chronological_last = windows.last().expect("windows non-empty: checked above");

    if pairs.is_empty() {
        return chronological_last.clone();
    }

    // Last *evaluable* window — state, static, prefix rate, GPU state gauges.
    let (last, _) = pairs.last().expect("pairs non-empty");
    let last = *last;
    let mut agg_v = collectors::VllmRawMetrics {
        model_name: last.vllm.model_name.clone(),
        max_num_seqs: last.vllm.max_num_seqs,
        ..Default::default()
    };
    let mut agg_g = collectors::GpuRawMetrics {
        gpu_name: last.gpu.gpu_name.clone(),
        gpu_index: last.gpu.gpu_index,
        gpu_uuid: last.gpu.gpu_uuid.clone(),
        power_limit_watts: last.gpu.power_limit_watts,
        ..Default::default()
    };

    // State gauges: last evaluable window only (ground truth at end of diagnosis).
    agg_v.num_requests_running = last.vllm.num_requests_running;
    agg_v.num_requests_waiting = last.vllm.num_requests_waiting;
    agg_v.kv_cache_usage_perc = last.vllm.kv_cache_usage_perc;
    agg_v.num_requests_swapped = last.vllm.num_requests_swapped;
    agg_v.cpu_cache_usage_perc = last.vllm.cpu_cache_usage_perc;
    agg_v.ttft_ms = weighted_metric_pairs(&pairs, |w| w.vllm.ttft_ms);
    agg_v.tpot_ms = weighted_metric_pairs(&pairs, |w| w.vllm.tpot_ms);
    agg_v.prefill_latency_ms = weighted_metric_pairs(&pairs, |w| w.vllm.prefill_latency_ms);
    agg_v.queue_delay_ms = weighted_metric_pairs(&pairs, |w| w.vllm.queue_delay_ms);
    agg_v.prompt_tokens_mean = weighted_metric_pairs(&pairs, |w| w.vllm.prompt_tokens_mean);
    agg_v.generation_tokens_per_sec =
        weighted_metric_pairs(&pairs, |w| w.vllm.generation_tokens_per_sec);
    agg_v.request_success_per_sec =
        weighted_metric_pairs(&pairs, |w| w.vllm.request_success_per_sec);
    agg_v.num_preemptions_per_sec =
        weighted_metric_pairs(&pairs, |w| w.vllm.num_preemptions_per_sec);
    agg_v.prefix_cache_hit_rate = last.vllm.prefix_cache_hit_rate;
    agg_v.generation_tokens_total = chronological_last.vllm.generation_tokens_total;
    agg_v.request_success_total = chronological_last.vllm.request_success_total;
    agg_v.num_preemptions_total = chronological_last.vllm.num_preemptions_total;
    agg_v.prefix_cache_scrape_samples = last.vllm.prefix_cache_scrape_samples.clone();
    // Static config labels don't change across windows — carry from last.
    agg_v.cache_config = last.vllm.cache_config.clone();

    agg_g.gpu_util_pct = weighted_metric_pairs(&pairs, |w| w.gpu.gpu_util_pct);
    agg_g.mem_util_pct = weighted_metric_pairs(&pairs, |w| w.gpu.mem_util_pct);
    agg_g.power_watts = weighted_metric_pairs(&pairs, |w| w.gpu.power_watts);
    agg_g.temperature_c = last.gpu.temperature_c;
    agg_g.sm_clock_mhz = last.gpu.sm_clock_mhz;
    agg_g.vram_used_mb = last.gpu.vram_used_mb;
    agg_g.vram_total_mb = last.gpu.vram_total_mb;

    collectors::RawSnapshot {
        gpu_observed_at: last.gpu_observed_at,
        vllm_observed_at: last.vllm_observed_at,
        timestamp: last.timestamp,
        vllm: agg_v,
        gpu: agg_g,
    }
}

fn weighted_metric_pairs<F>(
    pairs: &[(&collectors::RawSnapshot, Duration)],
    metric: F,
) -> Option<f64>
where
    F: Fn(&collectors::RawSnapshot) -> Option<f64>,
{
    let mut weighted_sum = 0.0;
    let mut total_weight_secs = 0.0;
    for (w, dur) in pairs {
        let Some(value) = metric(w) else {
            continue;
        };
        if !value.is_finite() {
            continue;
        }
        let weight_secs = dur.as_secs_f64();
        if weight_secs <= f64::EPSILON {
            continue;
        }
        weighted_sum += value * weight_secs;
        total_weight_secs += weight_secs;
    }
    (total_weight_secs > 0.0).then_some(weighted_sum / total_weight_secs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collectors::{GpuRawMetrics, PrefixCacheScrapeSample, RawSnapshot, VllmRawMetrics};

    fn mk_snap(
        run: Option<f64>,
        tps: Option<f64>,
        hits: Option<(f64, f64)>,
        q: Option<(f64, f64)>,
        prefix_hit_rate: Option<f64>,
        gpu: GpuRawMetrics,
        generation_tokens_total: Option<f64>,
    ) -> RawSnapshot {
        let samples = match (hits, q) {
            (Some((h0, h1)), Some((q0, q1))) => vec![
                PrefixCacheScrapeSample {
                    hits: Some(h0),
                    queries: Some(q0),
                },
                PrefixCacheScrapeSample {
                    hits: Some(h1),
                    queries: Some(q1),
                },
            ],
            _ => vec![],
        };
        RawSnapshot {
            gpu_observed_at: SystemTime::UNIX_EPOCH,
            vllm_observed_at: SystemTime::UNIX_EPOCH,
            timestamp: SystemTime::UNIX_EPOCH,
            vllm: VllmRawMetrics {
                num_requests_running: run,
                generation_tokens_per_sec: tps,
                prefix_cache_hit_rate: prefix_hit_rate,
                prefix_cache_scrape_samples: samples,
                generation_tokens_total,
                ..Default::default()
            },
            gpu,
        }
    }

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
    fn aggregate_windows_time_weights_rates_and_latencies_state_from_last_window() {
        let g1 = GpuRawMetrics {
            gpu_util_pct: Some(10.0),
            vram_used_mb: Some(1000),
            temperature_c: Some(40.0),
            sm_clock_mhz: Some(1000),
            ..Default::default()
        };
        let g2 = GpuRawMetrics {
            gpu_util_pct: Some(50.0),
            vram_used_mb: Some(2000),
            temperature_c: Some(60.0),
            sm_clock_mhz: Some(2000),
            ..Default::default()
        };
        let windows = vec![
            mk_snap(Some(2.0), Some(100.0), None, None, Some(0.1), g1, None),
            mk_snap(Some(10.0), Some(500.0), None, None, Some(0.9), g2, None),
        ];
        let durations = vec![Duration::from_secs(2), Duration::from_secs(10)];
        let agg = aggregate_windows(&windows, &durations, SystemTime::UNIX_EPOCH);
        assert!((agg.vllm.num_requests_running.unwrap() - 10.0).abs() < 1e-9);
        assert!((agg.vllm.generation_tokens_per_sec.unwrap() - 433.3333333).abs() < 1e-4);
        assert!((agg.vllm.prefix_cache_hit_rate.unwrap() - 0.9).abs() < 1e-9);
        assert!((agg.gpu.gpu_util_pct.unwrap() - 43.3333333).abs() < 1e-4);
        assert_eq!(agg.gpu.vram_used_mb, Some(2000));
        assert!((agg.gpu.temperature_c.unwrap() - 60.0).abs() < 1e-9);
        assert_eq!(agg.gpu.sm_clock_mhz, Some(2000));
    }

    #[test]
    fn aggregate_prefix_hit_rate_is_last_window_not_blend_of_earlier() {
        let w1 = mk_snap(
            Some(1.0),
            Some(100.0),
            Some((10.0, 20.0)),
            Some((50.0, 100.0)),
            Some(0.25),
            GpuRawMetrics::default(),
            None,
        );
        let w2 = mk_snap(
            Some(1.0),
            Some(100.0),
            Some((5.0, 15.0)),
            Some((10.0, 20.0)),
            Some(0.75),
            GpuRawMetrics::default(),
            None,
        );
        let agg = aggregate_windows(
            &[w1, w2],
            &[Duration::from_secs(2), Duration::from_secs(2)],
            SystemTime::UNIX_EPOCH,
        );
        assert!((agg.vllm.prefix_cache_hit_rate.unwrap() - 0.75).abs() < 1e-9);
    }

    #[test]
    fn aggregate_cumulative_tokens_from_chronological_last_not_last_evaluable() {
        let g = GpuRawMetrics::default();
        let w1 = mk_snap(
            Some(2.0),
            Some(100.0),
            None,
            None,
            None,
            g.clone(),
            Some(1000.0),
        );
        let w2 = mk_snap(Some(0.5), None, None, None, None, g, Some(9999.0));
        let agg = aggregate_windows(
            &[w1, w2],
            &[Duration::from_secs(2), Duration::from_secs(2)],
            SystemTime::UNIX_EPOCH,
        );
        assert_eq!(agg.vllm.generation_tokens_total, Some(9999.0));
        assert!((agg.vllm.num_requests_running.unwrap() - 2.0).abs() < 1e-9);
    }

    #[test]
    fn aggregate_all_non_evaluable_returns_chronological_last_snapshot() {
        let g = GpuRawMetrics::default();
        let w1 = mk_snap(Some(0.0), None, None, None, None, g.clone(), Some(10.0));
        let w2 = mk_snap(Some(0.0), None, None, None, None, g, Some(20.0));
        let agg = aggregate_windows(
            &[w1, w2],
            &[Duration::from_secs(2), Duration::from_secs(2)],
            SystemTime::UNIX_EPOCH,
        );
        assert_eq!(agg.vllm.generation_tokens_total, Some(20.0));
        assert!((agg.vllm.num_requests_running.unwrap()).abs() < 1e-9);
    }
}
