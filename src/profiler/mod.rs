//! Profiler: orchestrate collectors for `diagnose`.

use crate::collectors::{self, window_is_evaluable};
use std::time::{Duration, SystemTime};

#[derive(Debug, Clone)]
pub struct DiagnoseResult {
    pub snapshot: collectors::RawSnapshot,
    pub windows: Vec<collectors::RawSnapshot>,
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
    let windows = collect_windows(vllm_metrics_input, max_num_seqs, &window_durations)?;
    let snapshot = aggregate_windows(&windows, &window_durations, started_at);

    Ok(DiagnoseResult {
        snapshot,
        windows,
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

fn aggregate_windows(
    windows: &[collectors::RawSnapshot],
    window_durations: &[Duration],
    started_at: SystemTime,
) -> collectors::RawSnapshot {
    if windows.is_empty() {
        return collectors::RawSnapshot {
            gpu_observed_at: started_at,
            vllm_observed_at: started_at,
            timestamp: started_at,
            vllm: collectors::VllmRawMetrics::default(),
            gpu: collectors::GpuRawMetrics::default(),
        };
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

    if pairs.is_empty() {
        return collectors::RawSnapshot {
            gpu_observed_at: started_at,
            vllm_observed_at: started_at,
            timestamp: started_at,
            vllm: collectors::VllmRawMetrics::default(),
            gpu: collectors::GpuRawMetrics::default(),
        };
    }

    let last = pairs.last().expect("non-empty evaluable").0;
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

    agg_v.num_requests_running = weighted_metric_pairs(&pairs, |w| w.vllm.num_requests_running);
    agg_v.num_requests_waiting = weighted_metric_pairs(&pairs, |w| w.vllm.num_requests_waiting);
    agg_v.kv_cache_usage_perc = weighted_metric_pairs(&pairs, |w| w.vllm.kv_cache_usage_perc);
    agg_v.ttft_ms = weighted_metric_pairs(&pairs, |w| w.vllm.ttft_ms);
    agg_v.tpot_ms = weighted_metric_pairs(&pairs, |w| w.vllm.tpot_ms);
    agg_v.prefill_latency_ms = weighted_metric_pairs(&pairs, |w| w.vllm.prefill_latency_ms);
    agg_v.queue_delay_ms = weighted_metric_pairs(&pairs, |w| w.vllm.queue_delay_ms);
    agg_v.prompt_tokens_mean = weighted_metric_pairs(&pairs, |w| w.vllm.prompt_tokens_mean);
    agg_v.generation_tokens_per_sec =
        weighted_metric_pairs(&pairs, |w| w.vllm.generation_tokens_per_sec);
    let eval_refs: Vec<&collectors::RawSnapshot> = pairs.iter().map(|(w, _)| *w).collect();
    agg_v.prefix_cache_hit_rate = prefix_hit_rate_from_windows(&eval_refs);
    agg_v.generation_tokens_total = last.vllm.generation_tokens_total;
    agg_v.prefix_cache_scrape_samples = last.vllm.prefix_cache_scrape_samples.clone();

    agg_g.gpu_util_pct = weighted_metric_pairs(&pairs, |w| w.gpu.gpu_util_pct);
    agg_g.mem_util_pct = weighted_metric_pairs(&pairs, |w| w.gpu.mem_util_pct);
    agg_g.power_watts = weighted_metric_pairs(&pairs, |w| w.gpu.power_watts);
    agg_g.temperature_c = weighted_metric_pairs(&pairs, |w| w.gpu.temperature_c);
    agg_g.sm_clock_mhz = weighted_metric_pairs(&pairs, |w| w.gpu.sm_clock_mhz.map(|x| x as f64))
        .map(|x| x.round() as u32);
    agg_g.vram_used_mb = weighted_metric_pairs(&pairs, |w| w.gpu.vram_used_mb.map(|x| x as f64))
        .map(|x| x.round() as u64);
    agg_g.vram_total_mb = pairs.iter().filter_map(|(w, _)| w.gpu.vram_total_mb).max();

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

fn prefix_hit_rate_from_windows(windows: &[&collectors::RawSnapshot]) -> Option<f64> {
    let mut total_hits = 0.0;
    let mut total_queries = 0.0;
    let mut any = false;
    for w in windows {
        let samples = &w.vllm.prefix_cache_scrape_samples;
        if samples.len() < 2 {
            continue;
        }
        let first = &samples[0];
        let last = &samples[samples.len() - 1];
        let (Some(h0), Some(h1), Some(q0), Some(q1)) =
            (first.hits, last.hits, first.queries, last.queries)
        else {
            continue;
        };
        let dh = h1 - h0;
        let dq = q1 - q0;
        if dh >= 0.0 && dq > 0.0 && dh.is_finite() && dq.is_finite() {
            total_hits += dh;
            total_queries += dq;
            any = true;
        }
    }
    if any && total_queries > 0.0 {
        Some(total_hits / total_queries)
    } else {
        None
    }
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
    ) -> RawSnapshot {
        let samples = match (hits, q) {
            (Some((h0, h1)), Some((q0, q1))) => vec![
                PrefixCacheScrapeSample {
                    hits: Some(h0),
                    queries: Some(q0),
                    misses: None,
                },
                PrefixCacheScrapeSample {
                    hits: Some(h1),
                    queries: Some(q1),
                    misses: None,
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
                prefix_cache_scrape_samples: samples,
                ..Default::default()
            },
            gpu: GpuRawMetrics::default(),
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
    fn aggregate_windows_uses_time_weighted_means() {
        let windows = vec![
            mk_snap(Some(2.0), Some(100.0), None, None),
            mk_snap(Some(10.0), Some(500.0), None, None),
        ];
        let durations = vec![Duration::from_secs(2), Duration::from_secs(10)];
        let agg = aggregate_windows(&windows, &durations, SystemTime::UNIX_EPOCH);
        let run = agg.vllm.num_requests_running.expect("run");
        let tps = agg.vllm.generation_tokens_per_sec.expect("tps");
        assert!((run - 8.6666667).abs() < 1e-4);
        assert!((tps - 433.3333333).abs() < 1e-4);
    }

    #[test]
    fn prefix_hit_rate_recomputed_from_summed_deltas() {
        let w1 = mk_snap(
            Some(1.0),
            Some(100.0),
            Some((10.0, 20.0)),
            Some((50.0, 100.0)),
        );
        let w2 = mk_snap(
            Some(1.0),
            Some(100.0),
            Some((5.0, 15.0)),
            Some((10.0, 20.0)),
        );
        let refs = vec![&w1, &w2];
        let r = prefix_hit_rate_from_windows(&refs).expect("ratio");
        // (10 + 10) / (50 + 10) = 0.3333
        assert!((r - (20.0 / 60.0)).abs() < 1e-9);
    }
}
