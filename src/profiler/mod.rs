//! Profiler: orchestrate collectors for `diagnose`.
//!
//! Multi-window aggregation rules: **`docs/collection-policy.md`**.

use crate::collectors::{self, build_config, window_is_evaluable, HistogramWindowMass};
use crate::context::{RuntimeWindow, StaticContext};
use std::time::{Duration, SystemTime};

pub mod delta;
pub mod drift;
pub mod loop_runner;
pub mod poll;
pub mod state;

#[derive(Debug, Clone)]
pub struct DiagnoseResult {
    pub snapshot: collectors::RawSnapshot,
    pub windows: Vec<RuntimeWindow>,
    pub static_ctx: StaticContext,
    pub duration: Duration,
    pub started_at: SystemTime,
    /// False when every collected window failed `window_is_evaluable` — not an under-load diagnosis.
    pub any_evaluable: bool,
    /// Metrics URL passed to `diagnose` (for display when `any_evaluable` is false).
    pub metrics_input: String,
}

pub fn run_diagnose(
    vllm_metrics_input: &str,
    max_num_seqs: u32,
    duration: Duration,
) -> anyhow::Result<DiagnoseResult> {
    let started_at = SystemTime::now();
    let metrics_input = vllm_metrics_input.to_string();
    let window = logical_window_size(duration);
    let window_durations = build_window_durations(duration, window);
    let raw_windows = collect_windows(vllm_metrics_input, max_num_seqs, &window_durations)?;
    let any_evaluable = raw_windows.iter().any(window_is_evaluable);
    let snapshot = if raw_windows.is_empty() {
        empty_snapshot(started_at)
    } else if any_evaluable {
        aggregate_windows(&raw_windows, &window_durations, started_at)
    } else {
        context_only_diagnose_snapshot(
            raw_windows
                .last()
                .expect("raw_windows non-empty when not is_empty: checked above"),
            started_at,
        )
    };
    let config = build_config(vllm_metrics_input, &snapshot, max_num_seqs);
    eprintln!(
        "[debug] model_name={:?} model_root={:?}",
        config.model_name, config.model_root
    );
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

/// Identity fields only — no runtime gauges. Used when no window was evaluable so we do not imply an under-load diagnosis.
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
            cache_config: source.vllm.cache_config.clone(),
            ..Default::default()
        },
        gpu: collectors::GpuRawMetrics {
            gpu_name: source.gpu.gpu_name.clone(),
            gpu_index: source.gpu.gpu_index,
            gpu_uuid: source.gpu.gpu_uuid.clone(),
            power_limit_watts: source.gpu.power_limit_watts,
            vram_total_mb: source.gpu.vram_total_mb,
            ..Default::default()
        },
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

    // Running / waiting: duration-weighted mean over evaluable windows (same weight story as gpu_util_pct).
    agg_v.num_requests_running = weighted_metric_pairs(&pairs, |w| w.vllm.num_requests_running);
    agg_v.num_requests_waiting = weighted_metric_pairs(&pairs, |w| w.vllm.num_requests_waiting);
    let kv_avg = aggregate_kv_cache_avg_perc(&pairs);
    agg_v.kv_cache_usage_perc = kv_avg;
    agg_v.kv_cache_avg_perc = kv_avg;
    agg_v.kv_cache_peak_perc = aggregate_kv_cache_peak_perc(&pairs, last);
    agg_v.num_requests_swapped = last.vllm.num_requests_swapped;
    agg_v.cpu_cache_usage_perc = last.vllm.cpu_cache_usage_perc;
    agg_v.ttft_ms = aggregate_histogram_from_mass(&pairs, |v| v.ttft_window_mass, 1000.0)
        .or_else(|| weighted_metric_pairs(&pairs, |w| w.vllm.ttft_ms));
    agg_v.tpot_ms = aggregate_histogram_from_mass(&pairs, |v| v.tpot_window_mass, 1000.0)
        .or_else(|| weighted_metric_pairs(&pairs, |w| w.vllm.tpot_ms));
    agg_v.prefill_latency_ms =
        aggregate_histogram_from_mass(&pairs, |v| v.prefill_window_mass, 1000.0)
            .or_else(|| weighted_metric_pairs(&pairs, |w| w.vllm.prefill_latency_ms));
    agg_v.queue_delay_ms = aggregate_histogram_from_mass(&pairs, |v| v.queue_window_mass, 1000.0)
        .or_else(|| weighted_metric_pairs(&pairs, |w| w.vllm.queue_delay_ms));
    agg_v.prompt_tokens_mean =
        aggregate_histogram_from_mass(&pairs, |v| v.prompt_tokens_window_mass, 1.0)
            .or_else(|| weighted_metric_pairs(&pairs, |w| w.vllm.prompt_tokens_mean));
    let total_window_secs: f64 = pairs.iter().map(|(_, d)| d.as_secs_f64()).sum();
    if total_window_secs.is_finite() && total_window_secs > f64::EPSILON {
        agg_v.window_duration_secs = Some(total_window_secs);
    }
    agg_v.prefill_window_mass = aggregate_histogram_window_mass(&pairs, |v| v.prefill_window_mass);
    agg_v.generation_tokens_per_sec =
        weighted_metric_pairs(&pairs, |w| w.vllm.generation_tokens_per_sec);
    agg_v.request_success_per_sec =
        weighted_metric_pairs(&pairs, |w| w.vllm.request_success_per_sec);
    agg_v.num_preemptions_per_sec =
        weighted_metric_pairs(&pairs, |w| w.vllm.num_preemptions_per_sec);
    let eval_refs: Vec<&collectors::RawSnapshot> = pairs.iter().map(|(w, _)| *w).collect();
    agg_v.prefix_cache_hit_rate = prefix_hit_rate_sum_of_window_deltas(&eval_refs);
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
    agg_g.temperature_peak_c = aggregate_temperature_peak_c(&pairs, last);
    agg_g.sm_clock_mhz = last.gpu.sm_clock_mhz;
    agg_g.vram_used_mb = last.gpu.vram_used_mb;
    agg_g.vram_peak_mb = aggregate_vram_peak_mb(&pairs, last);
    agg_g.vram_total_mb = last.gpu.vram_total_mb;

    collectors::RawSnapshot {
        gpu_observed_at: last.gpu_observed_at,
        vllm_observed_at: last.vllm_observed_at,
        timestamp: last.timestamp,
        vllm: agg_v,
        gpu: agg_g,
    }
}

fn aggregate_kv_cache_avg_perc(pairs: &[(&collectors::RawSnapshot, Duration)]) -> Option<f64> {
    weighted_metric_pairs(pairs, |w| w.vllm.kv_cache_usage_perc)
}

/// `max(max per-window peaks, last evaluable window's landing KV %)` so aggregate peak ≥ displayed usage.
fn aggregate_kv_cache_peak_perc(
    pairs: &[(&collectors::RawSnapshot, Duration)],
    last: &collectors::RawSnapshot,
) -> Option<f64> {
    let from_windows = pairs
        .iter()
        .filter_map(|(w, _)| w.vllm.kv_cache_peak_perc)
        .reduce(|a, b| a.max(b));
    let landing = last.vllm.kv_cache_usage_perc.filter(|x| x.is_finite());
    match (from_windows, landing) {
        (Some(pw), Some(u)) => Some(pw.max(u)),
        (Some(pw), None) => Some(pw),
        (None, Some(u)) => Some(u),
        (None, None) => None,
    }
}

/// `max(max per-window VRAM peaks, last evaluable window's used MiB)` so aggregate peak ≥ displayed used.
fn aggregate_vram_peak_mb(
    pairs: &[(&collectors::RawSnapshot, Duration)],
    last: &collectors::RawSnapshot,
) -> Option<u64> {
    let from_windows = pairs.iter().filter_map(|(w, _)| w.gpu.vram_peak_mb).max();
    let landing = last.gpu.vram_used_mb;
    match (from_windows, landing) {
        (Some(pw), Some(u)) => Some(pw.max(u)),
        (Some(pw), None) => Some(pw),
        (None, Some(u)) => Some(u),
        (None, None) => None,
    }
}

/// `max(max per-window temp peaks, last evaluable landing °C)` so aggregate peak ≥ displayed current.
fn aggregate_temperature_peak_c(
    pairs: &[(&collectors::RawSnapshot, Duration)],
    last: &collectors::RawSnapshot,
) -> Option<f64> {
    let from_windows = pairs
        .iter()
        .filter_map(|(w, _)| w.gpu.temperature_peak_c)
        .filter(|t| t.is_finite())
        .reduce(|a, b| a.max(b));
    let landing = last.gpu.temperature_c.filter(|t| t.is_finite());
    match (from_windows, landing) {
        (Some(pw), Some(u)) => Some(pw.max(u)),
        (Some(pw), None) => Some(pw),
        (None, Some(u)) => Some(u),
        (None, None) => None,
    }
}

/// Sum histogram Δmass across evaluable windows (for saturation gates on aggregated snapshots).
fn aggregate_histogram_window_mass<M>(
    pairs: &[(&collectors::RawSnapshot, Duration)],
    mass: M,
) -> Option<HistogramWindowMass>
where
    M: Fn(&collectors::VllmRawMetrics) -> Option<HistogramWindowMass>,
{
    let mut sum = 0.0_f64;
    let mut count = 0.0_f64;
    for (w, _) in pairs {
        let Some(m) = mass(&w.vllm) else {
            continue;
        };
        if m.count_delta <= 0.0 || m.sum_delta < 0.0 {
            continue;
        }
        if !(m.sum_delta.is_finite() && m.count_delta.is_finite()) {
            continue;
        }
        sum += m.sum_delta;
        count += m.count_delta;
    }
    if count > 0.0 && sum.is_finite() {
        Some(HistogramWindowMass {
            sum_delta: sum,
            count_delta: count,
        })
    } else {
        None
    }
}

/// Multi-window mean for Prometheus histograms: **ΣΔsum / ΣΔcount** over evaluable windows.
/// `scale` converts Prometheus units to display units (1000 for latency seconds→ms, 1 for prompt tokens).
fn aggregate_histogram_from_mass<M>(
    pairs: &[(&collectors::RawSnapshot, Duration)],
    mass: M,
    scale: f64,
) -> Option<f64>
where
    M: Fn(&collectors::VllmRawMetrics) -> Option<HistogramWindowMass>,
{
    let mut sum = 0.0_f64;
    let mut count = 0.0_f64;
    for (w, _) in pairs {
        let Some(m) = mass(&w.vllm) else {
            continue;
        };
        if m.count_delta <= 0.0 || m.sum_delta < 0.0 {
            continue;
        }
        if !(m.sum_delta.is_finite() && m.count_delta.is_finite()) {
            continue;
        }
        sum += m.sum_delta;
        count += m.count_delta;
    }
    if count > 0.0 && sum.is_finite() {
        let mean = (sum / count) * scale;
        mean.is_finite().then_some(mean)
    } else {
        None
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

/// `Σ Δhits / Σ Δqueries` across evaluable windows — each window uses first vs last
/// `prefix_cache_scrape_samples` (same endpoints as `collectors::vllm::prefix_window_hit_rate`).
/// Skips windows with invalid deltas; `None` if no valid query mass.
fn prefix_hit_rate_sum_of_window_deltas(windows: &[&collectors::RawSnapshot]) -> Option<f64> {
    let mut sum_dh = 0.0_f64;
    let mut sum_dq = 0.0_f64;
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
        if !(h0.is_finite() && h1.is_finite() && q0.is_finite() && q1.is_finite()) {
            continue;
        }
        let dq = q1 - q0;
        if dq <= 0.0 {
            continue;
        }
        let dh = h1 - h0;
        if dh < 0.0 || !dh.is_finite() {
            continue;
        }
        sum_dh += dh;
        sum_dq += dq;
    }
    if sum_dq > 0.0 {
        let r = sum_dh / sum_dq;
        r.is_finite().then_some(r)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collectors::{
        CacheConfigLabels, GpuRawMetrics, HistogramWindowMass, PrefixCacheScrapeSample,
        RawSnapshot, VllmRawMetrics,
    };

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
                window_duration_secs: Some(2.0),
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
    fn aggregate_windows_time_weights_rates_latencies_running_waiting_and_state_from_last() {
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
            mk_snap(
                Some(2.0),
                Some(100.0),
                Some((10.0, 20.0)),
                Some((50.0, 130.0)),
                None,
                g1,
                None,
            ),
            mk_snap(
                Some(10.0),
                Some(500.0),
                Some((0.0, 10.0)),
                Some((10.0, 20.0)),
                None,
                g2,
                None,
            ),
        ];
        let durations = vec![Duration::from_secs(2), Duration::from_secs(10)];
        let agg = aggregate_windows(&windows, &durations, SystemTime::UNIX_EPOCH);
        // (2×2s + 10×10s) / 12s — not last window's 10.
        let expected_run = (2.0 * 2.0 + 10.0 * 10.0) / 12.0;
        assert!((agg.vllm.num_requests_running.unwrap() - expected_run).abs() < 1e-9);
        assert!((agg.vllm.generation_tokens_per_sec.unwrap() - 433.3333333).abs() < 1e-4);
        // (10+10)/(80+10) = 20/90 — sum of Δhits / sum of Δqueries, not last window only.
        assert!((agg.vllm.prefix_cache_hit_rate.unwrap() - 20.0 / 90.0).abs() < 1e-9);
        assert!((agg.gpu.gpu_util_pct.unwrap() - 43.3333333).abs() < 1e-4);
        assert_eq!(agg.gpu.vram_used_mb, Some(2000));
        assert!((agg.gpu.temperature_c.unwrap() - 60.0).abs() < 1e-9);
        assert_eq!(agg.gpu.sm_clock_mhz, Some(2000));
    }

    #[test]
    fn aggregate_prefix_hit_rate_is_sum_of_deltas_over_evaluable_windows() {
        // Window1: Δh=10, Δq=50 → alone 20%. Window2: Δh=10, Δq=10 → alone 100%.
        // Last-window-only would report 100%; aggregate = 20/60 ≈ 33.3%.
        let w1 = mk_snap(
            Some(1.0),
            Some(100.0),
            Some((10.0, 20.0)),
            Some((50.0, 100.0)),
            None,
            GpuRawMetrics::default(),
            None,
        );
        let w2 = mk_snap(
            Some(1.0),
            Some(100.0),
            Some((5.0, 15.0)),
            Some((10.0, 20.0)),
            None,
            GpuRawMetrics::default(),
            None,
        );
        let agg = aggregate_windows(
            &[w1, w2],
            &[Duration::from_secs(2), Duration::from_secs(2)],
            SystemTime::UNIX_EPOCH,
        );
        assert!((agg.vllm.prefix_cache_hit_rate.unwrap() - (20.0 / 60.0)).abs() < 1e-9);
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
        let w2 = mk_snap(None, None, None, None, None, g, Some(9999.0));
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
        let w1 = mk_snap(None, None, None, None, None, g.clone(), Some(10.0));
        let w2 = mk_snap(None, None, None, None, None, g, Some(20.0));
        let agg = aggregate_windows(
            &[w1, w2],
            &[Duration::from_secs(2), Duration::from_secs(2)],
            SystemTime::UNIX_EPOCH,
        );
        assert_eq!(agg.vllm.generation_tokens_total, Some(20.0));
        assert!(agg.vllm.num_requests_running.is_none());
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
            gpu_observed_at: SystemTime::UNIX_EPOCH,
            vllm_observed_at: SystemTime::UNIX_EPOCH,
            timestamp: SystemTime::UNIX_EPOCH,
            vllm: v,
            gpu: g,
        };
        let out = context_only_diagnose_snapshot(&src, SystemTime::UNIX_EPOCH);
        assert_eq!(out.vllm.model_name.as_deref(), Some("llama"));
        assert_eq!(out.vllm.max_num_seqs, Some(128));
        assert_eq!(out.vllm.cache_config.block_size, Some(16));
        assert!(out.vllm.num_requests_running.is_none());
        assert!(out.vllm.generation_tokens_total.is_none());
        assert_eq!(out.gpu.gpu_name.as_deref(), Some("H100"));
        assert_eq!(out.gpu.vram_total_mb, Some(80000));
        assert!(out.gpu.vram_used_mb.is_none());
    }

    #[test]
    fn aggregate_peak_kv_and_vram_max_over_evaluable_windows() {
        let v1 = VllmRawMetrics {
            num_requests_running: Some(2.0),
            generation_tokens_per_sec: Some(100.0),
            kv_cache_usage_perc: Some(40.0),
            kv_cache_peak_perc: Some(92.0),
            window_duration_secs: Some(2.0),
            ..Default::default()
        };
        let g1 = GpuRawMetrics {
            vram_used_mb: Some(60 * 1024),
            vram_peak_mb: Some(78 * 1024),
            vram_total_mb: Some(80 * 1024),
            ..Default::default()
        };
        let v2 = VllmRawMetrics {
            num_requests_running: Some(2.0),
            generation_tokens_per_sec: Some(100.0),
            kv_cache_usage_perc: Some(10.0),
            kv_cache_peak_perc: Some(15.0),
            window_duration_secs: Some(2.0),
            ..Default::default()
        };
        let g2 = GpuRawMetrics {
            vram_used_mb: Some(55 * 1024),
            vram_peak_mb: Some(56 * 1024),
            vram_total_mb: Some(80 * 1024),
            ..Default::default()
        };
        let w1 = RawSnapshot {
            gpu_observed_at: SystemTime::UNIX_EPOCH,
            vllm_observed_at: SystemTime::UNIX_EPOCH,
            timestamp: SystemTime::UNIX_EPOCH,
            vllm: v1,
            gpu: g1,
        };
        let w2 = RawSnapshot {
            gpu_observed_at: SystemTime::UNIX_EPOCH,
            vllm_observed_at: SystemTime::UNIX_EPOCH,
            timestamp: SystemTime::UNIX_EPOCH,
            vllm: v2,
            gpu: g2,
        };
        let agg = aggregate_windows(
            &[w1, w2],
            &[Duration::from_secs(2), Duration::from_secs(2)],
            SystemTime::UNIX_EPOCH,
        );
        assert!((agg.vllm.kv_cache_usage_perc.unwrap() - 25.0).abs() < 1e-9);
        assert!((agg.vllm.kv_cache_avg_perc.unwrap() - 25.0).abs() < 1e-9);
        assert!((agg.vllm.kv_cache_peak_perc.unwrap() - 92.0).abs() < 1e-9);
        assert_eq!(agg.gpu.vram_used_mb, Some(55 * 1024));
        assert_eq!(agg.gpu.vram_peak_mb, Some(78 * 1024));
    }

    #[test]
    fn aggregate_peak_kv_and_vram_include_last_landing_when_higher_than_window_peaks() {
        let v1 = VllmRawMetrics {
            num_requests_running: Some(2.0),
            generation_tokens_per_sec: Some(100.0),
            kv_cache_usage_perc: Some(40.0),
            kv_cache_peak_perc: Some(40.0),
            window_duration_secs: Some(2.0),
            ..Default::default()
        };
        let g1 = GpuRawMetrics {
            vram_used_mb: Some(60 * 1024),
            vram_peak_mb: Some(65 * 1024),
            vram_total_mb: Some(80 * 1024),
            ..Default::default()
        };
        let v2 = VllmRawMetrics {
            num_requests_running: Some(2.0),
            generation_tokens_per_sec: Some(100.0),
            kv_cache_usage_perc: Some(95.0),
            kv_cache_peak_perc: Some(50.0),
            window_duration_secs: Some(2.0),
            ..Default::default()
        };
        let g2 = GpuRawMetrics {
            vram_used_mb: Some(72 * 1024),
            vram_peak_mb: Some(56 * 1024),
            vram_total_mb: Some(80 * 1024),
            ..Default::default()
        };
        let w1 = RawSnapshot {
            gpu_observed_at: SystemTime::UNIX_EPOCH,
            vllm_observed_at: SystemTime::UNIX_EPOCH,
            timestamp: SystemTime::UNIX_EPOCH,
            vllm: v1,
            gpu: g1,
        };
        let w2 = RawSnapshot {
            gpu_observed_at: SystemTime::UNIX_EPOCH,
            vllm_observed_at: SystemTime::UNIX_EPOCH,
            timestamp: SystemTime::UNIX_EPOCH,
            vllm: v2,
            gpu: g2,
        };
        let agg = aggregate_windows(
            &[w1, w2],
            &[Duration::from_secs(2), Duration::from_secs(2)],
            SystemTime::UNIX_EPOCH,
        );
        assert!((agg.vllm.kv_cache_usage_perc.unwrap() - 67.5).abs() < 1e-9);
        assert!((agg.vllm.kv_cache_avg_perc.unwrap() - 67.5).abs() < 1e-9);
        assert!((agg.vllm.kv_cache_peak_perc.unwrap() - 95.0).abs() < 1e-9);
        assert_eq!(agg.gpu.vram_used_mb, Some(72 * 1024));
        assert_eq!(agg.gpu.vram_peak_mb, Some(72 * 1024));
    }

    #[test]
    fn aggregate_temperature_peak_folds_in_last_landing_when_higher_than_window_peaks() {
        let v = VllmRawMetrics {
            num_requests_running: Some(2.0),
            generation_tokens_per_sec: Some(100.0),
            window_duration_secs: Some(2.0),
            ..Default::default()
        };
        let g1 = GpuRawMetrics {
            temperature_c: Some(70.0),
            temperature_peak_c: Some(70.0),
            ..Default::default()
        };
        let g2 = GpuRawMetrics {
            temperature_c: Some(88.0),
            temperature_peak_c: Some(75.0),
            ..Default::default()
        };
        let w1 = RawSnapshot {
            gpu_observed_at: SystemTime::UNIX_EPOCH,
            vllm_observed_at: SystemTime::UNIX_EPOCH,
            timestamp: SystemTime::UNIX_EPOCH,
            vllm: v.clone(),
            gpu: g1,
        };
        let w2 = RawSnapshot {
            gpu_observed_at: SystemTime::UNIX_EPOCH,
            vllm_observed_at: SystemTime::UNIX_EPOCH,
            timestamp: SystemTime::UNIX_EPOCH,
            vllm: v,
            gpu: g2,
        };
        let agg = aggregate_windows(
            &[w1, w2],
            &[Duration::from_secs(2), Duration::from_secs(2)],
            SystemTime::UNIX_EPOCH,
        );
        assert!((agg.gpu.temperature_c.unwrap() - 88.0).abs() < 1e-9);
        assert!((agg.gpu.temperature_peak_c.unwrap() - 88.0).abs() < 1e-9);
    }

    #[test]
    fn aggregate_histogram_ttft_is_sum_delta_over_sum_count_not_duration_weighted() {
        let g = GpuRawMetrics::default();
        let v1 = VllmRawMetrics {
            num_requests_running: Some(2.0),
            generation_tokens_per_sec: Some(100.0),
            window_duration_secs: Some(10.0),
            ttft_ms: Some(5000.0),
            ttft_window_mass: Some(HistogramWindowMass {
                sum_delta: 5.0,
                count_delta: 1.0,
            }),
            ..Default::default()
        };
        let v2 = VllmRawMetrics {
            num_requests_running: Some(2.0),
            generation_tokens_per_sec: Some(100.0),
            window_duration_secs: Some(2.0),
            ttft_ms: Some(50.0),
            ttft_window_mass: Some(HistogramWindowMass {
                sum_delta: 25.0,
                count_delta: 500.0,
            }),
            ..Default::default()
        };
        let w1 = RawSnapshot {
            gpu_observed_at: SystemTime::UNIX_EPOCH,
            vllm_observed_at: SystemTime::UNIX_EPOCH,
            timestamp: SystemTime::UNIX_EPOCH,
            vllm: v1,
            gpu: g.clone(),
        };
        let w2 = RawSnapshot {
            gpu_observed_at: SystemTime::UNIX_EPOCH,
            vllm_observed_at: SystemTime::UNIX_EPOCH,
            timestamp: SystemTime::UNIX_EPOCH,
            vllm: v2,
            gpu: g,
        };
        let agg = aggregate_windows(
            &[w1, w2],
            &[Duration::from_secs(10), Duration::from_secs(2)],
            SystemTime::UNIX_EPOCH,
        );
        let expected_ms = (30.0_f64 / 501.0_f64) * 1000.0;
        assert!((agg.vllm.ttft_ms.unwrap() - expected_ms).abs() < 1e-6);
        let duration_weighted_ms = (5000.0 * 10.0 + 50.0 * 2.0) / 12.0;
        assert!((agg.vllm.ttft_ms.unwrap() - duration_weighted_ms).abs() > 100.0);
    }
}
