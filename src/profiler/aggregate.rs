//! Window aggregation: collapse `&[RawSnapshot]` into a single summary snapshot.
//!
//! Engine-agnostic - operates on `RawSnapshot` only. Supporting a new inference
//! engine means writing a collector that produces `RawSnapshot`; nothing here changes.
//! Hardware collector changes (ROCm, Gaudi) are similarly isolated to `collectors/gpu.rs`.

use std::time::{Duration, SystemTime};

use crate::collectors::{
    self, HistogramWindowMass, observations_aligned, window_is_active, window_is_evaluable,
};

/// Aggregate a slice of per-window snapshots into a single summary snapshot.
/// Returns `chronological_last` unchanged when no window is evaluable.
///
/// Called once per diagnose run after collection stops, not in the 250ms hot
/// loop. The owned clones below build the summary snapshot from borrowed windows;
/// they run once over a handful of windows, not per-sample. Intentional.
pub(super) fn aggregate_windows(
    windows: &[collectors::RawSnapshot],
    window_durations: &[Duration],
    started_at: SystemTime,
) -> collectors::RawSnapshot {
    if windows.is_empty() {
        return empty_aggregate(started_at);
    }

    let evaluable_pairs: Vec<(&collectors::RawSnapshot, Duration)> = windows
        .iter()
        .enumerate()
        .filter_map(|(i, w)| {
            if !window_is_evaluable(w) {
                return None;
            }
            let d = w
                .vllm
                .window_duration_secs
                .filter(|s| s.is_finite() && *s > f64::EPSILON)
                .map(Duration::from_secs_f64)
                .or_else(|| window_durations.get(i).copied())?;
            Some((w, d))
        })
        .collect();

    let active_pairs: Vec<(&collectors::RawSnapshot, Duration)> = evaluable_pairs
        .iter()
        .copied()
        .filter(|(w, _)| window_is_active(w))
        .collect();
    let energy_aligned_pairs: Vec<(&collectors::RawSnapshot, Duration)> = active_pairs
        .iter()
        .copied()
        .filter(|(w, _)| observations_aligned(w))
        .collect();

    // Cumulative Prometheus counters: chronologically last collection (idle tail included).
    let chronological_last = match windows.last() {
        Some(w) => w,
        None => return collectors::RawSnapshot::default(),
    };

    if evaluable_pairs.is_empty() {
        return chronological_last.clone();
    }

    // Last *evaluable* window - state, static, prefix rate, GPU state gauges.
    let (last, _) = match evaluable_pairs.last() {
        Some(p) => p,
        None => return chronological_last.clone(),
    };
    let last = *last;
    let mut agg_v = collectors::VllmRawMetrics {
        model_name: last.vllm.model_name.clone(),
        max_num_seqs: last.vllm.max_num_seqs,
        ..Default::default()
    };
    let mut agg_gpus = last.gpus.clone();

    // Running / waiting: duration-weighted mean over active windows. None if no active windows.
    agg_v.num_requests_running = weighted_mean(&active_pairs, |w| w.vllm.num_requests_running);
    agg_v.num_requests_waiting = weighted_mean(&active_pairs, |w| w.vllm.num_requests_waiting);
    let kv_avg = weighted_mean(&evaluable_pairs, |w| w.vllm.kv_cache_usage_perc);
    agg_v.kv_cache_usage_perc = kv_avg;
    agg_v.kv_cache_avg_perc = kv_avg;
    agg_v.kv_cache_peak_perc = kv_cache_peak_perc(&evaluable_pairs, last);
    agg_v.num_requests_swapped = last.vllm.num_requests_swapped;
    agg_v.cpu_cache_usage_perc = last.vllm.cpu_cache_usage_perc;
    agg_v.ttft_ms = histogram_mean(&active_pairs, |v| v.ttft_window_mass, 1000.0)
        .or_else(|| weighted_mean(&active_pairs, |w| w.vllm.ttft_ms));
    agg_v.tpot_ms = histogram_mean(&active_pairs, |v| v.tpot_window_mass, 1000.0)
        .or_else(|| weighted_mean(&active_pairs, |w| w.vllm.tpot_ms));
    // p99/p95: merge per-window delta bucket vectors, recompute quantile from merged result.
    // Averaging quantiles across windows is not mathematically correct.
    {
        let ttft_vecs: Vec<&[collectors::HistogramCount]> = active_pairs
            .iter()
            .map(|(w, _)| w.vllm.ttft_p99_buckets.as_slice())
            .collect();
        let merged_ttft = collectors::merge_p99_bucket_vecs(&ttft_vecs);
        agg_v.ttft_p99_ms =
            collectors::vllm::histogram_quantile(0.99, &merged_ttft).map(|s| s * 1000.0);
        agg_v.ttft_p95_ms =
            collectors::vllm::histogram_quantile(0.95, &merged_ttft).map(|s| s * 1000.0);

        let tpot_vecs: Vec<&[collectors::HistogramCount]> = active_pairs
            .iter()
            .map(|(w, _)| w.vllm.tpot_p99_buckets.as_slice())
            .collect();
        let merged_tpot = collectors::merge_p99_bucket_vecs(&tpot_vecs);
        agg_v.tpot_p99_ms =
            collectors::vllm::histogram_quantile(0.99, &merged_tpot).map(|s| s * 1000.0);
        agg_v.tpot_p95_ms =
            collectors::vllm::histogram_quantile(0.95, &merged_tpot).map(|s| s * 1000.0);

        let prompt_tok_vecs: Vec<&[collectors::HistogramCount]> = evaluable_pairs
            .iter()
            .map(|(w, _)| w.vllm.prompt_tokens_p99_buckets.as_slice())
            .collect();
        let merged_prompt_tok = collectors::merge_p99_bucket_vecs(&prompt_tok_vecs);
        agg_v.prompt_tokens_p99 = collectors::vllm::histogram_quantile(0.99, &merged_prompt_tok);

        let gen_tok_vecs: Vec<&[collectors::HistogramCount]> = evaluable_pairs
            .iter()
            .map(|(w, _)| w.vllm.generation_tokens_p99_buckets.as_slice())
            .collect();
        let merged_gen_tok = collectors::merge_p99_bucket_vecs(&gen_tok_vecs);
        agg_v.generation_tokens_p99 = collectors::vllm::histogram_quantile(0.99, &merged_gen_tok);
        agg_v.generation_tokens_completed =
            merged_gen_tok.last().map(|b| b.count).filter(|c| *c > 0.0);
    }
    agg_v.prefill_latency_ms = histogram_mean(&active_pairs, |v| v.prefill_window_mass, 1000.0)
        .or_else(|| weighted_mean(&active_pairs, |w| w.vllm.prefill_latency_ms));
    agg_v.queue_delay_ms = histogram_mean(&active_pairs, |v| v.queue_window_mass, 1000.0)
        .or_else(|| weighted_mean(&active_pairs, |w| w.vllm.queue_delay_ms));
    agg_v.prompt_tokens_mean =
        histogram_mean(&evaluable_pairs, |v| v.prompt_tokens_window_mass, 1.0)
            .or_else(|| weighted_mean(&evaluable_pairs, |w| w.vllm.prompt_tokens_mean));
    let total_window_secs: f64 = evaluable_pairs.iter().map(|(_, d)| d.as_secs_f64()).sum();
    if total_window_secs.is_finite() && total_window_secs > f64::EPSILON {
        agg_v.window_duration_secs = Some(total_window_secs);
    }
    agg_v.prefill_window_mass = accumulate_histogram_mass(&active_pairs, |v| v.prefill_window_mass);
    agg_v.ttft_window_mass = accumulate_histogram_mass(&active_pairs, |v| v.ttft_window_mass);
    agg_v.tpot_window_mass = accumulate_histogram_mass(&active_pairs, |v| v.tpot_window_mass);
    agg_v.queue_window_mass = accumulate_histogram_mass(&active_pairs, |v| v.queue_window_mass);
    agg_v.prompt_tokens_window_mass =
        accumulate_histogram_mass(&evaluable_pairs, |v| v.prompt_tokens_window_mass);
    agg_v.generation_tokens_per_sec =
        weighted_mean(&active_pairs, |w| w.vllm.generation_tokens_per_sec);
    agg_v.request_success_per_sec =
        weighted_mean(&active_pairs, |w| w.vllm.request_success_per_sec);
    agg_v.num_preemptions_per_sec =
        weighted_mean(&active_pairs, |w| w.vllm.num_preemptions_per_sec);
    let eval_refs: Vec<&collectors::RawSnapshot> =
        evaluable_pairs.iter().map(|(w, _)| *w).collect();
    agg_v.prefix_cache_hit_rate = prefix_hit_rate_sum_of_deltas(&eval_refs);
    agg_v.generation_tokens_total = chronological_last.vllm.generation_tokens_total;
    agg_v.request_success_total = chronological_last.vllm.request_success_total;
    agg_v.num_preemptions_total = chronological_last.vllm.num_preemptions_total;
    agg_v.prefix_cache_scrape_samples = last.vllm.prefix_cache_scrape_samples.clone();
    // Static config labels don't change across windows - carry from last.
    agg_v.cache_config = last.vllm.cache_config.clone();

    // Slot index is stable across windows: collect sorts gpus by identity() before storing,
    // so idx N always refers to the same physical GPU in every window.
    for (idx, agg_g) in agg_gpus.iter_mut().enumerate() {
        agg_g.gpu_util_pct = weighted_mean(&active_pairs, |w| {
            w.gpus.get(idx).and_then(|g| g.gpu_util_pct)
        });
        agg_g.mem_util_pct = weighted_mean(&active_pairs, |w| {
            w.gpus.get(idx).and_then(|g| g.mem_util_pct)
        });
        agg_g.power_watts = weighted_mean(&active_pairs, |w| {
            w.gpus.get(idx).and_then(|g| g.power_watts)
        });
        // Energy/cost join: only windows whose GPU and vLLM clocks align.
        agg_g.aligned_power_watts = weighted_mean(&energy_aligned_pairs, |w| {
            w.gpus.get(idx).and_then(|g| g.power_watts)
        });
        agg_g.temperature_c = last.gpus.get(idx).and_then(|g| g.temperature_c);
        agg_g.sm_clock_mhz = last.gpus.get(idx).and_then(|g| g.sm_clock_mhz);
        agg_g.vram_used_mb = last.gpus.get(idx).and_then(|g| g.vram_used_mb);
        agg_g.vram_peak_mb = vram_peak_mb_slot(&evaluable_pairs, last, idx);
        agg_g.temperature_peak_c = temperature_peak_c_slot(&evaluable_pairs, last, idx);
        agg_g.vram_total_mb = last.gpus.get(idx).and_then(|g| g.vram_total_mb);
    }

    collectors::RawSnapshot {
        gpu_observed_at: last.gpu_observed_at,
        vllm_observed_at: last.vllm_observed_at,
        timestamp: last.timestamp,
        vllm: agg_v,
        gpus: agg_gpus,
    }
}

fn empty_aggregate(at: SystemTime) -> collectors::RawSnapshot {
    collectors::RawSnapshot {
        gpu_observed_at: at,
        vllm_observed_at: at,
        timestamp: at,
        ..Default::default()
    }
}

/// `max(a, b)` for `Option<f64>` - `None` treated as absent, not zero.
#[inline]
fn max_option_f64(a: Option<f64>, b: Option<f64>) -> Option<f64> {
    match (a, b) {
        (Some(x), Some(y)) => Some(x.max(y)),
        _ => a.or(b),
    }
}

/// `max(a, b)` for `Option<u64>` - `None` treated as absent, not zero.
#[inline]
fn max_option_u64(a: Option<u64>, b: Option<u64>) -> Option<u64> {
    match (a, b) {
        (Some(x), Some(y)) => Some(x.max(y)),
        _ => a.or(b),
    }
}

/// `max(per-window peaks, last-evaluable landing KV%)` - aggregate peak ≥ displayed usage.
fn kv_cache_peak_perc(
    pairs: &[(&collectors::RawSnapshot, Duration)],
    last: &collectors::RawSnapshot,
) -> Option<f64> {
    let from_windows = pairs
        .iter()
        .filter_map(|(w, _)| w.vllm.kv_cache_peak_perc)
        .reduce(|a, b| a.max(b));
    let landing = last.vllm.kv_cache_usage_perc.filter(|x| x.is_finite());
    max_option_f64(from_windows, landing)
}

/// `max(per-window VRAM peaks, last-evaluable used MiB)` - aggregate peak ≥ displayed used.
fn vram_peak_mb_slot(
    pairs: &[(&collectors::RawSnapshot, Duration)],
    last: &collectors::RawSnapshot,
    slot: usize,
) -> Option<u64> {
    let from_windows = pairs
        .iter()
        .filter_map(|(w, _)| w.gpus.get(slot).and_then(|g| g.vram_peak_mb))
        .max();
    let landing = last.gpus.get(slot).and_then(|g| g.vram_used_mb);
    max_option_u64(from_windows, landing)
}

/// `max(per-window temp peaks, last-evaluable landing °C)` - aggregate peak ≥ displayed current.
fn temperature_peak_c_slot(
    pairs: &[(&collectors::RawSnapshot, Duration)],
    last: &collectors::RawSnapshot,
    slot: usize,
) -> Option<f64> {
    let from_windows = pairs
        .iter()
        .filter_map(|(w, _)| w.gpus.get(slot).and_then(|g| g.temperature_peak_c))
        .filter(|t| t.is_finite())
        .reduce(|a, b| a.max(b));
    let landing = last
        .gpus
        .get(slot)
        .and_then(|g| g.temperature_c)
        .filter(|t| t.is_finite());
    max_option_f64(from_windows, landing)
}

/// Sum ΔHistogramWindowMass across windows - base for both `histogram_mean` and mass carry-forward.
fn accumulate_histogram_mass<M>(
    pairs: &[(&collectors::RawSnapshot, Duration)],
    get_mass: M,
) -> Option<HistogramWindowMass>
where
    M: Fn(&collectors::VllmRawMetrics) -> Option<HistogramWindowMass>,
{
    let mut sum = 0.0_f64;
    let mut count = 0.0_f64;
    for (w, _) in pairs {
        let Some(m) = get_mass(&w.vllm) else {
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

/// Multi-window histogram mean: ΣΔsum / ΣΔcount. `scale` converts units (1000 for s→ms, 1 for tokens).
fn histogram_mean<M>(
    pairs: &[(&collectors::RawSnapshot, Duration)],
    mass: M,
    scale: f64,
) -> Option<f64>
where
    M: Fn(&collectors::VllmRawMetrics) -> Option<HistogramWindowMass>,
{
    accumulate_histogram_mass(pairs, mass).and_then(|m| {
        let mean = (m.sum_delta / m.count_delta) * scale;
        mean.is_finite().then_some(mean)
    })
}

/// Duration-weighted mean of a scalar metric across windows. `None` if no valid windows.
fn weighted_mean<F>(pairs: &[(&collectors::RawSnapshot, Duration)], metric: F) -> Option<f64>
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

/// ΣΔhits / ΣΔqueries across evaluable windows - mathematically correct multi-window prefix hit rate.
fn prefix_hit_rate_sum_of_deltas(windows: &[&collectors::RawSnapshot]) -> Option<f64> {
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
        GpuRawMetrics, HistogramCount, HistogramWindowMass, PrefixCacheScrapeSample, RawSnapshot,
        VllmRawMetrics,
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
        // Activity is running >= 1 OR tok/s >= 1. KV/GPU util are not gates.
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
            gpus: vec![gpu],
        }
    }

    #[test]
    fn aggregate_windows_populates_p95_latencies() {
        fn latency_buckets() -> Vec<HistogramCount> {
            vec![
                HistogramCount {
                    less_than: 0.1,
                    count: 50.0,
                },
                HistogramCount {
                    less_than: 0.2,
                    count: 100.0,
                },
                HistogramCount {
                    less_than: f64::INFINITY,
                    count: 100.0,
                },
            ]
        }
        let g = GpuRawMetrics::default();
        let mut w1 = mk_snap(Some(5.0), Some(100.0), None, None, None, g.clone(), None);
        w1.vllm.ttft_p99_buckets = latency_buckets();
        w1.vllm.tpot_p99_buckets = latency_buckets();
        let mut w2 = mk_snap(Some(5.0), Some(200.0), None, None, None, g, None);
        w2.vllm.ttft_p99_buckets = latency_buckets();
        w2.vllm.tpot_p99_buckets = latency_buckets();
        let agg = aggregate_windows(
            &[w1, w2],
            &[Duration::from_secs(2), Duration::from_secs(2)],
            SystemTime::UNIX_EPOCH,
        );
        let ttft_p95 = agg.vllm.ttft_p95_ms.expect("ttft p95");
        let ttft_p99 = agg.vllm.ttft_p99_ms.expect("ttft p99");
        let tpot_p95 = agg.vllm.tpot_p95_ms.expect("tpot p95");
        let tpot_p99 = agg.vllm.tpot_p99_ms.expect("tpot p99");
        assert!(ttft_p95 <= ttft_p99);
        assert!(tpot_p95 <= tpot_p99);
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
        let mut w1 = mk_snap(
            Some(2.0),
            Some(100.0),
            Some((10.0, 20.0)),
            Some((50.0, 130.0)),
            None,
            g1,
            None,
        );
        w1.vllm.window_duration_secs = Some(2.0);
        let mut w2 = mk_snap(
            Some(10.0),
            Some(500.0),
            Some((0.0, 10.0)),
            Some((10.0, 20.0)),
            None,
            g2,
            None,
        );
        w2.vllm.window_duration_secs = Some(10.0);
        let agg = aggregate_windows(
            &[w1, w2],
            &[Duration::from_secs(2), Duration::from_secs(10)],
            SystemTime::UNIX_EPOCH,
        );
        // Both active. Duration-weighted: (2×2 + 10×10) / 12, (100×2 + 500×10) / 12,
        // util (10×2 + 50×10) / 12.
        assert!((agg.vllm.num_requests_running.unwrap() - (104.0 / 12.0)).abs() < 1e-9);
        assert!((agg.vllm.generation_tokens_per_sec.unwrap() - (5200.0 / 12.0)).abs() < 1e-4);
        // (10+10)/(80+10) = 20/90 - sum of Δhits / sum of Δqueries, not last window only.
        assert!((agg.vllm.prefix_cache_hit_rate.unwrap() - 20.0 / 90.0).abs() < 1e-9);
        assert!(
            (agg.gpus.first().and_then(|g| g.gpu_util_pct).unwrap() - (520.0 / 12.0)).abs() < 1e-4
        );
        assert_eq!(agg.gpus.first().and_then(|g| g.vram_used_mb), Some(2000));
        assert!((agg.gpus.first().and_then(|g| g.temperature_c).unwrap() - 60.0).abs() < 1e-9);
        assert_eq!(agg.gpus.first().and_then(|g| g.sm_clock_mhz), Some(2000));
    }

    #[test]
    fn actual_duration_used_over_planned_when_present() {
        let g = GpuRawMetrics::default();
        let mut w1 = mk_snap(Some(2.0), Some(100.0), None, None, None, g.clone(), None);
        w1.vllm.window_duration_secs = Some(2.0);
        let mut w2 = mk_snap(Some(2.0), Some(200.0), None, None, None, g, None);
        w2.vllm.window_duration_secs = Some(6.0);
        let planned = vec![Duration::from_secs(2), Duration::from_secs(2)];
        let agg = aggregate_windows(&[w1, w2], &planned, SystemTime::UNIX_EPOCH);
        // Planned-only weighting would yield (100×2 + 200×2) / 4 = 150.0.
        assert!((agg.vllm.generation_tokens_per_sec.unwrap() - 175.0).abs() < 1e-9);
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
        let mk = |v, g| RawSnapshot {
            gpu_observed_at: SystemTime::UNIX_EPOCH,
            vllm_observed_at: SystemTime::UNIX_EPOCH,
            timestamp: SystemTime::UNIX_EPOCH,
            vllm: v,
            gpus: vec![g],
        };
        let agg = aggregate_windows(
            &[mk(v1, g1), mk(v2, g2)],
            &[Duration::from_secs(2), Duration::from_secs(2)],
            SystemTime::UNIX_EPOCH,
        );
        assert!((agg.vllm.kv_cache_usage_perc.unwrap() - 25.0).abs() < 1e-9);
        assert!((agg.vllm.kv_cache_avg_perc.unwrap() - 25.0).abs() < 1e-9);
        assert!((agg.vllm.kv_cache_peak_perc.unwrap() - 92.0).abs() < 1e-9);
        assert_eq!(
            agg.gpus.first().and_then(|g| g.vram_used_mb),
            Some(55 * 1024)
        );
        assert_eq!(
            agg.gpus.first().and_then(|g| g.vram_peak_mb),
            Some(78 * 1024)
        );
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
        let mk = |v, g| RawSnapshot {
            gpu_observed_at: SystemTime::UNIX_EPOCH,
            vllm_observed_at: SystemTime::UNIX_EPOCH,
            timestamp: SystemTime::UNIX_EPOCH,
            vllm: v,
            gpus: vec![g],
        };
        let agg = aggregate_windows(
            &[mk(v1, g1), mk(v2, g2)],
            &[Duration::from_secs(2), Duration::from_secs(2)],
            SystemTime::UNIX_EPOCH,
        );
        assert!((agg.vllm.kv_cache_usage_perc.unwrap() - 67.5).abs() < 1e-9);
        assert!((agg.vllm.kv_cache_avg_perc.unwrap() - 67.5).abs() < 1e-9);
        assert!((agg.vllm.kv_cache_peak_perc.unwrap() - 95.0).abs() < 1e-9);
        assert_eq!(
            agg.gpus.first().and_then(|g| g.vram_used_mb),
            Some(72 * 1024)
        );
        assert_eq!(
            agg.gpus.first().and_then(|g| g.vram_peak_mb),
            Some(72 * 1024)
        );
    }

    #[test]
    fn aggregate_temperature_peak_folds_in_last_landing_when_higher_than_window_peaks() {
        let v = VllmRawMetrics {
            num_requests_running: Some(2.0),
            generation_tokens_per_sec: Some(100.0),
            window_duration_secs: Some(2.0),
            ..Default::default()
        };
        let mk = |v: VllmRawMetrics, g| RawSnapshot {
            gpu_observed_at: SystemTime::UNIX_EPOCH,
            vllm_observed_at: SystemTime::UNIX_EPOCH,
            timestamp: SystemTime::UNIX_EPOCH,
            vllm: v,
            gpus: vec![g],
        };
        let w1 = mk(
            v.clone(),
            GpuRawMetrics {
                temperature_c: Some(70.0),
                temperature_peak_c: Some(70.0),
                ..Default::default()
            },
        );
        let w2 = mk(
            v,
            GpuRawMetrics {
                temperature_c: Some(88.0),
                temperature_peak_c: Some(75.0),
                ..Default::default()
            },
        );
        let agg = aggregate_windows(
            &[w1, w2],
            &[Duration::from_secs(2), Duration::from_secs(2)],
            SystemTime::UNIX_EPOCH,
        );
        assert!((agg.gpus.first().and_then(|g| g.temperature_c).unwrap() - 88.0).abs() < 1e-9);
        assert!((agg.gpus.first().and_then(|g| g.temperature_peak_c).unwrap() - 88.0).abs() < 1e-9);
    }

    #[test]
    fn aggregate_histogram_ttft_is_sum_delta_over_sum_count_not_duration_weighted() {
        let g = GpuRawMetrics::default();
        let mk = |v: VllmRawMetrics| RawSnapshot {
            gpu_observed_at: SystemTime::UNIX_EPOCH,
            vllm_observed_at: SystemTime::UNIX_EPOCH,
            timestamp: SystemTime::UNIX_EPOCH,
            vllm: v,
            gpus: vec![g.clone()],
        };
        let w1 = mk(VllmRawMetrics {
            num_requests_running: Some(2.0),
            kv_cache_usage_perc: None,
            generation_tokens_per_sec: Some(100.0),
            window_duration_secs: Some(10.0),
            ttft_ms: Some(5000.0),
            ttft_window_mass: Some(HistogramWindowMass {
                sum_delta: 5.0,
                count_delta: 1.0,
            }),
            ..Default::default()
        });
        let w2 = mk(VllmRawMetrics {
            num_requests_running: Some(2.0),
            kv_cache_usage_perc: None,
            generation_tokens_per_sec: Some(100.0),
            window_duration_secs: Some(2.0),
            ttft_ms: Some(50.0),
            ttft_window_mass: Some(HistogramWindowMass {
                sum_delta: 25.0,
                count_delta: 500.0,
            }),
            ..Default::default()
        });
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

    #[test]
    fn aggregate_stores_mass_fields_on_aggregate() {
        let g = GpuRawMetrics::default();
        let mk = |v: VllmRawMetrics| RawSnapshot {
            gpu_observed_at: SystemTime::UNIX_EPOCH,
            vllm_observed_at: SystemTime::UNIX_EPOCH,
            timestamp: SystemTime::UNIX_EPOCH,
            vllm: v,
            gpus: vec![g.clone()],
        };
        let w1 = mk(VllmRawMetrics {
            num_requests_running: Some(2.0),
            kv_cache_usage_perc: None,
            window_duration_secs: Some(2.0),
            ttft_window_mass: Some(HistogramWindowMass {
                sum_delta: 2.0,
                count_delta: 4.0,
            }),
            tpot_window_mass: Some(HistogramWindowMass {
                sum_delta: 1.0,
                count_delta: 10.0,
            }),
            queue_window_mass: Some(HistogramWindowMass {
                sum_delta: 0.5,
                count_delta: 5.0,
            }),
            prompt_tokens_window_mass: Some(HistogramWindowMass {
                sum_delta: 100.0,
                count_delta: 2.0,
            }),
            ..Default::default()
        });
        let w2 = mk(VllmRawMetrics {
            num_requests_running: Some(2.0),
            kv_cache_usage_perc: None,
            window_duration_secs: Some(2.0),
            ttft_window_mass: Some(HistogramWindowMass {
                sum_delta: 3.0,
                count_delta: 6.0,
            }),
            tpot_window_mass: Some(HistogramWindowMass {
                sum_delta: 2.0,
                count_delta: 20.0,
            }),
            queue_window_mass: Some(HistogramWindowMass {
                sum_delta: 1.5,
                count_delta: 15.0,
            }),
            prompt_tokens_window_mass: Some(HistogramWindowMass {
                sum_delta: 200.0,
                count_delta: 4.0,
            }),
            ..Default::default()
        });
        let agg = aggregate_windows(
            &[w1, w2],
            &[Duration::from_secs(2), Duration::from_secs(2)],
            SystemTime::UNIX_EPOCH,
        );
        assert_eq!(
            agg.vllm.ttft_window_mass,
            Some(HistogramWindowMass {
                sum_delta: 5.0,
                count_delta: 10.0
            })
        );
        assert_eq!(
            agg.vllm.tpot_window_mass,
            Some(HistogramWindowMass {
                sum_delta: 3.0,
                count_delta: 30.0
            })
        );
        assert_eq!(
            agg.vllm.queue_window_mass,
            Some(HistogramWindowMass {
                sum_delta: 2.0,
                count_delta: 20.0
            })
        );
        assert_eq!(
            agg.vllm.prompt_tokens_window_mass,
            Some(HistogramWindowMass {
                sum_delta: 300.0,
                count_delta: 6.0
            })
        );
    }

    #[test]
    fn aggregate_means_use_active_windows_only() {
        let mk = |v: VllmRawMetrics, g: GpuRawMetrics| RawSnapshot {
            gpu_observed_at: SystemTime::UNIX_EPOCH,
            vllm_observed_at: SystemTime::UNIX_EPOCH,
            timestamp: SystemTime::UNIX_EPOCH,
            vllm: v,
            gpus: vec![g],
        };
        // True idle: evaluable, running < 1 and tok/s < 1. High KV proves KV is not a gate.
        let idle_v = VllmRawMetrics {
            num_requests_running: Some(0.0),
            kv_cache_usage_perc: Some(90.0),
            generation_tokens_per_sec: Some(0.0),
            window_duration_secs: Some(2.0),
            ..Default::default()
        };
        let active_v = VllmRawMetrics {
            num_requests_running: Some(20.0),
            generation_tokens_per_sec: Some(100.0),
            window_duration_secs: Some(2.0),
            ..Default::default()
        };
        let windows = vec![
            mk(
                idle_v,
                GpuRawMetrics {
                    gpu_util_pct: Some(5.0),
                    ..Default::default()
                },
            ),
            mk(
                active_v.clone(),
                GpuRawMetrics {
                    gpu_util_pct: Some(60.0),
                    ..Default::default()
                },
            ),
            mk(
                active_v,
                GpuRawMetrics {
                    gpu_util_pct: Some(60.0),
                    ..Default::default()
                },
            ),
        ];
        let durations = vec![
            Duration::from_secs(2),
            Duration::from_secs(2),
            Duration::from_secs(2),
        ];
        let agg = aggregate_windows(&windows, &durations, SystemTime::UNIX_EPOCH);
        assert!((agg.vllm.generation_tokens_per_sec.unwrap() - 100.0).abs() < 1e-9);
        assert!((agg.vllm.num_requests_running.unwrap() - 20.0).abs() < 1e-9);
        assert!((agg.gpus.first().and_then(|g| g.gpu_util_pct).unwrap() - 60.0).abs() < 1e-9);
        // If idle were included: (0+100+100)/3 = 66.7 tok/s.
        assert!((agg.vllm.generation_tokens_per_sec.unwrap() - (200.0 / 3.0)).abs() > 10.0);
    }

    #[test]
    fn aggregate_windows_merges_metrics_per_gpu_slot() {
        let v = VllmRawMetrics {
            num_requests_running: Some(10.0),
            generation_tokens_per_sec: Some(100.0),
            window_duration_secs: Some(2.0),
            ..Default::default()
        };
        let mk = |u0: f64, u1: f64, p0: f64, p1: f64| RawSnapshot {
            gpu_observed_at: SystemTime::UNIX_EPOCH,
            vllm_observed_at: SystemTime::UNIX_EPOCH,
            timestamp: SystemTime::UNIX_EPOCH,
            vllm: v.clone(),
            gpus: vec![
                GpuRawMetrics {
                    gpu_index: Some(0),
                    gpu_util_pct: Some(u0),
                    power_watts: Some(p0),
                    ..Default::default()
                },
                GpuRawMetrics {
                    gpu_index: Some(1),
                    gpu_util_pct: Some(u1),
                    power_watts: Some(p1),
                    ..Default::default()
                },
            ],
        };
        let agg = aggregate_windows(
            &[mk(40.0, 60.0, 100.0, 200.0), mk(80.0, 20.0, 300.0, 100.0)],
            &[Duration::from_secs(2), Duration::from_secs(2)],
            SystemTime::UNIX_EPOCH,
        );
        assert_eq!(agg.gpus.len(), 2);
        assert!((agg.gpus[0].gpu_util_pct.unwrap() - 60.0).abs() < 1e-9);
        assert!((agg.gpus[1].gpu_util_pct.unwrap() - 40.0).abs() < 1e-9);
        assert!((agg.gpus[0].power_watts.unwrap() - 200.0).abs() < 1e-9);
        assert!((agg.gpus[1].power_watts.unwrap() - 150.0).abs() < 1e-9);
        assert!((agg.gpus[0].aligned_power_watts.unwrap() - 200.0).abs() < 1e-9);
        assert!((agg.gpus[1].aligned_power_watts.unwrap() - 150.0).abs() < 1e-9);
    }

    #[test]
    fn aggregate_aligned_power_excludes_skewed_active_windows() {
        let v = VllmRawMetrics {
            num_requests_running: Some(10.0),
            generation_tokens_per_sec: Some(100.0),
            window_duration_secs: Some(2.0),
            ..Default::default()
        };
        let aligned = RawSnapshot {
            gpu_observed_at: SystemTime::UNIX_EPOCH,
            vllm_observed_at: SystemTime::UNIX_EPOCH,
            timestamp: SystemTime::UNIX_EPOCH,
            vllm: v.clone(),
            gpus: vec![GpuRawMetrics {
                gpu_index: Some(0),
                power_watts: Some(100.0),
                ..Default::default()
            }],
        };
        let skewed = RawSnapshot {
            gpu_observed_at: SystemTime::UNIX_EPOCH,
            vllm_observed_at: SystemTime::UNIX_EPOCH + Duration::from_secs(5),
            timestamp: SystemTime::UNIX_EPOCH,
            vllm: v,
            gpus: vec![GpuRawMetrics {
                gpu_index: Some(0),
                power_watts: Some(300.0),
                ..Default::default()
            }],
        };
        let agg = aggregate_windows(
            &[aligned, skewed],
            &[Duration::from_secs(2), Duration::from_secs(2)],
            SystemTime::UNIX_EPOCH,
        );
        // Display power includes both active windows.
        assert!((agg.gpus[0].power_watts.unwrap() - 200.0).abs() < 1e-9);
        // Aligned energy power keeps only the in-skew window.
        assert!((agg.gpus[0].aligned_power_watts.unwrap() - 100.0).abs() < 1e-9);
    }

    #[test]
    fn aggregate_aligned_power_none_when_all_active_windows_skewed() {
        let v = VllmRawMetrics {
            num_requests_running: Some(10.0),
            generation_tokens_per_sec: Some(100.0),
            window_duration_secs: Some(2.0),
            ..Default::default()
        };
        let skewed = |power: f64| RawSnapshot {
            gpu_observed_at: SystemTime::UNIX_EPOCH,
            vllm_observed_at: SystemTime::UNIX_EPOCH + Duration::from_secs(5),
            timestamp: SystemTime::UNIX_EPOCH,
            vllm: v.clone(),
            gpus: vec![GpuRawMetrics {
                gpu_index: Some(0),
                power_watts: Some(power),
                ..Default::default()
            }],
        };
        let agg = aggregate_windows(
            &[skewed(100.0), skewed(300.0)],
            &[Duration::from_secs(2), Duration::from_secs(2)],
            SystemTime::UNIX_EPOCH,
        );
        assert!(agg.gpus[0].power_watts.is_some());
        assert!(agg.gpus[0].aligned_power_watts.is_none());
    }

    #[test]
    fn weighted_mean_returns_none_when_all_windows_missing_metric() {
        let snap = RawSnapshot::default();
        let pairs = vec![(&snap, Duration::from_secs(2))];
        assert!(weighted_mean(&pairs, |w| w.vllm.generation_tokens_per_sec).is_none());
    }

    #[test]
    fn accumulate_histogram_mass_ignores_negative_sum_delta() {
        let snap = RawSnapshot {
            vllm: VllmRawMetrics {
                ttft_window_mass: Some(HistogramWindowMass {
                    sum_delta: -1.0,
                    count_delta: 5.0,
                }),
                ..Default::default()
            },
            ..Default::default()
        };
        let pairs = vec![(&snap, Duration::from_secs(2))];
        assert!(accumulate_histogram_mass(&pairs, |v| v.ttft_window_mass).is_none());
    }
}
