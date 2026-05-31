use std::thread;
use std::time::{Duration, Instant, SystemTime};

use anyhow::{Context, Result};
use prometheus_parse::{Scrape, Value};

use super::sampling::{sample_count_for, SAMPLE_INTERVAL};
use super::types::{
    CacheConfigLabels, HistogramWindowMass, PrefixCacheScrapeSample, VllmRawMetrics,
};
const REQ_TIMEOUT: Duration = Duration::from_secs(10);

/// vLLM exposes metrics with a `vllm:` prefix. The `prometheus-parse` crate only accepts
/// `[a-zA-Z0-9_]` in metric names (`\w+`), so lines containing `:` are skipped unless we
/// normalize them to underscores before parsing.
fn normalize_vllm_prometheus_text(body: &str) -> String {
    body.replace("vllm:", "vllm_")
}

/// Resolves the GET URL for Prometheus text. Accepts a server base URL or a URL that already ends with `/metrics`.
fn metrics_url(input: &str) -> String {
    let t = input.trim().trim_end_matches('/');
    if t.ends_with("/metrics") {
        t.to_string()
    } else {
        format!("{}/metrics", t)
    }
}

fn fetch_metrics_body(client: &reqwest::blocking::Client, url: &str) -> Result<String> {
    client
        .get(url)
        .send()
        .with_context(|| format!("failed to GET {}", url))?
        .error_for_status()
        .with_context(|| format!("non-success response from {}", url))?
        .text()
        .with_context(|| format!("failed to read response body from {}", url))
}

fn scrape_from_body(body: &str) -> Result<Scrape> {
    let normalized = normalize_vllm_prometheus_text(body);
    Scrape::parse(normalized.lines().map(|s| Ok(s.to_string())))
        .context("failed to parse Prometheus text format")
}

/// KV cache fill ratio 0–100, same as last scrape in `parse_vllm_metrics`.
fn kv_cache_usage_perc_from_scrape(scrape: &Scrape) -> Option<f64> {
    first_gauge(scrape, "vllm_kv_cache_usage_perc")
        .or_else(|| first_gauge(scrape, "vllm_gpu_cache_usage_perc"))
        .map(|v| v * 100.0)
}

fn first_gauge(scrape: &Scrape, name: &str) -> Option<f64> {
    scrape
        .samples
        .iter()
        .find(|s| s.metric == name)
        .and_then(|s| match s.value {
            Value::Gauge(v) | Value::Untyped(v) => Some(v),
            _ => None,
        })
}

fn sum_metric_samples(scrape: &Scrape, name: &str) -> Option<f64> {
    let mut total = 0.0;
    let mut any = false;
    for s in &scrape.samples {
        if s.metric != name {
            continue;
        }
        if let Value::Gauge(v) | Value::Counter(v) | Value::Untyped(v) = s.value {
            total += v;
            any = true;
        }
    }
    any.then_some(total)
}

fn total_generation_tokens(scrape: &Scrape) -> Option<f64> {
    sum_metric_samples(scrape, "vllm_generation_tokens_total")
        .or_else(|| sum_metric_samples(scrape, "vllm_iteration_tokens_total_sum"))
}

fn total_preemptions(scrape: &Scrape) -> Option<f64> {
    sum_metric_samples(scrape, "vllm_num_preemptions_total")
        .or_else(|| sum_metric_samples(scrape, "vllm_num_preemptions"))
}

fn total_request_success(scrape: &Scrape) -> Option<f64> {
    sum_metric_samples(scrape, "vllm_request_success_total")
        .or_else(|| sum_metric_samples(scrape, "vllm_request_success"))
}

/// `(last - first) / window_secs` when monotonic; `None` on reset, missing endpoints, or **zero window** (no divide-by-zero).
fn counter_delta_per_sec(first: Option<f64>, last: Option<f64>, window_secs: f64) -> Option<f64> {
    if window_secs <= f64::EPSILON {
        return None;
    }
    let a = first?;
    let b = last?;
    let d = b - a;
    if d < 0.0 {
        return None;
    }
    let out = d / window_secs;
    out.is_finite().then_some(out)
}

fn sum_two_metric_series(scrape: &Scrape, a: &str, b: &str) -> Option<f64> {
    match (sum_metric_samples(scrape, a), sum_metric_samples(scrape, b)) {
        (None, None) => None,
        (Some(x), None) => Some(x),
        (None, Some(y)) => Some(y),
        (Some(x), Some(y)) => Some(x + y),
    }
}

/// Prefer `*_total` counter names (current vLLM); fall back to legacy names without `_total`.
fn sum_two_metric_series_prefix(
    scrape: &Scrape,
    a_total: &str,
    b_total: &str,
    a_legacy: &str,
    b_legacy: &str,
) -> Option<f64> {
    sum_two_metric_series(scrape, a_total, b_total)
        .or_else(|| sum_two_metric_series(scrape, a_legacy, b_legacy))
}

/// `(hits, queries)` summed over internal + external prefix cache counters.
fn prefix_counter_totals(scrape: &Scrape) -> (Option<f64>, Option<f64>) {
    let hits = sum_two_metric_series_prefix(
        scrape,
        "vllm_prefix_cache_hits_total",
        "vllm_external_prefix_cache_hits_total",
        "vllm_prefix_cache_hits",
        "vllm_external_prefix_cache_hits",
    );
    let queries = sum_two_metric_series_prefix(
        scrape,
        "vllm_prefix_cache_queries_total",
        "vllm_external_prefix_cache_queries_total",
        "vllm_prefix_cache_queries",
        "vllm_external_prefix_cache_queries",
    );
    (hits, queries)
}

fn prefix_scrape_sample(scrape: &Scrape) -> PrefixCacheScrapeSample {
    let (hits, queries) = prefix_counter_totals(scrape);
    PrefixCacheScrapeSample { hits, queries }
}

/// `(hits_last - hits_first) / (queries_last - queries_first)` over the first→last scrape window.
///
/// Returns `None` when:
/// - either scrape lacks hits/queries totals,
/// - **`Δqueries <= 0`** (zero-query window, flat counters, or non-monotonic queries) — **never divide by zero**,
/// - `Δhits < 0` (counter reset),
/// - non-finite values.
///
/// `None` means prefix hit rate cannot be computed for this window (e.g. **Δqueries ≤ 0**).
fn prefix_window_hit_rate(first: &Scrape, last: &Scrape) -> Option<f64> {
    let (h0, q0) = prefix_counter_totals(first);
    let (h1, q1) = prefix_counter_totals(last);
    let h0 = h0?;
    let h1 = h1?;
    let q0 = q0?;
    let q1 = q1?;
    if !(h0.is_finite() && h1.is_finite() && q0.is_finite() && q1.is_finite()) {
        return None;
    }
    let dq = q1 - q0;
    if dq <= 0.0 {
        return None;
    }
    let dh = h1 - h0;
    if dh < 0.0 || !dh.is_finite() {
        return None;
    }
    let rate = dh / dq;
    rate.is_finite().then_some(rate)
}

fn compute_counter_rates(
    first: &Scrape,
    last: &Scrape,
    window_secs: f64,
) -> (Option<f64>, Option<f64>) {
    let gen_per_sec = counter_delta_per_sec(
        total_generation_tokens(first),
        total_generation_tokens(last),
        window_secs,
    );
    let prefix = prefix_window_hit_rate(first, last);
    (gen_per_sec, prefix)
}

/// Cumulative mean from a single scrape (`sum`/`count` across labeled series). **`count == 0` → `None`** (no divide-by-zero).
fn histogram_mean_from_scrape(scrape: &Scrape, base: &str) -> Option<f64> {
    let sum = sum_metric_samples(scrape, &format!("{base}_sum"));
    let count = sum_metric_samples(scrape, &format!("{base}_count"));
    match (sum, count) {
        (Some(s), Some(c)) if c > 0.0 => {
            let m = s / c;
            m.is_finite().then_some(m)
        }
        _ => None,
    }
}

fn histogram_mean_ms_from_scrape(scrape: &Scrape, base: &str) -> Option<f64> {
    histogram_mean_from_scrape(scrape, base).map(|m| m * 1000.0)
}

/// Aggregated `(Δsum)/(Δcount)` across all series for `base` (histogram `_sum` / `_count`).
/// Units match the histogram (seconds vs tokens). `None` if **`Δcount <= 0`** (no new observations), reset, or non-finite.
fn histogram_window_mass(first: &Scrape, last: &Scrape, base: &str) -> Option<HistogramWindowMass> {
    let sum_key = format!("{base}_sum");
    let count_key = format!("{base}_count");
    let s0 = sum_metric_samples(first, &sum_key)?;
    let c0 = sum_metric_samples(first, &count_key)?;
    let s1 = sum_metric_samples(last, &sum_key)?;
    let c1 = sum_metric_samples(last, &count_key)?;
    let ds = s1 - s0;
    let dc = c1 - c0;
    if dc <= 0.0 || ds < 0.0 || !(ds.is_finite() && dc.is_finite()) {
        return None;
    }
    Some(HistogramWindowMass {
        sum_delta: ds,
        count_delta: dc,
    })
}

fn histogram_window_mean(first: &Scrape, last: &Scrape, base: &str) -> Option<f64> {
    let m = histogram_window_mass(first, last, base)?;
    let x = m.sum_delta / m.count_delta;
    x.is_finite().then_some(x)
}

fn tpot_window_mass(first: &Scrape, last: &Scrape) -> Option<HistogramWindowMass> {
    histogram_window_mass(first, last, "vllm_request_time_per_output_token_seconds")
        .or_else(|| histogram_window_mass(first, last, "vllm_time_per_output_token_seconds"))
}

fn histogram_window_mean_ms(first: &Scrape, last: &Scrape, base: &str) -> Option<f64> {
    histogram_window_mean(first, last, base).map(|sec| sec * 1000.0)
}

fn tpot_window_ms(first: &Scrape, last: &Scrape) -> Option<f64> {
    tpot_window_mass(first, last).and_then(|m| {
        let sec = m.sum_delta / m.count_delta;
        sec.is_finite().then_some(sec * 1000.0)
    })
}

/// `first` = scrape from sample 1, `last` = scrape from sample [`SAMPLE_COUNT`] (~2s later).
fn apply_histogram_window(first: &Scrape, last: &Scrape, m: &mut VllmRawMetrics) {
    m.ttft_window_mass = histogram_window_mass(first, last, "vllm_time_to_first_token_seconds");
    m.tpot_window_mass = tpot_window_mass(first, last);
    m.prefill_window_mass = histogram_window_mass(first, last, "vllm_request_prefill_time_seconds");
    m.queue_window_mass = histogram_window_mass(first, last, "vllm_request_queue_time_seconds");
    m.prompt_tokens_window_mass = histogram_window_mass(first, last, "vllm_request_prompt_tokens");

    // Prefer Δsum/Δcount over that window; if no new observations, use last-scrape mean.
    m.ttft_ms = histogram_window_mean_ms(first, last, "vllm_time_to_first_token_seconds")
        .or_else(|| histogram_mean_ms_from_scrape(last, "vllm_time_to_first_token_seconds"));
    m.tpot_ms = tpot_window_ms(first, last).or_else(|| {
        histogram_mean_ms_from_scrape(last, "vllm_request_time_per_output_token_seconds")
            .or_else(|| histogram_mean_ms_from_scrape(last, "vllm_time_per_output_token_seconds"))
    });
    m.prefill_latency_ms =
        histogram_window_mean_ms(first, last, "vllm_request_prefill_time_seconds")
            .or_else(|| histogram_mean_ms_from_scrape(last, "vllm_request_prefill_time_seconds"));
    m.queue_delay_ms = histogram_window_mean_ms(first, last, "vllm_request_queue_time_seconds")
        .or_else(|| histogram_mean_ms_from_scrape(last, "vllm_request_queue_time_seconds"));
    m.prompt_tokens_mean = histogram_window_mean(first, last, "vllm_request_prompt_tokens")
        .or_else(|| histogram_mean_from_scrape(last, "vllm_request_prompt_tokens"));
}

fn max_num_seqs_from_gauge(scrape: &Scrape) -> Option<u32> {
    first_gauge(scrape, "vllm_max_num_seqs").and_then(|v| {
        if v.is_finite() && v >= 0.0 {
            let r = v.round();
            if r <= u32::MAX as f64 {
                Some(r as u32)
            } else {
                Some(u32::MAX)
            }
        } else {
            None
        }
    })
}

pub fn collect_vllm_metrics_for(
    input: &str,
    window: Duration,
) -> Result<(VllmRawMetrics, SystemTime)> {
    let client = reqwest::blocking::Client::builder()
        .timeout(REQ_TIMEOUT)
        .build()
        .context("failed to build HTTP client")?;
    let url = metrics_url(input);
    let sample_count = sample_count_for(window);
    let mut prefix_samples = Vec::with_capacity(sample_count);
    let mut window_start: Option<Instant> = None;
    let mut first_scrape: Option<Scrape> = None;
    let mut last_scrape: Option<Scrape> = None;
    let mut kv_cache_peak_perc: Option<f64> = None;

    for i in 0..sample_count {
        let body = fetch_metrics_body(&client, &url)?;
        let scrape = scrape_from_body(&body)?;
        if let Some(k) = kv_cache_usage_perc_from_scrape(&scrape).filter(|x| x.is_finite()) {
            kv_cache_peak_perc = Some(kv_cache_peak_perc.map_or(k, |p| p.max(k)));
        }
        prefix_samples.push(prefix_scrape_sample(&scrape));

        if i == 0 {
            window_start = Some(Instant::now());
        }
        if i == 0 {
            first_scrape = Some(scrape);
        } else {
            last_scrape = Some(scrape);
        }

        if i + 1 < sample_count {
            thread::sleep(SAMPLE_INTERVAL);
        }
    }

    let window_secs = window_start
        .map(|t| t.elapsed().as_secs_f64())
        .unwrap_or(0.0);

    let first_scrape = first_scrape.context("vLLM gauge window missing first scrape")?;
    let last_scrape = last_scrape.context("vLLM gauge window missing last scrape")?;
    let mut m = parse_vllm_metrics(&last_scrape)?;
    m.kv_cache_peak_perc = kv_cache_peak_perc;

    let (gen_per_sec, prefix_hit) = compute_counter_rates(&first_scrape, &last_scrape, window_secs);
    m.generation_tokens_per_sec = gen_per_sec;
    m.prefix_cache_hit_rate = prefix_hit;
    m.prefix_cache_scrape_samples = prefix_samples;
    m.request_success_per_sec = counter_delta_per_sec(
        total_request_success(&first_scrape),
        total_request_success(&last_scrape),
        window_secs,
    );
    m.num_preemptions_per_sec = counter_delta_per_sec(
        total_preemptions(&first_scrape),
        total_preemptions(&last_scrape),
        window_secs,
    );

    apply_histogram_window(&first_scrape, &last_scrape, &mut m);

    if window_secs.is_finite() && window_secs > f64::EPSILON {
        m.window_duration_secs = Some(window_secs);
    }

    Ok((m, SystemTime::now()))
}

fn parse_cache_config_labels(scrape: &Scrape) -> CacheConfigLabels {
    let Some(s) = scrape
        .samples
        .iter()
        .find(|s| s.metric == "vllm_cache_config_info")
    else {
        return CacheConfigLabels::default();
    };
    CacheConfigLabels {
        block_size: s.labels.get("block_size").and_then(|v| v.parse().ok()),
        num_gpu_blocks: s.labels.get("num_gpu_blocks").and_then(|v| v.parse().ok()),
        cache_dtype: s.labels.get("cache_dtype").map(|v| v.to_string()),
        enable_prefix_caching: s
            .labels
            .get("enable_prefix_caching")
            .and_then(super::config::parse_bool),
        enable_chunked_prefill: s
            .labels
            .get("enable_chunked_prefill")
            .and_then(super::config::parse_bool),
    }
}

fn parse_vllm_metrics(scrape: &Scrape) -> Result<VllmRawMetrics> {
    let model_name = scrape
        .samples
        .iter()
        .find(|s| s.metric.starts_with("vllm_"))
        .and_then(|s| s.labels.get("model_name").map(str::to_string));

    let num_requests_running = first_gauge(scrape, "vllm_num_requests_running");
    let num_requests_waiting = first_gauge(scrape, "vllm_num_requests_waiting");
    let kv_cache_usage_perc = kv_cache_usage_perc_from_scrape(scrape);

    let ttft_ms = histogram_mean_ms_from_scrape(scrape, "vllm_time_to_first_token_seconds");
    let tpot_ms =
        histogram_mean_ms_from_scrape(scrape, "vllm_request_time_per_output_token_seconds")
            .or_else(|| {
                histogram_mean_ms_from_scrape(scrape, "vllm_time_per_output_token_seconds")
            });
    let prefill_latency_ms =
        histogram_mean_ms_from_scrape(scrape, "vllm_request_prefill_time_seconds");
    let queue_delay_ms = histogram_mean_ms_from_scrape(scrape, "vllm_request_queue_time_seconds");
    let prompt_tokens_mean = histogram_mean_from_scrape(scrape, "vllm_request_prompt_tokens");

    let generation_tokens_total = total_generation_tokens(scrape);

    let max_num_seqs = max_num_seqs_from_gauge(scrape);
    let num_requests_swapped = first_gauge(scrape, "vllm_num_requests_swapped");
    let num_preemptions_total = total_preemptions(scrape);
    let cpu_cache_usage_perc = first_gauge(scrape, "vllm_cpu_cache_usage_perc").map(|v| v * 100.0);
    let request_success_total = total_request_success(scrape);

    let cache_config = parse_cache_config_labels(scrape);

    Ok(VllmRawMetrics {
        model_name,
        num_requests_running,
        num_requests_waiting,
        kv_cache_usage_perc,
        kv_cache_avg_perc: None,
        kv_cache_peak_perc: None,
        ttft_ms,
        tpot_ms,
        prefill_latency_ms,
        queue_delay_ms,
        prompt_tokens_mean,
        window_duration_secs: None,
        ttft_window_mass: None,
        tpot_window_mass: None,
        prefill_window_mass: None,
        queue_window_mass: None,
        prompt_tokens_window_mass: None,
        generation_tokens_total,
        generation_tokens_per_sec: None,
        prefix_cache_hit_rate: None,
        prefix_cache_scrape_samples: vec![],
        max_num_seqs,
        num_requests_swapped,
        num_preemptions_total,
        num_preemptions_per_sec: None,
        cpu_cache_usage_perc,
        request_success_total,
        request_success_per_sec: None,
        cache_config,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collectors::types::VllmRawMetrics;

    #[test]
    fn colon_prefixed_vllm_metrics_parse_after_normalize() {
        let body = r#"
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_sum{model_name="llama3"} 0.5
vllm:time_to_first_token_seconds_count{model_name="llama3"} 4
vllm:generation_tokens_total{model_name="llama3"} 99
"#;
        let scrape = scrape_from_body(body).unwrap();
        let m = parse_vllm_metrics(&scrape).unwrap();
        assert!((m.ttft_ms.unwrap() - 125.0).abs() < 1e-6);
        assert_eq!(m.generation_tokens_total, Some(99.0));
        assert_eq!(m.model_name.as_deref(), Some("llama3"));
        assert!(
            m.generation_tokens_per_sec.is_none() && m.prefix_cache_hit_rate.is_none(),
            "parse-only must not set windowed counters"
        );
        assert!(m.prompt_tokens_mean.is_none());
    }

    #[test]
    fn state_gauges_use_last_scrape_in_window() {
        let a = r#"
vllm_num_requests_running 2
vllm_num_requests_waiting 1
vllm_max_num_seqs 10
"#;
        let b = r#"
vllm_num_requests_running 4
vllm_num_requests_waiting 0
vllm_max_num_seqs 256
"#;
        let last = scrape_from_body(b).unwrap();
        let m = parse_vllm_metrics(&last).unwrap();
        assert!((m.num_requests_running.unwrap() - 4.0).abs() < 1e-9);
        assert!((m.num_requests_waiting.unwrap()).abs() < 1e-9);
        assert_eq!(m.max_num_seqs, Some(256));
        let early = scrape_from_body(a).unwrap();
        let m_early = parse_vllm_metrics(&early).unwrap();
        assert!((m_early.num_requests_running.unwrap() - 2.0).abs() < 1e-9);
        assert!((m_early.num_requests_waiting.unwrap() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn max_num_seqs_from_gauge_rounds() {
        let body = "vllm_max_num_seqs 15.7\n";
        let s = scrape_from_body(body).unwrap();
        assert_eq!(max_num_seqs_from_gauge(&s), Some(16));
    }

    #[test]
    fn max_num_seqs_absent_is_none() {
        let body = "vllm_num_requests_running 1\n";
        let s = scrape_from_body(body).unwrap();
        assert_eq!(max_num_seqs_from_gauge(&s), None);
    }

    #[test]
    fn counter_delta_per_sec_monotonic() {
        assert_eq!(
            counter_delta_per_sec(Some(100.0), Some(250.0), 1.5),
            Some(100.0)
        );
        assert_eq!(counter_delta_per_sec(Some(10.0), Some(5.0), 1.0), None);
        assert_eq!(counter_delta_per_sec(Some(1.0), Some(2.0), 0.0), None);
    }

    #[test]
    fn counter_delta_per_sec_zero_delta_is_valid() {
        assert_eq!(
            counter_delta_per_sec(Some(50.0), Some(50.0), 2.0),
            Some(0.0)
        );
    }

    #[test]
    fn counter_delta_per_sec_missing_endpoint() {
        assert_eq!(counter_delta_per_sec(None, Some(10.0), 1.0), None);
        assert_eq!(counter_delta_per_sec(Some(1.0), None, 1.0), None);
    }

    #[test]
    fn sum_metric_samples_sums_labeled_series() {
        let body = r#"
vllm_generation_tokens_total{model_name="a"} 40
vllm_generation_tokens_total{model_name="b"} 60
"#;
        let s = scrape_from_body(body).unwrap();
        assert_eq!(
            sum_metric_samples(&s, "vllm_generation_tokens_total"),
            Some(100.0)
        );
    }

    #[test]
    fn total_generation_tokens_prefers_generation_over_iteration_sum() {
        let body = r#"
vllm_generation_tokens_total 10
vllm_iteration_tokens_total_sum 999
"#;
        let s = scrape_from_body(body).unwrap();
        assert_eq!(total_generation_tokens(&s), Some(10.0));
    }

    #[test]
    fn total_generation_tokens_falls_back_to_iteration_sum() {
        let body = "vllm_iteration_tokens_total_sum 42\n";
        let s = scrape_from_body(body).unwrap();
        assert_eq!(total_generation_tokens(&s), Some(42.0));
    }

    #[test]
    fn compute_counter_rates_generation_throughput() {
        let a = "vllm_generation_tokens_total 100\n";
        let b = "vllm_generation_tokens_total 250\n";
        let (tps, prefix) = compute_counter_rates(
            &scrape_from_body(a).unwrap(),
            &scrape_from_body(b).unwrap(),
            1.5,
        );
        assert!((tps.unwrap() - 100.0).abs() < 1e-9);
        assert_eq!(prefix, None);
    }

    #[test]
    fn compute_counter_rates_zero_gen_delta() {
        let a = "vllm_generation_tokens_total 50\n";
        let (tps, _) = compute_counter_rates(
            &scrape_from_body(a).unwrap(),
            &scrape_from_body(a).unwrap(),
            2.0,
        );
        assert_eq!(tps, Some(0.0));
    }

    #[test]
    fn compute_counter_rates_missing_gen_on_first_scrape() {
        let first = "vllm_num_requests_running 1\n";
        let last = "vllm_generation_tokens_total 10\n";
        let (tps, _) = compute_counter_rates(
            &scrape_from_body(first).unwrap(),
            &scrape_from_body(last).unwrap(),
            1.0,
        );
        assert!(tps.is_none());
    }

    #[test]
    fn compute_counter_rates_prefix_reuse_and_queries_flat() {
        let a = r#"
vllm_prefix_cache_hits 1
vllm_prefix_cache_queries 10
"#;
        let b = r#"
vllm_prefix_cache_hits 3
vllm_prefix_cache_queries 10
"#;
        let (_, hit_rate) = compute_counter_rates(
            &scrape_from_body(a).unwrap(),
            &scrape_from_body(b).unwrap(),
            1.0,
        );
        assert_eq!(hit_rate, None);
    }

    #[test]
    fn compute_counter_rates_prefix_reads_total_suffix_counters() {
        let first = r#"
vllm_prefix_cache_hits_total{model_name="llama3"} 400
vllm_prefix_cache_queries_total{model_name="llama3"} 550
"#;
        let last = r#"
vllm_prefix_cache_hits_total{model_name="llama3"} 448
vllm_prefix_cache_queries_total{model_name="llama3"} 615
"#;
        let (_, hit_rate) = compute_counter_rates(
            &scrape_from_body(first).unwrap(),
            &scrape_from_body(last).unwrap(),
            1.0,
        );
        let dh = 448.0 - 400.0;
        let dq = 615.0 - 550.0;
        assert!((hit_rate.unwrap() - dh / dq).abs() < 1e-9);
    }

    #[test]
    fn compute_counter_rates_prefix_negative_delta_hits_returns_none() {
        let a = r#"
vllm_prefix_cache_hits 10
vllm_prefix_cache_queries 20
"#;
        let b = r#"
vllm_prefix_cache_hits 2
vllm_prefix_cache_queries 30
"#;
        let (_, hit_rate) = compute_counter_rates(
            &scrape_from_body(a).unwrap(),
            &scrape_from_body(b).unwrap(),
            1.0,
        );
        assert_eq!(hit_rate, None);
    }

    #[test]
    fn compute_counter_rates_prefix_partial_first_series() {
        let a = "vllm_prefix_cache_hits 1\n";
        let b = r#"
vllm_prefix_cache_hits 5
vllm_prefix_cache_queries 100
"#;
        let (_, hit_rate) = compute_counter_rates(
            &scrape_from_body(a).unwrap(),
            &scrape_from_body(b).unwrap(),
            1.0,
        );
        assert_eq!(hit_rate, None);
    }

    #[test]
    fn compute_counter_rates_prefix_hits_only_returns_none() {
        let a = "vllm_prefix_cache_hits 1\n";
        let b = "vllm_prefix_cache_hits 2\n";
        let (_, hit_rate) = compute_counter_rates(
            &scrape_from_body(a).unwrap(),
            &scrape_from_body(b).unwrap(),
            1.0,
        );
        assert_eq!(hit_rate, None);
    }

    #[test]
    fn compute_counter_rates_prefix_happy_path() {
        let a = r#"
vllm_prefix_cache_hits 2
vllm_prefix_cache_queries 10
"#;
        let b = r#"
vllm_prefix_cache_hits 5
vllm_prefix_cache_queries 20
"#;
        let (_, hit_rate) = compute_counter_rates(
            &scrape_from_body(a).unwrap(),
            &scrape_from_body(b).unwrap(),
            1.0,
        );
        assert!((hit_rate.unwrap() - 0.3).abs() < 1e-9);
    }

    #[test]
    fn compute_counter_rates_iteration_fallback_both_scrapes() {
        let a = "vllm_iteration_tokens_total_sum 1000\n";
        let b = "vllm_iteration_tokens_total_sum 1060\n";
        let (tps, _) = compute_counter_rates(
            &scrape_from_body(a).unwrap(),
            &scrape_from_body(b).unwrap(),
            2.0,
        );
        assert!((tps.unwrap() - 30.0).abs() < 1e-9);
    }

    #[test]
    fn compute_counter_rates_zero_window_yields_no_rates() {
        let a = "vllm_generation_tokens_total 1\n";
        let b = "vllm_generation_tokens_total 9\n";
        let (tps, _) = compute_counter_rates(
            &scrape_from_body(a).unwrap(),
            &scrape_from_body(b).unwrap(),
            0.0,
        );
        assert!(tps.is_none());
    }

    #[test]
    fn histogram_window_mean_ms_ttft() {
        let a =
            "vllm_time_to_first_token_seconds_sum 1\nvllm_time_to_first_token_seconds_count 2\n";
        let b =
            "vllm_time_to_first_token_seconds_sum 5\nvllm_time_to_first_token_seconds_count 10\n";
        let fa = scrape_from_body(a).unwrap();
        let fb = scrape_from_body(b).unwrap();
        let ms = histogram_window_mean_ms(&fa, &fb, "vllm_time_to_first_token_seconds").unwrap();
        assert!((ms - 500.0).abs() < 1e-6);
    }

    #[test]
    fn histogram_window_mean_prompt_tokens_not_seconds() {
        let a = "vllm_request_prompt_tokens_sum 10\nvllm_request_prompt_tokens_count 2\n";
        let b = "vllm_request_prompt_tokens_sum 40\nvllm_request_prompt_tokens_count 5\n";
        let m = histogram_window_mean(
            &scrape_from_body(a).unwrap(),
            &scrape_from_body(b).unwrap(),
            "vllm_request_prompt_tokens",
        )
        .unwrap();
        assert!((m - 10.0).abs() < 1e-9);
    }

    #[test]
    fn histogram_window_mean_dc_zero_returns_none() {
        let a =
            "vllm_time_to_first_token_seconds_sum 1\nvllm_time_to_first_token_seconds_count 4\n";
        let b =
            "vllm_time_to_first_token_seconds_sum 2\nvllm_time_to_first_token_seconds_count 4\n";
        assert!(histogram_window_mean_ms(
            &scrape_from_body(a).unwrap(),
            &scrape_from_body(b).unwrap(),
            "vllm_time_to_first_token_seconds",
        )
        .is_none());
    }

    #[test]
    fn histogram_window_mean_negative_delta_sum_returns_none() {
        let a =
            "vllm_time_to_first_token_seconds_sum 10\nvllm_time_to_first_token_seconds_count 4\n";
        let b =
            "vllm_time_to_first_token_seconds_sum 5\nvllm_time_to_first_token_seconds_count 8\n";
        assert!(histogram_window_mean_ms(
            &scrape_from_body(a).unwrap(),
            &scrape_from_body(b).unwrap(),
            "vllm_time_to_first_token_seconds",
        )
        .is_none());
    }

    #[test]
    fn apply_histogram_window_ttft_from_deltas() {
        let fa = scrape_from_body(
            "vllm_time_to_first_token_seconds_sum 1\nvllm_time_to_first_token_seconds_count 1\n",
        )
        .unwrap();
        let fb = scrape_from_body(
            "vllm_time_to_first_token_seconds_sum 3\nvllm_time_to_first_token_seconds_count 3\n",
        )
        .unwrap();
        let mut m = VllmRawMetrics::default();
        apply_histogram_window(&fa, &fb, &mut m);
        assert!((m.ttft_ms.unwrap() - 1000.0).abs() < 1e-6);
    }

    #[test]
    fn tpot_window_ms_fallback_metric_name() {
        let fa = scrape_from_body(
            "vllm_time_per_output_token_seconds_sum 1\nvllm_time_per_output_token_seconds_count 2\n",
        )
        .unwrap();
        let fb = scrape_from_body(
            "vllm_time_per_output_token_seconds_sum 3\nvllm_time_per_output_token_seconds_count 6\n",
        )
        .unwrap();
        let ms = tpot_window_ms(&fa, &fb).unwrap();
        assert!((ms - 500.0).abs() < 1e-6);
    }

    #[test]
    fn apply_histogram_fallback_when_window_has_no_new_observations() {
        let body = r#"
vllm_time_to_first_token_seconds_sum 1.0
vllm_time_to_first_token_seconds_count 2
vllm_request_time_per_output_token_seconds_sum 0.5
vllm_request_time_per_output_token_seconds_count 4
vllm_request_prompt_tokens_sum 100
vllm_request_prompt_tokens_count 5
"#;
        let s = scrape_from_body(body).unwrap();
        let mut m = VllmRawMetrics::default();
        apply_histogram_window(&s, &s, &mut m);
        assert!((m.ttft_ms.unwrap() - 500.0).abs() < 1e-6);
        assert!((m.tpot_ms.unwrap() - 125.0).abs() < 1e-6);
        assert!((m.prompt_tokens_mean.unwrap() - 20.0).abs() < 1e-6);
    }

    #[test]
    fn parse_tpot_prefers_request_time_per_output_token_seconds() {
        let body = r#"
vllm_request_time_per_output_token_seconds_sum 0.5
vllm_request_time_per_output_token_seconds_count 4
"#;
        let scrape = scrape_from_body(body).unwrap();
        let m = parse_vllm_metrics(&scrape).unwrap();
        assert!((m.tpot_ms.unwrap() - 125.0).abs() < 1e-6);
    }

    #[test]
    fn compute_counter_rates_prefix_external_totals() {
        let a = r#"
vllm_external_prefix_cache_hits 1
vllm_external_prefix_cache_queries 10
"#;
        let b = r#"
vllm_external_prefix_cache_hits 3
vllm_external_prefix_cache_queries 14
"#;
        let (_, hit_rate) = compute_counter_rates(
            &scrape_from_body(a).unwrap(),
            &scrape_from_body(b).unwrap(),
            1.0,
        );
        assert!((hit_rate.unwrap() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn request_success_and_preemptions_delta_per_sec() {
        let a = "vllm_request_success_total 100\nvllm_num_preemptions_total 4\n";
        let b = "vllm_request_success_total 150\nvllm_num_preemptions_total 9\n";
        let fa = scrape_from_body(a).unwrap();
        let fb = scrape_from_body(b).unwrap();
        let w = 2.0_f64;
        let qps = counter_delta_per_sec(total_request_success(&fa), total_request_success(&fb), w);
        let pps = counter_delta_per_sec(total_preemptions(&fa), total_preemptions(&fb), w);
        assert!((qps.unwrap() - 25.0).abs() < 1e-9);
        assert!((pps.unwrap() - 2.5).abs() < 1e-9);
    }

    #[test]
    fn parse_single_scrape_leaves_counter_rates_none() {
        let body = "vllm_request_success_total 10\nvllm_num_preemptions_total 3\n";
        let m = parse_vllm_metrics(&scrape_from_body(body).unwrap()).unwrap();
        assert!(m.request_success_per_sec.is_none() && m.num_preemptions_per_sec.is_none());
        assert_eq!(m.request_success_total, Some(10.0));
        assert_eq!(m.num_preemptions_total, Some(3.0));
    }

    #[test]
    fn metrics_url_appends_when_base() {
        assert_eq!(
            metrics_url("http://localhost:8000"),
            "http://localhost:8000/metrics"
        );
        assert_eq!(
            metrics_url("http://localhost:8000/"),
            "http://localhost:8000/metrics"
        );
    }

    #[test]
    fn metrics_url_preserves_full_metrics_path() {
        assert_eq!(
            metrics_url("http://localhost:8000/metrics"),
            "http://localhost:8000/metrics"
        );
    }

    #[test]
    fn cache_config_info_labels_parsed() {
        let body = r#"
# HELP vllm:cache_config_info Information of the cache configuration.
# TYPE vllm:cache_config_info gauge
vllm:cache_config_info{block_size="16",cache_dtype="auto",cpu_offload_gb="0",enable_prefix_caching="True",enable_chunked_prefill="False",num_cpu_blocks="2048",num_gpu_blocks="4096",sliding_window="None"} 1.0
"#;
        let scrape = scrape_from_body(body).unwrap();
        let cc = parse_cache_config_labels(&scrape);
        assert_eq!(cc.block_size, Some(16));
        assert_eq!(cc.num_gpu_blocks, Some(4096));
        assert_eq!(cc.cache_dtype.as_deref(), Some("auto"));
        assert_eq!(cc.enable_prefix_caching, Some(true));
        assert_eq!(cc.enable_chunked_prefill, Some(false));
    }

    #[test]
    fn cache_config_info_absent_yields_default() {
        let body = "vllm_num_requests_running 1\n";
        let scrape = scrape_from_body(body).unwrap();
        let cc = parse_cache_config_labels(&scrape);
        assert!(cc.block_size.is_none());
        assert!(cc.num_gpu_blocks.is_none());
        assert!(cc.cache_dtype.is_none());
        assert!(cc.enable_prefix_caching.is_none());
        assert!(cc.enable_chunked_prefill.is_none());
    }

    #[test]
    fn cache_config_info_fp8_dtype() {
        let body = r#"
vllm:cache_config_info{block_size="32",cache_dtype="fp8",enable_prefix_caching="False"} 1.0
"#;
        let scrape = scrape_from_body(body).unwrap();
        let cc = parse_cache_config_labels(&scrape);
        assert_eq!(cc.block_size, Some(32));
        assert_eq!(cc.cache_dtype.as_deref(), Some("fp8"));
        assert_eq!(cc.enable_prefix_caching, Some(false));
    }
}
