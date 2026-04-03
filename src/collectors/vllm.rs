use std::sync::OnceLock;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use prometheus_parse::{Scrape, Value};

use super::types::VllmRawMetrics;

const SAMPLE_COUNT: usize = 8;
const SAMPLE_INTERVAL: Duration = Duration::from_millis(250);
const REQ_TIMEOUT: Duration = Duration::from_secs(10);

/// vLLM exposes metrics with a `vllm:` prefix. The `prometheus-parse` crate only accepts
/// `[a-zA-Z0-9_]` in metric names (`\w+`), so lines containing `:` are skipped unless we
/// normalize them to underscores before parsing.
fn normalize_vllm_prometheus_text(body: &str) -> String {
    body.replace("vllm:", "vllm_")
}

fn http_client() -> &'static reqwest::blocking::Client {
    static CLIENT: OnceLock<reqwest::blocking::Client> = OnceLock::new();
    CLIENT.get_or_init(|| {
        reqwest::blocking::Client::builder()
            .timeout(REQ_TIMEOUT)
            .build()
            .expect("reqwest ClientBuilder with rustls")
    })
}

fn metrics_url(base_url: &str) -> String {
    format!("{}/metrics", base_url.trim_end_matches('/'))
}

fn fetch_metrics_body(url: &str) -> Result<String> {
    let client = http_client();
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

fn mean_option(values: &[Option<f64>]) -> Option<f64> {
    let mut sum = 0.0;
    let mut n = 0u32;
    for v in values {
        if let Some(x) = *v {
            sum += x;
            n += 1;
        }
    }
    (n > 0).then_some(sum / f64::from(n))
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

/// `(last - first) / window_secs` when monotonic; `None` on reset or bad window.
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
    Some(d / window_secs)
}

/// Δhits / Δqueries when Δqueries > 0 and counters did not reset.
fn prefix_hit_rate_window(
    first_hits: Option<f64>,
    first_queries: Option<f64>,
    last_hits: Option<f64>,
    last_queries: Option<f64>,
) -> Option<f64> {
    let fq = first_queries?;
    let lq = last_queries?;
    let dq = lq - fq;
    if dq <= 0.0 {
        return None;
    }
    let fh = first_hits?;
    let lh = last_hits?;
    let dh = lh - fh;
    if dh < 0.0 {
        return None;
    }
    Some(dh / dq)
}

/// Same logic as the first→last `/metrics` window in [`collect_vllm_metrics`].
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
    let prefix = prefix_hit_rate_window(
        sum_metric_samples(first, "vllm_prefix_cache_hits"),
        sum_metric_samples(first, "vllm_prefix_cache_queries"),
        sum_metric_samples(last, "vllm_prefix_cache_hits"),
        sum_metric_samples(last, "vllm_prefix_cache_queries"),
    );
    (gen_per_sec, prefix)
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

/// Fetch raw metrics from vLLM /metrics endpoint.
pub fn collect_vllm_metrics(base_url: &str) -> Result<VllmRawMetrics> {
    let url = metrics_url(base_url);
    let mut running_samples = Vec::with_capacity(SAMPLE_COUNT);
    let mut waiting_samples = Vec::with_capacity(SAMPLE_COUNT);
    let mut last_body: Option<String> = None;
    let mut window_start: Option<Instant> = None;
    let mut first_body: Option<String> = None;

    for i in 0..SAMPLE_COUNT {
        let body = fetch_metrics_body(&url)?;
        let scrape = scrape_from_body(&body)?;

        if i == 0 {
            window_start = Some(Instant::now());
            first_body = Some(body.clone());
        }

        running_samples.push(first_gauge(&scrape, "vllm_num_requests_running"));
        waiting_samples.push(first_gauge(&scrape, "vllm_num_requests_waiting"));
        last_body = Some(body);

        if i + 1 < SAMPLE_COUNT {
            thread::sleep(SAMPLE_INTERVAL);
        }
    }

    let last_body = last_body.context("vLLM gauge window produced no scrapes")?;
    let window_secs = window_start
        .map(|t| t.elapsed().as_secs_f64())
        .unwrap_or(0.0);

    let first_scrape = scrape_from_body(
        first_body
            .as_deref()
            .context("vLLM gauge window missing first body")?,
    )?;
    let last_scrape = scrape_from_body(&last_body)?;
    let mut m = parse_vllm_metrics(&last_body)?;
    m.num_requests_running = mean_option(&running_samples);
    m.num_requests_waiting = mean_option(&waiting_samples);

    let (gen_per_sec, prefix_hit) = compute_counter_rates(&first_scrape, &last_scrape, window_secs);
    m.generation_tokens_per_sec = gen_per_sec;
    m.prefix_cache_hit_rate = prefix_hit;

    Ok(m)
}

fn parse_vllm_metrics(body: &str) -> Result<VllmRawMetrics> {
    let scrape = scrape_from_body(body)?;

    let get_histogram_mean_ms = |base: &str| -> Option<f64> {
        let sum = sum_metric_samples(&scrape, &format!("{base}_sum"));
        let count = sum_metric_samples(&scrape, &format!("{base}_count"));
        match (sum, count) {
            (Some(s), Some(c)) if c > 0.0 => Some((s / c) * 1000.0),
            _ => None,
        }
    };

    let model_name = scrape
        .samples
        .iter()
        .find(|s| s.metric.starts_with("vllm_"))
        .and_then(|s| s.labels.get("model_name").map(str::to_string));

    let num_requests_running = first_gauge(&scrape, "vllm_num_requests_running");
    let num_requests_waiting = first_gauge(&scrape, "vllm_num_requests_waiting");
    let kv_cache_usage_perc = first_gauge(&scrape, "vllm_kv_cache_usage_perc")
        .or_else(|| first_gauge(&scrape, "vllm_gpu_cache_usage_perc"))
        .map(|v| v * 100.0);

    let ttft_ms = get_histogram_mean_ms("vllm_time_to_first_token_seconds");
    let tpot_ms = get_histogram_mean_ms("vllm_time_per_output_token_seconds");
    let prefill_latency_ms = get_histogram_mean_ms("vllm_request_prefill_time_seconds");
    let queue_delay_ms = get_histogram_mean_ms("vllm_request_queue_time_seconds");

    let generation_tokens_total = total_generation_tokens(&scrape);

    let max_num_seqs = max_num_seqs_from_gauge(&scrape);

    Ok(VllmRawMetrics {
        model_name,
        num_requests_running,
        num_requests_waiting,
        kv_cache_usage_perc,
        ttft_ms,
        tpot_ms,
        prefill_latency_ms,
        queue_delay_ms,
        generation_tokens_total,
        generation_tokens_per_sec: None,
        prefix_cache_hit_rate: None,
        max_num_seqs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn colon_prefixed_vllm_metrics_parse_after_normalize() {
        let body = r#"
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_sum{model_name="llama3"} 0.5
vllm:time_to_first_token_seconds_count{model_name="llama3"} 4
vllm:generation_tokens_total{model_name="llama3"} 99
"#;
        let m = parse_vllm_metrics(body).unwrap();
        assert!((m.ttft_ms.unwrap() - 125.0).abs() < 1e-6);
        assert_eq!(m.generation_tokens_total, Some(99.0));
        assert_eq!(m.model_name.as_deref(), Some("llama3"));
        assert!(
            m.generation_tokens_per_sec.is_none() && m.prefix_cache_hit_rate.is_none(),
            "parse-only must not set windowed counters"
        );
    }

    #[test]
    fn mean_option_skips_none_and_averages_rest() {
        assert_eq!(mean_option(&[None, Some(2.0), None, Some(4.0)]), Some(3.0));
        assert_eq!(mean_option(&[None, None]), None);
        assert_eq!(mean_option(&[]), None);
    }

    #[test]
    fn gauge_window_mean_and_max_num_seqs_last_scrape() {
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
        let bodies = [a, a, a, a, a, a, a, b];
        let mut running = Vec::with_capacity(8);
        let mut waiting = Vec::with_capacity(8);
        let mut max_last = None;
        for (i, body) in bodies.iter().enumerate() {
            let scrape = scrape_from_body(body).unwrap();
            running.push(first_gauge(&scrape, "vllm_num_requests_running"));
            waiting.push(first_gauge(&scrape, "vllm_num_requests_waiting"));
            if i + 1 == bodies.len() {
                max_last = max_num_seqs_from_gauge(&scrape);
            }
        }
        assert!((mean_option(&running).unwrap() - 2.25).abs() < 1e-9);
        assert!((mean_option(&waiting).unwrap() - 0.875).abs() < 1e-9);
        assert_eq!(max_last, Some(256));
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
    fn prefix_hit_rate_window_delta() {
        assert_eq!(
            prefix_hit_rate_window(Some(1.0), Some(10.0), Some(3.0), Some(20.0)),
            Some(0.2)
        );
        assert_eq!(
            prefix_hit_rate_window(Some(1.0), Some(10.0), Some(3.0), Some(10.0)),
            None
        );
        assert_eq!(
            prefix_hit_rate_window(Some(5.0), Some(10.0), Some(3.0), Some(20.0)),
            None
        );
    }

    #[test]
    fn prefix_hit_rate_window_all_hits() {
        assert_eq!(
            prefix_hit_rate_window(Some(0.0), Some(0.0), Some(10.0), Some(10.0)),
            Some(1.0)
        );
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
        assert!(prefix.is_none());
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
        assert!(hit_rate.is_none());
    }

    #[test]
    fn compute_counter_rates_prefix_hits_drop_treated_as_reset() {
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
        assert!(hit_rate.is_none());
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
        assert!(hit_rate.is_none());
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
}
