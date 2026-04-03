use std::sync::OnceLock;
use std::thread;
use std::time::Duration;

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
    scrape.samples.iter().find(|s| s.metric == name).and_then(|s| {
        match s.value {
            Value::Gauge(v) | Value::Untyped(v) => Some(v),
            _ => None,
        }
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

    for i in 0..SAMPLE_COUNT {
        let body = fetch_metrics_body(&url)?;
        let scrape = scrape_from_body(&body)?;
        running_samples.push(first_gauge(&scrape, "vllm_num_requests_running"));
        waiting_samples.push(first_gauge(&scrape, "vllm_num_requests_waiting"));
        last_body = Some(body);

        if i + 1 < SAMPLE_COUNT {
            thread::sleep(SAMPLE_INTERVAL);
        }
    }

    let last_body = last_body.context("vLLM gauge window produced no scrapes")?;
    let mut m = parse_vllm_metrics(&last_body)?;
    m.num_requests_running = mean_option(&running_samples);
    m.num_requests_waiting = mean_option(&waiting_samples);

    Ok(m)
}

fn parse_vllm_metrics(body: &str) -> Result<VllmRawMetrics> {
    let scrape = scrape_from_body(body)?;

    let sum_numeric_samples = |name: &str| -> Option<f64> {
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
    };

    let get_histogram_mean_ms = |base: &str| -> Option<f64> {
        let sum = sum_numeric_samples(&format!("{base}_sum"));
        let count = sum_numeric_samples(&format!("{base}_count"));
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

    let generation_tokens_total = sum_numeric_samples("vllm_generation_tokens_total")
        .or_else(|| sum_numeric_samples("vllm_iteration_tokens_total_sum"));

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
}
