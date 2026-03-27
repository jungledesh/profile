use anyhow::{Context, Result};
use prometheus_parse::{Scrape, Value};

use super::types::VllmRawMetrics;

/// vLLM exposes metrics with a `vllm:` prefix. The `prometheus-parse` crate only accepts
/// `[a-zA-Z0-9_]` in metric names (`\w+`), so lines containing `:` are skipped unless we
/// normalize them to underscores before parsing.
fn normalize_vllm_prometheus_text(body: &str) -> String {
    body.replace("vllm:", "vllm_")
}

/// Fetch raw metrics from vLLM /metrics endpoint.
pub fn collect_vllm_metrics(base_url: &str) -> Result<VllmRawMetrics> {
    let url = format!("{}/metrics", base_url.trim_end_matches('/'));

    let body = reqwest::blocking::get(&url)
        .with_context(|| format!("failed to GET {}", url))?
        .error_for_status()
        .with_context(|| format!("non-success response from {}", url))?
        .text()
        .with_context(|| format!("failed to read response body from {}", url))?;

    parse_vllm_metrics(&body)
}

fn parse_vllm_metrics(body: &str) -> Result<VllmRawMetrics> {
    let normalized = normalize_vllm_prometheus_text(body);
    let scrape = Scrape::parse(normalized.lines().map(|s| Ok(s.to_string())))
        .context("failed to parse Prometheus text format")?;

    let get_gauge = |name: &str| -> Option<f64> {
        scrape
            .samples
            .iter()
            .find(|s| s.metric == name)
            .and_then(|s| {
                if let Value::Gauge(v) = s.value {
                    Some(v)
                } else {
                    None
                }
            })
    };

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

    let num_requests_running = get_gauge("vllm_num_requests_running");
    let num_requests_waiting = get_gauge("vllm_num_requests_waiting");
    let kv_cache_usage_perc = get_gauge("vllm_kv_cache_usage_perc")
        .or_else(|| get_gauge("vllm_gpu_cache_usage_perc"))
        .map(|v| v * 100.0);

    let ttft_ms = get_histogram_mean_ms("vllm_time_to_first_token_seconds");
    let tpot_ms = get_histogram_mean_ms("vllm_time_per_output_token_seconds");
    let prefill_latency_ms = get_histogram_mean_ms("vllm_request_prefill_time_seconds");
    let queue_delay_ms = get_histogram_mean_ms("vllm_request_queue_time_seconds");

    let generation_tokens_total = sum_numeric_samples("vllm_generation_tokens_total")
        .or_else(|| sum_numeric_samples("vllm_iteration_tokens_total_sum"));

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
        max_num_seqs: None,
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
}
