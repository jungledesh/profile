use anyhow::{Context, Result};
use prometheus_parse::{Scrape, Value};

use super::types::VllmRawMetrics;

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
    let scrape = Scrape::parse(body.lines().map(|s| Ok(s.to_string())))
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

    let get_number = |name: &str| -> Option<f64> {
        scrape
            .samples
            .iter()
            .find(|s| s.metric == name)
            .and_then(|s| match s.value {
                Value::Gauge(v) | Value::Counter(v) | Value::Untyped(v) => Some(v),
                _ => None,
            })
    };

    let get_histogram_mean_ms = |base: &str| -> Option<f64> {
        let sum = get_number(&format!("{}_sum", base));
        let count = get_number(&format!("{}_count", base));
        match (sum, count) {
            (Some(s), Some(c)) if c > 0.0 => Some((s / c) * 1000.0),
            _ => None,
        }
    };

    let model_name = scrape
        .samples
        .iter()
        .find(|s| s.metric.starts_with("vllm:"))
        .and_then(|s| s.labels.get("model_name").map(str::to_string));

    let num_requests_running = get_gauge("vllm:num_requests_running");
    let num_requests_waiting = get_gauge("vllm:num_requests_waiting");
    let kv_cache_usage_perc = get_gauge("vllm:kv_cache_usage_perc")
        .or_else(|| get_gauge("vllm:gpu_cache_usage_perc"))
        .map(|v| v * 100.0);

    let ttft_ms = get_histogram_mean_ms("vllm:time_to_first_token_seconds");
    let tpot_ms = get_histogram_mean_ms("vllm:time_per_output_token_seconds");
    let prefill_latency_ms = get_histogram_mean_ms("vllm:request_prefill_time_seconds");
    let queue_delay_ms = get_histogram_mean_ms("vllm:request_queue_time_seconds");

    let generation_tokens_total = get_number("vllm:generation_tokens_total")
        .or_else(|| get_number("vllm:iteration_tokens_total_sum"));

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
