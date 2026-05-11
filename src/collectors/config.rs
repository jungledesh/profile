use std::env;
use std::time::Duration;

use crate::collectors::RawSnapshot;

// 2s is enough for a local vLLM; longer hangs are user-visible at startup.
const API_TIMEOUT: Duration = Duration::from_secs(2);

/// vLLM deployment configuration.
/// All fields `Option<T>` — graceful degradation when sources are unavailable.
#[derive(Debug, Clone, Default)]
pub struct VllmConfig {
    pub model_name: Option<String>,
    /// HF repo id or weight path from `/v1/models` `root` when present.
    pub model_root: Option<String>,
    pub max_num_seqs: Option<u32>,
    pub tensor_parallel_size: Option<u32>,
    pub pipeline_parallel_size: Option<u32>,
    pub dtype: Option<String>,
    pub quantization: Option<String>,
    pub max_model_len: Option<u32>,
    pub gpu_memory_utilization: Option<f64>,
    /// KV cache element dtype (e.g. "auto", "fp8", "fp16"). Affects bytes_per_param in decode ceiling.
    pub kv_cache_dtype: Option<String>,
    pub enable_chunked_prefill: Option<bool>,
    pub block_size: Option<u32>,
    pub enable_prefix_caching: Option<bool>,
}

/// Build config from snapshot fields + env vars. No I/O.
/// `max_num_seqs` priority: scrape gauge → CLI flag → env var.
pub(crate) fn config_from_snapshot(snapshot: &RawSnapshot, cli_max_num_seqs: u32) -> VllmConfig {
    let cc = &snapshot.vllm.cache_config;
    VllmConfig {
        model_name: snapshot
            .vllm
            .model_name
            .clone()
            .or_else(|| env_str("VLLM_MODEL")),
        model_root: None,
        max_num_seqs: snapshot
            .vllm
            .max_num_seqs
            .or(Some(cli_max_num_seqs).filter(|&v| v > 0))
            .or_else(|| env_u32("MAX_NUM_SEQS")),
        tensor_parallel_size: env_u32("TENSOR_PARALLEL_SIZE")
            .or_else(|| env_u32("VLLM_TENSOR_PARALLEL_SIZE")),
        pipeline_parallel_size: env_u32("PIPELINE_PARALLEL_SIZE")
            .or_else(|| env_u32("VLLM_PIPELINE_PARALLEL_SIZE")),
        dtype: env_str("DTYPE").or_else(|| env_str("VLLM_DTYPE")),
        quantization: env_str("QUANTIZATION").or_else(|| env_str("VLLM_QUANTIZATION")),
        max_model_len: env_u32("MAX_MODEL_LEN").or_else(|| env_u32("VLLM_MAX_MODEL_LEN")),
        gpu_memory_utilization: env_f64("GPU_MEMORY_UTILIZATION")
            .or_else(|| env_f64("VLLM_GPU_MEMORY_UTILIZATION")),
        kv_cache_dtype: cc
            .cache_dtype
            .clone()
            .or_else(|| env_str("KV_CACHE_DTYPE"))
            .or_else(|| env_str("VLLM_KV_CACHE_DTYPE")),
        enable_chunked_prefill: cc
            .enable_chunked_prefill
            .or_else(|| env_bool("ENABLE_CHUNKED_PREFILL"))
            .or_else(|| env_bool("VLLM_ENABLE_CHUNKED_PREFILL")),
        block_size: cc
            .block_size
            .or_else(|| env_u32("BLOCK_SIZE"))
            .or_else(|| env_u32("VLLM_BLOCK_SIZE")),
        enable_prefix_caching: cc
            .enable_prefix_caching
            .or_else(|| env_bool("ENABLE_PREFIX_CACHING"))
            .or_else(|| env_bool("VLLM_ENABLE_PREFIX_CACHING")),
    }
}

/// Full config: snapshot + env vars + CLI flag, enriched with GET /v1/models where available.
/// The API call is best-effort; any failure falls back to snapshot/env values.
pub fn build_config(
    metrics_url: &str,
    snapshot: &RawSnapshot,
    cli_max_num_seqs: u32,
) -> VllmConfig {
    let mut cfg = config_from_snapshot(snapshot, cli_max_num_seqs);
    let base = base_url_from_metrics(metrics_url);
    let api = fetch_config_from_api(&base);
    cfg.model_name = api.model_name.or(cfg.model_name);
    cfg.model_root = api.model_root.or(cfg.model_root);
    cfg.max_model_len = api.max_model_len.or(cfg.max_model_len);
    cfg
}

/// Strip `/metrics` suffix to get the vLLM server base URL.
fn base_url_from_metrics(input: &str) -> String {
    let t = input.trim().trim_end_matches('/');
    if let Some(base) = t.strip_suffix("/metrics") {
        base.to_string()
    } else {
        t.to_string()
    }
}

/// GET /v1/models and extract model_name + max_model_len. Returns Default on any failure.
fn fetch_config_from_api(base_url: &str) -> VllmConfig {
    let url = format!("{}/v1/models", base_url.trim_end_matches('/'));
    let client = match reqwest::blocking::Client::builder()
        .use_rustls_tls()
        .timeout(API_TIMEOUT)
        .build()
    {
        Ok(c) => c,
        Err(_) => return VllmConfig::default(),
    };
    let text = match client.get(&url).send().and_then(|r| r.text()) {
        Ok(t) => t,
        Err(_) => return VllmConfig::default(),
    };
    let val: serde_json::Value = match serde_json::from_str(&text) {
        Ok(v) => v,
        Err(_) => return VllmConfig::default(),
    };
    let entry = val.get("data").and_then(|d| d.get(0));
    let model_name = entry
        .and_then(|e| e.get("id"))
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .map(str::to_string);
    let model_root = entry
        .and_then(|e| e.get("root"))
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .map(str::to_string);
    let max_model_len = entry
        .and_then(|e| e.get("max_model_len"))
        .and_then(|v| v.as_u64())
        .and_then(|n| u32::try_from(n).ok());
    VllmConfig {
        model_name,
        model_root,
        max_model_len,
        ..VllmConfig::default()
    }
}

fn env_str(key: &str) -> Option<String> {
    env::var(key).ok().filter(|s| !s.is_empty())
}

fn env_u32(key: &str) -> Option<u32> {
    env::var(key).ok()?.parse().ok()
}

fn env_f64(key: &str) -> Option<f64> {
    env::var(key).ok()?.parse().ok()
}

fn env_bool(key: &str) -> Option<bool> {
    parse_bool(&env::var(key).ok()?)
}

pub(crate) fn parse_bool(s: &str) -> Option<bool> {
    match s.to_lowercase().as_str() {
        "1" | "true" | "yes" => Some(true),
        "0" | "false" | "no" => Some(false),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collectors::{GpuRawMetrics, RawSnapshot, VllmRawMetrics};
    use std::time::SystemTime;

    fn mk_snap(model: Option<&str>, max_seqs: Option<u32>) -> RawSnapshot {
        RawSnapshot {
            gpu_observed_at: SystemTime::UNIX_EPOCH,
            vllm_observed_at: SystemTime::UNIX_EPOCH,
            timestamp: SystemTime::UNIX_EPOCH,
            vllm: VllmRawMetrics {
                model_name: model.map(str::to_string),
                max_num_seqs: max_seqs,
                ..Default::default()
            },
            gpu: GpuRawMetrics::default(),
        }
    }

    #[test]
    fn config_from_snapshot_uses_snapshot_fields() {
        let s = mk_snap(Some("test-model"), Some(128));
        let cfg = config_from_snapshot(&s, 0);
        assert_eq!(cfg.model_name, Some("test-model".to_string()));
        assert_eq!(cfg.max_num_seqs, Some(128));
    }

    #[test]
    fn config_from_snapshot_cli_fallback_when_scrape_missing() {
        let s = mk_snap(None, None);
        let cfg = config_from_snapshot(&s, 64);
        assert_eq!(cfg.max_num_seqs, Some(64));
    }

    #[test]
    fn config_from_snapshot_scrape_beats_cli() {
        let s = mk_snap(None, Some(256));
        let cfg = config_from_snapshot(&s, 64);
        assert_eq!(cfg.max_num_seqs, Some(256));
    }

    #[test]
    fn config_from_snapshot_handles_missing_fields() {
        let s = mk_snap(None, None);
        let cfg = config_from_snapshot(&s, 0);
        assert!(cfg.tensor_parallel_size.is_none());
        assert!(cfg.pipeline_parallel_size.is_none());
        assert!(cfg.kv_cache_dtype.is_none());
        assert!(cfg.enable_chunked_prefill.is_none());
        assert!(cfg.block_size.is_none());
        assert!(cfg.enable_prefix_caching.is_none());
    }

    #[test]
    fn config_from_snapshot_prefers_cache_config_labels_over_env() {
        use crate::collectors::types::CacheConfigLabels;
        let mut s = mk_snap(None, None);
        s.vllm.cache_config = CacheConfigLabels {
            block_size: Some(32),
            cache_dtype: Some("fp8".to_string()),
            enable_prefix_caching: Some(true),
            enable_chunked_prefill: Some(false),
        };
        let cfg = config_from_snapshot(&s, 0);
        assert_eq!(cfg.block_size, Some(32));
        assert_eq!(cfg.kv_cache_dtype.as_deref(), Some("fp8"));
        assert_eq!(cfg.enable_prefix_caching, Some(true));
        assert_eq!(cfg.enable_chunked_prefill, Some(false));
    }

    #[test]
    fn base_url_from_metrics_strips_suffix() {
        assert_eq!(
            base_url_from_metrics("http://localhost:8000/metrics"),
            "http://localhost:8000"
        );
        assert_eq!(
            base_url_from_metrics("http://localhost:8000/metrics/"),
            "http://localhost:8000"
        );
        assert_eq!(
            base_url_from_metrics("http://localhost:8000"),
            "http://localhost:8000"
        );
    }

    #[test]
    fn fetch_config_from_api_returns_default_on_bad_url() {
        let cfg = fetch_config_from_api("http://127.0.0.1:1");
        assert!(cfg.model_name.is_none());
        assert!(cfg.max_model_len.is_none());
    }

    #[test]
    fn parse_bool_recognizes_truthy_and_falsy() {
        for s in &["1", "true", "True", "TRUE", "yes", "YES"] {
            assert_eq!(parse_bool(s), Some(true), "expected true for {s:?}");
        }
        for s in &["0", "false", "False", "FALSE", "no", "NO"] {
            assert_eq!(parse_bool(s), Some(false), "expected false for {s:?}");
        }
        assert_eq!(parse_bool(""), None);
        assert_eq!(parse_bool("maybe"), None);
    }
}
