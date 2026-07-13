use std::env;
use std::time::Duration;

use crate::collectors::RawSnapshot;

// 2s is enough for a local vLLM; longer hangs are user-visible at startup.
const API_TIMEOUT: Duration = Duration::from_secs(2);

/// vLLM default when --gpu-memory-utilization is not set.
pub const DEFAULT_GPU_MEMORY_UTILIZATION: f64 = 0.90;

/// vLLM deployment configuration.
/// All fields `Option<T>` - graceful degradation when sources are unavailable.
#[derive(Debug, Clone, Default, PartialEq)]
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
    /// KV cache element dtype (e.g. "auto", "fp8", "fp16").
    pub kv_cache_dtype: Option<String>,
    /// Weight dtype reported by vLLM Prometheus label metrics, when available.
    pub vllm_reported_dtype: Option<String>,
    /// True when `vllm_reported_dtype` was resolved from GET `/info` (Prometheus had `auto` or none).
    pub vllm_reported_dtype_resolved: bool,
    /// Quantization scheme from GET /info `model_config.quantization`, or None if unquantized/unknown.
    pub vllm_reported_quantization: Option<String>,
    pub enable_chunked_prefill: Option<bool>,
    pub block_size: Option<u32>,
    pub enable_prefix_caching: Option<bool>,
    /// True when vLLM runs with `--enforce-eager` (GPU graph capture disabled).
    pub enforce_eager: Option<bool>,
    /// Operator-supplied GPU cost ($/hr). Overrides catalog estimate when set.
    pub cost_per_hour: Option<f64>,
}

/// Build config from snapshot fields + env vars. No I/O.
/// `max_num_seqs` priority: scrape gauge → CLI flag → env var.
pub(crate) fn config_from_snapshot(
    snapshot: &RawSnapshot,
    cli_max_num_seqs: Option<u32>,
) -> VllmConfig {
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
            .or(cli_max_num_seqs)
            .or_else(|| env_u32("MAX_NUM_SEQS")),
        tensor_parallel_size: None,
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
        vllm_reported_dtype: snapshot.vllm.model_weight_dtype.clone(),
        vllm_reported_dtype_resolved: false,
        vllm_reported_quantization: None,
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
        enforce_eager: env_bool("ENFORCE_EAGER").or_else(|| env_bool("VLLM_ENFORCE_EAGER")),
        cost_per_hour: None,
    }
}

/// Full config: snapshot + env vars + CLI flag, enriched with GET /v1/models where available.
/// The API call is best-effort; any failure falls back to snapshot/env values.
pub fn build_config(
    metrics_url: &str,
    snapshot: &RawSnapshot,
    cli_max_num_seqs: Option<u32>,
) -> VllmConfig {
    let mut cfg = config_from_snapshot(snapshot, cli_max_num_seqs);
    let base = base_url_from_metrics(metrics_url);
    if let Some(client) = blocking_api_client() {
        let api = fetch_config_from_api(&client, &base);
        cfg.model_name = api.model_name.or(cfg.model_name);
        cfg.model_root = api.model_root.or(cfg.model_root);
        cfg.max_model_len = api.max_model_len.or(cfg.max_model_len);
        let info = fetch_info(&client, &base);
        if (cfg.vllm_reported_dtype.as_deref() == Some("auto") || cfg.vllm_reported_dtype.is_none())
            && let Some(resolved) = info.dtype
        {
            cfg.vllm_reported_dtype = Some(resolved);
            cfg.vllm_reported_dtype_resolved = true;
        }
        cfg.vllm_reported_quantization = info.quantization;
    }
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
fn fetch_config_from_api(client: &reqwest::blocking::Client, base_url: &str) -> VllmConfig {
    let url = format!("{}/v1/models", base_url.trim_end_matches('/'));
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

fn blocking_api_client() -> Option<reqwest::blocking::Client> {
    reqwest::blocking::Client::builder()
        .use_rustls_tls()
        .timeout(API_TIMEOUT)
        .build()
        .ok()
}

struct InfoData {
    dtype: Option<String>,
    quantization: Option<String>,
}

fn fetch_info(client: &reqwest::blocking::Client, base_url: &str) -> InfoData {
    let url = format!("{}/info", base_url.trim_end_matches('/'));
    let text = match client.get(&url).send().and_then(|r| r.text()) {
        Ok(t) => t,
        Err(_) => {
            return InfoData {
                dtype: None,
                quantization: None,
            };
        }
    };
    parse_info_body(&text)
}

fn parse_info_body(text: &str) -> InfoData {
    InfoData {
        dtype: parse_dtype_from_info_body(text),
        quantization: parse_quantization_from_info_body(text),
    }
}

fn parse_quantization_from_info_body(text: &str) -> Option<String> {
    let val: serde_json::Value = serde_json::from_str(text).ok()?;
    let s = val.get("model_config")?.get("quantization")?.as_str()?;
    let t = s.trim().to_ascii_lowercase();
    if t.is_empty() {
        return None;
    }
    Some(t)
}

fn parse_dtype_from_info_body(text: &str) -> Option<String> {
    let val: serde_json::Value = serde_json::from_str(text).ok()?;
    for key in ["dtype", "model_dtype"] {
        if let Some(s) = val.get(key).and_then(|v| v.as_str()) {
            let t = s.trim();
            if !t.is_empty() && !t.eq_ignore_ascii_case("auto") {
                return Some(t.to_string());
            }
        }
    }
    None
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
    use crate::collectors::{RawSnapshot, VllmRawMetrics};
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
            gpus: vec![],
        }
    }

    #[test]
    fn config_from_snapshot_uses_snapshot_fields() {
        let s = mk_snap(Some("test-model"), Some(128));
        let cfg = config_from_snapshot(&s, None);
        assert_eq!(cfg.model_name, Some("test-model".to_string()));
        assert_eq!(cfg.max_num_seqs, Some(128));
    }

    #[test]
    fn config_from_snapshot_cli_fallback_when_scrape_missing() {
        let s = mk_snap(None, None);
        let cfg = config_from_snapshot(&s, Some(64));
        assert_eq!(cfg.max_num_seqs, Some(64));
    }

    #[test]
    fn config_from_snapshot_scrape_beats_cli() {
        let s = mk_snap(None, Some(256));
        let cfg = config_from_snapshot(&s, Some(64));
        assert_eq!(cfg.max_num_seqs, Some(256));
    }

    #[test]
    fn config_from_snapshot_none_when_gauge_and_cli_absent() {
        let s = mk_snap(None, None);
        let cfg = config_from_snapshot(&s, None);
        assert_eq!(cfg.max_num_seqs, None);
    }

    #[test]
    fn config_from_snapshot_handles_missing_fields() {
        let s = mk_snap(None, None);
        let cfg = config_from_snapshot(&s, None);
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
            num_gpu_blocks: None,
            cache_dtype: Some("fp8".to_string()),
            enable_prefix_caching: Some(true),
            enable_chunked_prefill: Some(false),
        };
        let cfg = config_from_snapshot(&s, None);
        assert_eq!(cfg.block_size, Some(32));
        assert_eq!(cfg.kv_cache_dtype.as_deref(), Some("fp8"));
        assert_eq!(cfg.enable_prefix_caching, Some(true));
        assert_eq!(cfg.enable_chunked_prefill, Some(false));
    }

    #[test]
    fn config_from_snapshot_populates_vllm_reported_dtype() {
        let mut s = mk_snap(None, None);
        s.vllm.model_weight_dtype = Some("bfloat16".to_string());
        let cfg = config_from_snapshot(&s, None);
        assert_eq!(cfg.vllm_reported_dtype.as_deref(), Some("bfloat16"));
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
        let client = blocking_api_client().expect("client");
        let cfg = fetch_config_from_api(&client, "http://127.0.0.1:1");
        assert!(cfg.model_name.is_none());
        assert!(cfg.max_model_len.is_none());
    }

    #[test]
    fn parse_dtype_from_info_body_extracts_dtype_and_model_dtype() {
        assert_eq!(
            parse_dtype_from_info_body(r#"{"dtype":"bfloat16"}"#).as_deref(),
            Some("bfloat16")
        );
        assert_eq!(
            parse_dtype_from_info_body(r#"{"model_dtype":"half"}"#).as_deref(),
            Some("half")
        );
    }

    #[test]
    fn parse_dtype_from_info_body_rejects_auto_and_empty() {
        assert!(parse_dtype_from_info_body(r#"{"dtype":"auto"}"#).is_none());
        assert!(parse_dtype_from_info_body(r#"{"dtype":""}"#).is_none());
        assert!(parse_dtype_from_info_body("not json").is_none());
    }

    #[test]
    fn parse_quantization_from_info_body_extracts_from_model_config() {
        assert_eq!(
            parse_quantization_from_info_body(r#"{"model_config":{"quantization":"awq"}}"#)
                .as_deref(),
            Some("awq")
        );
    }

    #[test]
    fn parse_quantization_from_info_body_returns_none_when_null() {
        assert!(
            parse_quantization_from_info_body(r#"{"model_config":{"quantization":null}}"#)
                .is_none()
        );
    }

    #[test]
    fn parse_quantization_from_info_body_returns_none_when_missing() {
        assert!(parse_quantization_from_info_body(r#"{"dtype":"bfloat16"}"#).is_none());
    }

    #[test]
    fn parse_info_body_extracts_both_fields() {
        let body = r#"{"dtype":"bfloat16","model_config":{"quantization":"awq"}}"#;
        let info = parse_info_body(body);
        assert_eq!(info.dtype.as_deref(), Some("bfloat16"));
        assert_eq!(info.quantization.as_deref(), Some("awq"));
    }

    /// Live vLLM only: `cargo test build_config_resolves_auto_dtype_via_info -- --ignored --nocapture`
    #[test]
    #[ignore = "requires live vLLM with GET /info reporting dtype"]
    fn build_config_resolves_auto_dtype_via_info() {
        let mut s = mk_snap(None, None);
        s.vllm.model_weight_dtype = Some("auto".to_string());
        let cfg = build_config("http://localhost:8000/metrics", &s, None);
        assert!(
            cfg.vllm_reported_dtype_resolved,
            "expected /info to resolve auto dtype"
        );
        assert_ne!(cfg.vllm_reported_dtype.as_deref(), Some("auto"));
    }

    /// Live vLLM only: `cargo test build_config_populates_vllm_reported_quantization -- --ignored --nocapture`
    #[test]
    #[ignore = "requires live vLLM with GET /info reporting model_config.quantization"]
    fn build_config_populates_vllm_reported_quantization() {
        let s = mk_snap(None, None);
        let cfg = build_config("http://localhost:8000/metrics", &s, None);
        assert!(
            cfg.vllm_reported_quantization.is_some(),
            "expected /info to populate vllm_reported_quantization"
        );
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
