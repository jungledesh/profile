use std::env;
use std::time::Duration;

use crate::collectors::RawSnapshot;

// 2s is enough for a local vLLM; longer hangs are user-visible at startup.
const API_TIMEOUT: Duration = Duration::from_secs(2);
/// Enrichment probes (`/info`, `/server_info`): short so three fallbacks cannot
/// stack to 3× `API_TIMEOUT` when endpoints are missing or hang.
const INFO_PROBE_TIMEOUT: Duration = Duration::from_millis(500);

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
    /// Scheduler token budget (`--max-num-batched-tokens`).
    pub max_num_batched_tokens: Option<u32>,
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
        max_num_batched_tokens: snapshot
            .vllm
            .max_num_batched_tokens
            .or_else(|| env_u32("MAX_NUM_BATCHED_TOKENS"))
            .or_else(|| env_u32("VLLM_MAX_NUM_BATCHED_TOKENS")),
        tensor_parallel_size: None,
        pipeline_parallel_size: env_u32("PIPELINE_PARALLEL_SIZE")
            .or_else(|| env_u32("VLLM_PIPELINE_PARALLEL_SIZE")),
        dtype: env_str("DTYPE").or_else(|| env_str("VLLM_DTYPE")),
        quantization: env_str("QUANTIZATION").or_else(|| env_str("VLLM_QUANTIZATION")),
        max_model_len: env_u32("MAX_MODEL_LEN").or_else(|| env_u32("VLLM_MAX_MODEL_LEN")),
        gpu_memory_utilization: cc
            .gpu_memory_utilization
            .or_else(|| env_f64("GPU_MEMORY_UTILIZATION"))
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
        // Scrape label/gauge often missing on modern vLLM (SchedulerConfig-only).
        apply_info_scheduler_gaps(&mut cfg, &info);
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

/// Fill scheduler knobs from `/info` enrichment only when scrape/env left them unset.
fn apply_info_scheduler_gaps(cfg: &mut VllmConfig, info: &InfoData) {
    if cfg.enable_chunked_prefill.is_none() {
        cfg.enable_chunked_prefill = info.enable_chunked_prefill;
    }
    if cfg.max_num_batched_tokens.is_none() {
        cfg.max_num_batched_tokens = info.max_num_batched_tokens;
    }
}

/// Strip `/metrics` suffix to get the vLLM server base URL.
/// Strips /metrics, preserves any other path segments. Consistent with
/// collector URL handling. Common diagnose URL (scheme://host/metrics)
/// resolves identically to the old scheme://host preflight.
pub(crate) fn base_url_from_metrics(input: &str) -> String {
    let t = input.trim().trim_end_matches('/');
    if let Some(base) = t.strip_suffix("/metrics") {
        base.to_string()
    } else {
        t.to_string()
    }
}

/// Startup-only: served model id from GET `/v1/models`. None on any failure.
pub(crate) fn preflight_served_model_id(url: &str, timeout: Duration) -> Option<String> {
    let client = reqwest::blocking::Client::builder()
        .use_rustls_tls()
        .timeout(timeout)
        .build()
        .ok()?;
    let base = base_url_from_metrics(url);
    let models_url = format!("{}/v1/models", base.trim_end_matches('/'));
    let body = client.get(&models_url).send().ok()?.text().ok()?;
    let json: serde_json::Value = serde_json::from_str(&body).ok()?;
    json["data"][0]["id"]
        .as_str()
        .filter(|s| !s.is_empty())
        .map(str::to_string)
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
    /// Scheduler: chunked prefill. Absent from modern `cache_config_info`.
    enable_chunked_prefill: Option<bool>,
    /// Scheduler: `--max-num-batched-tokens`. Often missing from Prometheus gauges.
    max_num_batched_tokens: Option<u32>,
}

impl InfoData {
    fn empty() -> Self {
        Self {
            dtype: None,
            quantization: None,
            enable_chunked_prefill: None,
            max_num_batched_tokens: None,
        }
    }

    fn merge_from(&mut self, other: InfoData) {
        if self.dtype.is_none() {
            self.dtype = other.dtype;
        }
        if self.quantization.is_none() {
            self.quantization = other.quantization;
        }
        if self.enable_chunked_prefill.is_none() {
            self.enable_chunked_prefill = other.enable_chunked_prefill;
        }
        if self.max_num_batched_tokens.is_none() {
            self.max_num_batched_tokens = other.max_num_batched_tokens;
        }
    }

    /// Both scheduler knobs known: no need to probe further endpoints.
    fn scheduler_complete(&self) -> bool {
        self.enable_chunked_prefill.is_some() && self.max_num_batched_tokens.is_some()
    }
}

/// Best-effort: GET `/info`, then `/server_info` (+ `?config_format=json`).
/// Modern vLLM puts chunked-prefill / batched-tokens on `SchedulerConfig`, which
/// is not exported on `cache_config_info`. These endpoints are the fallback.
///
/// Graceful across versions: 404 / DEV_MODE-gated `/server_info` / HTML error
/// pages / empty bodies are ignored. Missing fields stay `None` (unknown), never
/// fabricated. Scrape/env values already on `VllmConfig` are never overwritten.
/// Stops once both scheduler fields are filled so later probes are not paid.
fn fetch_info(client: &reqwest::blocking::Client, base_url: &str) -> InfoData {
    let base = base_url.trim_end_matches('/');
    let mut out = InfoData::empty();
    for path in ["/info", "/server_info?config_format=json", "/server_info"] {
        if out.scheduler_complete() {
            break;
        }
        let url = format!("{base}{path}");
        // Per-request timeout overrides the client default so missing endpoints
        // cannot burn three full `API_TIMEOUT` waits at diagnose startup.
        let Ok(resp) = client.get(&url).timeout(INFO_PROBE_TIMEOUT).send() else {
            continue;
        };
        if !resp.status().is_success() {
            continue;
        }
        let Ok(text) = resp.text() else {
            continue;
        };
        if text.trim().is_empty() {
            continue;
        }
        out.merge_from(parse_info_body(&text));
    }
    out
}

fn looks_like_html(text: &str) -> bool {
    let t = text.trim_start();
    t.starts_with('<')
        || t.get(..15)
            .is_some_and(|s| s.eq_ignore_ascii_case("<!doctype html"))
}

fn parse_info_body(text: &str) -> InfoData {
    let mut data = InfoData {
        dtype: parse_dtype_from_info_body(text),
        quantization: parse_quantization_from_info_body(text),
        enable_chunked_prefill: None,
        max_num_batched_tokens: None,
    };
    if let Ok(val) = serde_json::from_str::<serde_json::Value>(text) {
        data.enable_chunked_prefill = parse_chunked_prefill_from_json(&val);
        data.max_num_batched_tokens = parse_max_num_batched_tokens_from_json(&val);
        // `server_info` sometimes wraps the snapshot as a string under vllm_config.
        if (data.enable_chunked_prefill.is_none() || data.max_num_batched_tokens.is_none())
            && let Some(s) = val.get("vllm_config").and_then(|v| v.as_str())
        {
            if data.enable_chunked_prefill.is_none() {
                data.enable_chunked_prefill = parse_chunked_prefill_from_text(s);
            }
            if data.max_num_batched_tokens.is_none() {
                data.max_num_batched_tokens = parse_max_num_batched_tokens_from_text(s);
            }
        }
    } else if !looks_like_html(text) {
        // Plain-text config dump only. Never scrape HTML error pages for knobs.
        data.enable_chunked_prefill = parse_chunked_prefill_from_text(text);
        data.max_num_batched_tokens = parse_max_num_batched_tokens_from_text(text);
    }
    data
}

fn json_bool_field(v: &serde_json::Value, key: &str) -> Option<bool> {
    match v.get(key)? {
        serde_json::Value::Bool(b) => Some(*b),
        serde_json::Value::String(s) => parse_bool(s),
        _ => None,
    }
}

fn json_u32_field(v: &serde_json::Value, key: &str) -> Option<u32> {
    match v.get(key)? {
        serde_json::Value::Number(n) => {
            if let Some(u) = n.as_u64() {
                return u32::try_from(u).ok();
            }
            // Some dumps emit floats (2048.0). Accept only finite whole numbers.
            let f = n.as_f64()?;
            if f.is_finite() && f > 0.0 && f.fract() == 0.0 && f <= f64::from(u32::MAX) {
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                return Some(f as u32);
            }
            None
        }
        serde_json::Value::String(s) => s.parse().ok(),
        _ => None,
    }
}

/// Nested objects that may hold scheduler knobs across vLLM /info shapes.
const SCHEDULER_JSON_PATHS: &[&[&str]] = &[
    &["scheduler_config"],
    &["scheduler"],
    &["vllm_config", "scheduler_config"],
    &["vllm_config", "scheduler"],
];

fn json_at_path<'a>(val: &'a serde_json::Value, path: &[&str]) -> Option<&'a serde_json::Value> {
    let mut cur = val;
    for key in path {
        cur = cur.get(*key)?;
    }
    Some(cur)
}

fn parse_chunked_prefill_from_json(val: &serde_json::Value) -> Option<bool> {
    for path in SCHEDULER_JSON_PATHS {
        if let Some(node) = json_at_path(val, path)
            && let Some(b) = json_bool_field(node, "enable_chunked_prefill")
                .or_else(|| json_bool_field(node, "chunked_prefill_enabled"))
        {
            return Some(b);
        }
    }
    json_bool_field(val, "enable_chunked_prefill")
        .or_else(|| json_bool_field(val, "chunked_prefill_enabled"))
}

fn parse_max_num_batched_tokens_from_json(val: &serde_json::Value) -> Option<u32> {
    for path in SCHEDULER_JSON_PATHS {
        if let Some(node) = json_at_path(val, path)
            && let Some(n) = json_u32_field(node, "max_num_batched_tokens")
        {
            return Some(n);
        }
    }
    json_u32_field(val, "max_num_batched_tokens")
}

fn parse_chunked_prefill_from_text(text: &str) -> Option<bool> {
    for key in ["enable_chunked_prefill=", "chunked_prefill_enabled="] {
        if let Some(rest) = text.split(key).nth(1) {
            let token = rest.split([',', ' ', '\n', '\'', '"']).next().unwrap_or("");
            if let Some(b) = parse_bool(token) {
                return Some(b);
            }
        }
    }
    None
}

fn parse_max_num_batched_tokens_from_text(text: &str) -> Option<u32> {
    let key = "max_num_batched_tokens=";
    let rest = text.split(key).nth(1)?;
    let token = rest
        .split(|c: char| !c.is_ascii_digit())
        .next()
        .unwrap_or("");
    token.parse().ok()
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
            host_memory: None,
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
            gpu_memory_utilization: Some(0.85),
            ..Default::default()
        };
        let cfg = config_from_snapshot(&s, None);
        assert_eq!(cfg.block_size, Some(32));
        assert_eq!(cfg.kv_cache_dtype.as_deref(), Some("fp8"));
        assert_eq!(cfg.enable_prefix_caching, Some(true));
        assert_eq!(cfg.enable_chunked_prefill, Some(false));
        assert_eq!(cfg.gpu_memory_utilization, Some(0.85));
    }

    #[test]
    fn config_from_snapshot_reads_max_num_batched_tokens() {
        let mut s = mk_snap(None, None);
        s.vllm.max_num_batched_tokens = Some(2048);
        let cfg = config_from_snapshot(&s, None);
        assert_eq!(cfg.max_num_batched_tokens, Some(2048));
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
        assert!(info.enable_chunked_prefill.is_none());
        assert!(info.max_num_batched_tokens.is_none());
    }

    #[test]
    fn parse_info_body_reads_scheduler_config() {
        let body = r#"{
            "dtype":"bfloat16",
            "scheduler_config":{
                "enable_chunked_prefill":true,
                "max_num_batched_tokens":2048
            }
        }"#;
        let info = parse_info_body(body);
        assert_eq!(info.enable_chunked_prefill, Some(true));
        assert_eq!(info.max_num_batched_tokens, Some(2048));
    }

    #[test]
    fn parse_info_body_reads_scheduler_nested_alias() {
        let body = r#"{
            "scheduler":{
                "chunked_prefill_enabled":false,
                "max_num_batched_tokens":"8192"
            }
        }"#;
        let info = parse_info_body(body);
        assert_eq!(info.enable_chunked_prefill, Some(false));
        assert_eq!(info.max_num_batched_tokens, Some(8192));
    }

    #[test]
    fn parse_info_body_reads_server_info_text_vllm_config() {
        let body = r#"{
            "vllm_config":"model='x', enable_chunked_prefill=True, max_num_batched_tokens=2048, dtype=torch.bfloat16"
        }"#;
        let info = parse_info_body(body);
        assert_eq!(info.enable_chunked_prefill, Some(true));
        assert_eq!(info.max_num_batched_tokens, Some(2048));
    }

    #[test]
    fn parse_chunked_prefill_from_plain_text() {
        assert_eq!(
            parse_chunked_prefill_from_text("foo chunked_prefill_enabled=False bar"),
            Some(false)
        );
        assert_eq!(
            parse_max_num_batched_tokens_from_text("max_num_batched_tokens=4096,"),
            Some(4096)
        );
    }

    #[test]
    fn parse_info_body_html_error_page_yields_unknown_scheduler() {
        // 404 / DEV_MODE-gated pages must not invent knobs.
        let body = r#"<!DOCTYPE html><html><body>Not Found
        enable_chunked_prefill=True max_num_batched_tokens=2048
        </body></html>"#;
        let info = parse_info_body(body);
        assert!(info.enable_chunked_prefill.is_none());
        assert!(info.max_num_batched_tokens.is_none());
    }

    #[test]
    fn parse_info_body_json_without_scheduler_stays_unknown() {
        let info = parse_info_body(r#"{"detail":"Not Found"}"#);
        assert!(info.enable_chunked_prefill.is_none());
        assert!(info.max_num_batched_tokens.is_none());
        let info = parse_info_body("{}");
        assert!(info.enable_chunked_prefill.is_none());
        assert!(info.max_num_batched_tokens.is_none());
    }

    #[test]
    fn parse_info_body_accepts_float_batched_tokens() {
        let body = r#"{"scheduler_config":{"max_num_batched_tokens":2048.0}}"#;
        let info = parse_info_body(body);
        assert_eq!(info.max_num_batched_tokens, Some(2048));
    }

    #[test]
    fn info_merge_fills_gaps_only() {
        let mut a = InfoData {
            dtype: Some("bf16".into()),
            quantization: None,
            enable_chunked_prefill: Some(true),
            max_num_batched_tokens: None,
        };
        a.merge_from(InfoData {
            dtype: Some("fp16".into()),
            quantization: Some("awq".into()),
            enable_chunked_prefill: Some(false),
            max_num_batched_tokens: Some(4096),
        });
        assert_eq!(a.dtype.as_deref(), Some("bf16"));
        assert_eq!(a.quantization.as_deref(), Some("awq"));
        assert_eq!(a.enable_chunked_prefill, Some(true));
        assert_eq!(a.max_num_batched_tokens, Some(4096));
    }

    #[test]
    fn scheduler_complete_only_when_both_knobs_known() {
        let mut d = InfoData::empty();
        assert!(!d.scheduler_complete());
        d.enable_chunked_prefill = Some(true);
        assert!(!d.scheduler_complete());
        d.max_num_batched_tokens = Some(2048);
        assert!(d.scheduler_complete());
    }

    #[test]
    fn scrape_chunked_flag_not_clobbered_when_info_absent() {
        // Older builds: label on cache_config_info. Enrichment None must not clear it.
        let mut s = mk_snap(None, None);
        s.vllm.cache_config.enable_chunked_prefill = Some(false);
        s.vllm.max_num_batched_tokens = Some(2048);
        let cfg = config_from_snapshot(&s, None);
        assert_eq!(cfg.enable_chunked_prefill, Some(false));
        assert_eq!(cfg.max_num_batched_tokens, Some(2048));
        let mut enriched = cfg.clone();
        apply_info_scheduler_gaps(&mut enriched, &InfoData::empty());
        assert_eq!(enriched.enable_chunked_prefill, Some(false));
        assert_eq!(enriched.max_num_batched_tokens, Some(2048));
    }

    #[test]
    fn apply_info_scheduler_gaps_fills_only_unset() {
        let mut cfg = VllmConfig {
            enable_chunked_prefill: Some(true),
            max_num_batched_tokens: None,
            ..VllmConfig::default()
        };
        apply_info_scheduler_gaps(
            &mut cfg,
            &InfoData {
                dtype: None,
                quantization: None,
                enable_chunked_prefill: Some(false),
                max_num_batched_tokens: Some(4096),
            },
        );
        assert_eq!(cfg.enable_chunked_prefill, Some(true));
        assert_eq!(cfg.max_num_batched_tokens, Some(4096));
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
