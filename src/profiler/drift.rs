use crate::collectors::VllmConfig;
use crate::context::StaticContext;

/// Absolute epsilon for `gpu_memory_utilization` baseline drift (fraction units).
const GPU_MEM_UTIL_EPS: f64 = 1e-6;

/// True when both sides lack a scraped value for `field` (None == None is not evidence).
fn both_unread<T>(a: Option<T>, b: Option<T>) -> bool {
    a.is_none() && b.is_none()
}

/// Non-baseline knobs used for the "no change" claim.
fn non_baseline_knobs_visible(cfg: &VllmConfig) -> bool {
    cfg.max_num_seqs.is_some()
        || cfg.max_num_batched_tokens.is_some()
        || cfg.enable_chunked_prefill.is_some()
        || cfg.enable_prefix_caching.is_some()
        || cfg.enforce_eager.is_some()
}

/// Fix text named a scrapeable knob that is unread on both windows.
///
/// Driven by primary `display_lines` flag names, not rule tables.
pub fn prescribed_knob_unread(
    prev: &VllmConfig,
    curr: &VllmConfig,
    display_lines: &[String],
) -> bool {
    let text = display_lines.join("\n");
    let mut unread = false;
    if text.contains("--max-num-batched-tokens") {
        unread |= both_unread(prev.max_num_batched_tokens, curr.max_num_batched_tokens);
    }
    if text.contains("--enable-chunked-prefill") {
        unread |= both_unread(prev.enable_chunked_prefill, curr.enable_chunked_prefill);
    }
    if text.contains("--max-num-seqs") {
        unread |= both_unread(prev.max_num_seqs, curr.max_num_seqs);
    }
    if text.contains("--enable-prefix-caching") {
        unread |= both_unread(prev.enable_prefix_caching, curr.enable_prefix_caching);
    }
    if text.contains("--enforce-eager") {
        unread |= both_unread(prev.enforce_eager, curr.enforce_eager);
    }
    if text.contains("--max-model-len") {
        unread |= both_unread(prev.max_model_len, curr.max_model_len);
    }
    if text.contains("--gpu-memory-utilization") {
        unread |= both_unread(prev.gpu_memory_utilization, curr.gpu_memory_utilization);
    }
    if text.contains("--kv-cache-dtype") {
        unread |= both_unread(
            prev.kv_cache_dtype.as_deref(),
            curr.kv_cache_dtype.as_deref(),
        );
    }
    unread
}

/// When no config drift fired: whether "No change detected" would lie.
///
/// 1. Prior fix named a knob we still cannot read → unverifiable.
/// 2. No prior fix (or none with scrapeable knobs): need a visible non-baseline
///    knob to claim no change; otherwise unverifiable.
pub fn change_unverifiable(
    prev: &VllmConfig,
    curr: &VllmConfig,
    prescribed_display_lines: Option<&[String]>,
) -> bool {
    if let Some(lines) = prescribed_display_lines.filter(|l| !l.is_empty()) {
        if prescribed_knob_unread(prev, curr, lines) {
            return true;
        }
        // Prescription present; scrapeable knobs readable (or none named) → verifiable.
        return false;
    }
    !non_baseline_knobs_visible(prev) && !non_baseline_knobs_visible(curr)
}

/// Returns true if any config field that affects the physics baseline changed.
///
/// Note: `CacheConfigLabels::{kv_cache_max_concurrency,
/// mamba_block_size,mamba_page_size_padded}` are allocator consequences of
/// config, not config themselves. Do not add them here.
pub fn config_changed(prev: &StaticContext, curr: &StaticContext) -> bool {
    prev.config.tensor_parallel_size != curr.config.tensor_parallel_size
        || prev.config.pipeline_parallel_size != curr.config.pipeline_parallel_size
        || prev.config.dtype != curr.config.dtype
        || prev.config.kv_cache_dtype != curr.config.kv_cache_dtype
        || prev.config.max_model_len != curr.config.max_model_len
        || prev.config.quantization != curr.config.quantization
        || prev.config.vllm_reported_dtype != curr.config.vllm_reported_dtype
        || prev.config.vllm_reported_quantization != curr.config.vllm_reported_quantization
        || prev.config.block_size != curr.config.block_size
        || f64_opt_changed(
            prev.config.gpu_memory_utilization,
            curr.config.gpu_memory_utilization,
            GPU_MEM_UTIL_EPS,
        )
}

/// True when a non-baseline config knob changed (scheduler / caching knobs).
/// Does not reset the physics baseline; stdout only labels "Config changed."
pub fn non_baseline_drifted(prev: &StaticContext, curr: &StaticContext) -> bool {
    prev.config.max_num_seqs != curr.config.max_num_seqs
        || prev.config.max_num_batched_tokens != curr.config.max_num_batched_tokens
        || prev.config.enable_chunked_prefill != curr.config.enable_chunked_prefill
        || prev.config.enable_prefix_caching != curr.config.enable_prefix_caching
        || prev.config.enforce_eager != curr.config.enforce_eager
}

fn f64_opt_changed(a: Option<f64>, b: Option<f64>, eps: f64) -> bool {
    match (a, b) {
        (Some(x), Some(y)) if x.is_finite() && y.is_finite() => (x - y).abs() > eps,
        (None, None) => false,
        (Some(x), None) | (None, Some(x)) => x.is_finite(),
        _ => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::{GPUModel, ModelArch};

    fn ctx(cfg: VllmConfig) -> StaticContext {
        StaticContext {
            model: ModelArch::default(),
            gpu: GPUModel::default(),
            config: cfg,
            fp8_compiler_available: false,
        }
    }

    fn base_cfg() -> VllmConfig {
        VllmConfig {
            tensor_parallel_size: Some(1),
            dtype: Some("bf16".into()),
            kv_cache_dtype: Some("auto".into()),
            max_model_len: Some(8192),
            quantization: None,
            gpu_memory_utilization: Some(0.90),
            pipeline_parallel_size: Some(1),
            block_size: Some(16),
            ..Default::default()
        }
    }

    #[test]
    fn no_change_false() {
        let c = base_cfg();
        let a = ctx(c.clone());
        let b = ctx(c);
        assert!(!config_changed(&a, &b));
        assert!(!non_baseline_drifted(&a, &b));
    }

    #[test]
    fn tensor_parallel_change_true() {
        let mut c2 = base_cfg();
        c2.tensor_parallel_size = Some(2);
        assert!(config_changed(&ctx(base_cfg()), &ctx(c2)));
    }

    #[test]
    fn dtype_change_true() {
        let mut c2 = base_cfg();
        c2.dtype = Some("fp16".into());
        assert!(config_changed(&ctx(base_cfg()), &ctx(c2)));
    }

    #[test]
    fn kv_cache_dtype_change_true() {
        let mut c2 = base_cfg();
        c2.kv_cache_dtype = Some("fp8".into());
        assert!(config_changed(&ctx(base_cfg()), &ctx(c2)));
    }

    #[test]
    fn max_model_len_change_true() {
        let mut c2 = base_cfg();
        c2.max_model_len = Some(4096);
        assert!(config_changed(&ctx(base_cfg()), &ctx(c2)));
    }

    #[test]
    fn quantization_change_true() {
        let mut c2 = base_cfg();
        c2.quantization = Some("awq".into());
        assert!(config_changed(&ctx(base_cfg()), &ctx(c2)));
    }

    #[test]
    fn vllm_reported_dtype_change_true() {
        let mut c2 = base_cfg();
        c2.vllm_reported_dtype = Some("fp8".into());
        assert!(config_changed(&ctx(base_cfg()), &ctx(c2)));
    }

    #[test]
    fn vllm_reported_dtype_appearing_triggers_rebaseline() {
        let mut c2 = base_cfg();
        c2.vllm_reported_dtype = Some("bfloat16".into());
        assert!(config_changed(&ctx(base_cfg()), &ctx(c2)));
    }

    #[test]
    fn vllm_reported_quantization_change_triggers_rebaseline() {
        let mut c2 = base_cfg();
        c2.vllm_reported_quantization = Some("awq".into());
        assert!(config_changed(&ctx(base_cfg()), &ctx(c2)));
    }

    #[test]
    fn gpu_memory_utilization_change_is_baseline_drift() {
        let mut c2 = base_cfg();
        c2.gpu_memory_utilization = Some(0.95);
        assert!(config_changed(&ctx(base_cfg()), &ctx(c2)));
    }

    #[test]
    fn gpu_memory_utilization_within_epsilon_not_drift() {
        let mut c2 = base_cfg();
        c2.gpu_memory_utilization = Some(0.90 + 1e-9);
        assert!(!config_changed(&ctx(base_cfg()), &ctx(c2)));
    }

    #[test]
    fn pipeline_parallel_change_is_baseline_drift() {
        let mut c2 = base_cfg();
        c2.pipeline_parallel_size = Some(2);
        assert!(config_changed(&ctx(base_cfg()), &ctx(c2)));
    }

    #[test]
    fn block_size_change_is_baseline_drift() {
        let mut c2 = base_cfg();
        c2.block_size = Some(32);
        assert!(config_changed(&ctx(base_cfg()), &ctx(c2)));
    }

    #[test]
    fn max_num_seqs_change_not_baseline_drift() {
        let mut c2 = base_cfg();
        c2.max_num_seqs = Some(98);
        let prev = ctx(base_cfg());
        let curr = ctx(c2);
        assert!(!config_changed(&prev, &curr));
        assert!(non_baseline_drifted(&prev, &curr));
    }

    #[test]
    fn max_num_batched_tokens_change_is_non_baseline() {
        let mut c2 = base_cfg();
        c2.max_num_batched_tokens = Some(2048);
        let prev = ctx(base_cfg());
        let curr = ctx(c2);
        assert!(!config_changed(&prev, &curr));
        assert!(non_baseline_drifted(&prev, &curr));
    }

    #[test]
    fn enable_chunked_prefill_change_is_non_baseline() {
        let mut c2 = base_cfg();
        c2.enable_chunked_prefill = Some(true);
        let prev = ctx(base_cfg());
        let curr = ctx(c2);
        assert!(non_baseline_drifted(&prev, &curr));
        assert!(!config_changed(&prev, &curr));
    }

    #[test]
    fn enable_prefix_caching_change_is_non_baseline() {
        let mut c2 = base_cfg();
        c2.enable_prefix_caching = Some(true);
        assert!(non_baseline_drifted(&ctx(base_cfg()), &ctx(c2)));
    }

    #[test]
    fn enforce_eager_change_is_non_baseline() {
        let mut c2 = base_cfg();
        c2.enforce_eager = Some(true);
        assert!(non_baseline_drifted(&ctx(base_cfg()), &ctx(c2)));
    }

    #[test]
    fn non_baseline_false_when_unchanged() {
        let c = base_cfg();
        assert!(!non_baseline_drifted(&ctx(c.clone()), &ctx(c)));
    }

    #[test]
    fn prescribed_batched_tokens_unread_both_sides() {
        let lines = vec![
            "      • Set --max-num-batched-tokens to 2048 (default); page floor is 784 (do not go below). Lower for smoother TPOT, raise for lower TTFT."
                .to_string(),
        ];
        let prev = VllmConfig::default();
        let curr = VllmConfig::default();
        assert!(prescribed_knob_unread(&prev, &curr, &lines));
        assert!(change_unverifiable(&prev, &curr, Some(&lines)));
    }

    #[test]
    fn prescribed_batched_tokens_readable_not_unverifiable() {
        let lines = vec![
            "      • Set --max-num-batched-tokens to 2048 (default); page floor is 784 (do not go below). Lower for smoother TPOT, raise for lower TTFT."
                .to_string(),
        ];
        let prev = VllmConfig {
            max_num_batched_tokens: Some(8192),
            ..Default::default()
        };
        let curr = VllmConfig {
            max_num_batched_tokens: Some(8192),
            ..Default::default()
        };
        assert!(!prescribed_knob_unread(&prev, &curr, &lines));
        assert!(!change_unverifiable(&prev, &curr, Some(&lines)));
    }

    #[test]
    fn prescribed_batched_tokens_floor_above_default_form_still_detected() {
        let lines = vec![
            "      • Default --max-num-batched-tokens is 2048; page floor is 2496 (do not go below). Lower for smoother TPOT, raise for lower TTFT."
                .to_string(),
        ];
        let prev = VllmConfig::default();
        let curr = VllmConfig::default();
        assert!(prescribed_knob_unread(&prev, &curr, &lines));
    }

    #[test]
    fn prescribed_batched_tokens_unread_guide_form_still_detected() {
        let lines = vec![
            "      • --max-num-batched-tokens unread on this server.".to_string(),
            "        Page floor is 1568 (do not go below). Lower for smoother TPOT, raise for lower TTFT."
                .to_string(),
        ];
        let prev = VllmConfig::default();
        let curr = VllmConfig::default();
        assert!(prescribed_knob_unread(&prev, &curr, &lines));
        assert!(change_unverifiable(&prev, &curr, Some(&lines)));
    }

    #[test]
    fn no_prescription_all_non_baseline_unread_is_unverifiable() {
        let prev = VllmConfig::default();
        let curr = VllmConfig::default();
        assert!(change_unverifiable(&prev, &curr, None));
        assert!(change_unverifiable(&prev, &curr, Some(&[])));
    }

    #[test]
    fn no_prescription_visible_knob_allows_no_change_claim() {
        let prev = VllmConfig {
            max_num_seqs: Some(128),
            ..Default::default()
        };
        let curr = VllmConfig {
            max_num_seqs: Some(128),
            ..Default::default()
        };
        assert!(!change_unverifiable(&prev, &curr, None));
    }

    #[test]
    fn prescription_without_scrapeable_flags_not_forced_unverifiable() {
        let lines = vec!["      • Reduce prompt length where possible.".to_string()];
        let prev = VllmConfig::default();
        let curr = VllmConfig::default();
        assert!(!prescribed_knob_unread(&prev, &curr, &lines));
        assert!(!change_unverifiable(&prev, &curr, Some(&lines)));
    }
}
