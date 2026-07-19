use crate::context::StaticContext;

/// Absolute epsilon for `gpu_memory_utilization` baseline drift (fraction units).
const GPU_MEM_UTIL_EPS: f64 = 1e-6;

/// Returns true if any config field that affects the physics baseline changed.
///
/// Note: `CacheConfigLabels::{kv_cache_size_tokens,kv_cache_max_concurrency,
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
    use crate::collectors::VllmConfig;
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
}
