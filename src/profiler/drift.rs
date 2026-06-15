use crate::context::StaticContext;

/// Returns true if any config field that affects the physics baseline changed.
pub fn config_changed(prev: &StaticContext, curr: &StaticContext) -> bool {
    prev.config.tensor_parallel_size != curr.config.tensor_parallel_size
        || prev.config.dtype != curr.config.dtype
        || prev.config.kv_cache_dtype != curr.config.kv_cache_dtype
        || prev.config.max_model_len != curr.config.max_model_len
        || prev.config.quantization != curr.config.quantization
        || prev.config.vllm_reported_dtype != curr.config.vllm_reported_dtype
        || prev.config.vllm_reported_quantization != curr.config.vllm_reported_quantization
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
        }
    }

    fn base_cfg() -> VllmConfig {
        VllmConfig {
            tensor_parallel_size: Some(1),
            dtype: Some("bf16".into()),
            kv_cache_dtype: Some("auto".into()),
            max_model_len: Some(8192),
            quantization: None,
            ..Default::default()
        }
    }

    #[test]
    fn no_change_false() {
        let c = base_cfg();
        let a = ctx(c.clone());
        let b = ctx(c);
        assert!(!config_changed(&a, &b));
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
}
