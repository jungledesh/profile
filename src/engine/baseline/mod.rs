mod math;
mod roofline;

pub use math::{
    ACTIVATION_KV_BUFFER_GB, KvCacheDtypeSource, attn_pages, bytes_per_seq,
    catalog_hybrid_state_bytes, catalog_model_weight_gb, catalog_state_pages,
    counterfactual_concurrency, effective_kv_cache_dtype, kv_bytes_per_element,
    observed_state_pages, page_model_fits, resolve_kv_cache_element, state_dtype_bytes, weight_gb,
};
pub use roofline::{
    CeilingEstimate, CostEstimate, CostSource, PhysicsBaseline, SPEC_GUARD_LIMITER_LINE,
    SPEC_GUARD_WARNING_LINE, SpecDetector, SpecEvidence, WeightDtypeSource,
    baseline_missing_reason, compute,
};
pub(crate) use roofline::{apply_spec_run_or, stronger_spec_evidence};
