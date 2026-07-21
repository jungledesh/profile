mod math;
mod roofline;

pub use math::{
    ACTIVATION_KV_BUFFER_GB, attn_pages, bytes_per_seq, catalog_hybrid_state_bytes,
    catalog_model_weight_gb, catalog_state_pages, counterfactual_concurrency, kv_bytes_per_element,
    observed_state_pages, page_model_fits, state_dtype_bytes, weight_gb,
};
pub use roofline::{
    CeilingEstimate, CostEstimate, CostSource, PhysicsBaseline, WeightDtypeSource, compute,
};
