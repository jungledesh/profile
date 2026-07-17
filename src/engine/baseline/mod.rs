mod math;
mod roofline;

pub use math::{
    ACTIVATION_KV_BUFFER_GB, catalog_hybrid_state_bytes, catalog_state_pages,
    counterfactual_concurrency, kv_bytes_per_element, kv_max_concurrent_seqs, observed_state_pages,
    state_dtype_bytes, weight_gb,
};
pub use roofline::{
    CeilingEstimate, CostEstimate, CostSource, PhysicsBaseline, WeightDtypeSource, compute,
};
