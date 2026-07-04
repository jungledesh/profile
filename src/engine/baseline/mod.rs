mod math;
mod roofline;

pub use math::{ACTIVATION_KV_BUFFER_GB, kv_bytes_per_element, kv_max_concurrent_seqs, weight_gb};
pub use roofline::{
    CeilingEstimate, CostEstimate, CostSource, PhysicsBaseline, WeightDtypeSource, compute,
    prefill_time_fraction_from_snapshot,
};
