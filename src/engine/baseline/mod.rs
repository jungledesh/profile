mod math;
mod roofline;

pub use math::{ACTIVATION_KV_BUFFER_GB, kv_bytes_per_element, kv_max_concurrent_seqs};
pub use roofline::{
    CeilingEstimate, CostEstimate, CostSource, PhysicsBaseline, WeightDtypeSource, compute,
};
