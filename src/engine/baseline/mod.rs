mod math;
mod roofline;

pub use math::ACTIVATION_KV_BUFFER_GB;
pub use roofline::{
    compute, CeilingEstimate, CostEstimate, CostSource, PhysicsBaseline, WeightDtypeSource,
};
