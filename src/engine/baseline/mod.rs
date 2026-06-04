mod math;
mod roofline;

pub use roofline::{
    compute, CeilingEstimate, CostEstimate, CostSource, PhysicsBaseline, WeightDtypeSource,
};
