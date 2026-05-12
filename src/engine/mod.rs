pub mod baseline;
mod rules;

use crate::context::AnalysisInput;

pub use baseline::{CeilingEstimate, PhysicsBaseline, WeightDtypeSource};
pub use rules::*;

#[derive(Debug, Clone)]
pub struct Report {
    pub baseline: Option<PhysicsBaseline>,
}

pub fn build_report(input: AnalysisInput<'_>) -> Report {
    Report {
        baseline: baseline::compute(&input),
    }
}
