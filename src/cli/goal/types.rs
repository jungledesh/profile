#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Objective {
    MaxThroughput,
    MinLatency,
    ReduceCost,
}

impl Objective {
    pub fn label(&self) -> &'static str {
        match self {
            Objective::MaxThroughput => "Maximize throughput",
            Objective::MinLatency => "Minimize latency",
            Objective::ReduceCost => "Reduce cost",
        }
    }
}

#[derive(Debug, Clone)]
pub struct InferredObjective {
    pub objective: Objective,
    /// Shown to user, e.g. "GPU at 34% of ceiling — under-batching detected"
    pub reason: String,
}

#[derive(Debug, Clone)]
pub struct Goal {
    pub objective: Objective,
}

#[derive(Debug, Clone)]
pub enum FeasibilityResult {
    Reachable,
    AtCeiling { headroom_pct: f64 },
}
