use crate::engine::PhysicsBaseline;

use super::types::{FeasibilityResult, Goal};

pub fn check_feasibility(goal: &Goal, baseline: Option<&PhysicsBaseline>) -> FeasibilityResult {
    let _ = goal;
    let Some(b) = baseline else {
        return FeasibilityResult::Reachable;
    };
    let headroom = b.headroom_pct.unwrap_or(100.0);
    if headroom < 5.0 {
        FeasibilityResult::AtCeiling {
            headroom_pct: headroom,
        }
    } else {
        FeasibilityResult::Reachable
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::goal::types::{Goal, Objective};
    use crate::engine::{CeilingEstimate, WeightDtypeSource};

    fn baseline(headroom_pct: Option<f64>) -> PhysicsBaseline {
        PhysicsBaseline {
            decode: CeilingEstimate {
                lower: 1.0,
                expected: 1.0,
                upper: 1.0,
            },
            prefill: None,
            efficiency_pct: None,
            headroom_pct,
            weight_dtype_source: WeightDtypeSource::Fallback,
            weight_gb: 0.0,
            kv_headroom_gb: None,
            tpot_floor_ms: 1.0,
            prefill_latency_floor_ms: None,
            ridge_batch_size: 1.0,
        }
    }

    fn goal_throughput() -> Goal {
        Goal {
            objective: Objective::MaxThroughput,
        }
    }

    #[test]
    fn headroom_below_five_at_ceiling() {
        match check_feasibility(&goal_throughput(), Some(&baseline(Some(3.0)))) {
            FeasibilityResult::AtCeiling { headroom_pct } => {
                assert!((headroom_pct - 3.0).abs() < 1e-9);
            }
            FeasibilityResult::Reachable => panic!("expected AtCeiling"),
        }
    }

    #[test]
    fn headroom_at_or_above_five_reachable() {
        assert!(matches!(
            check_feasibility(&goal_throughput(), Some(&baseline(Some(5.0)))),
            FeasibilityResult::Reachable
        ));
        assert!(matches!(
            check_feasibility(&goal_throughput(), Some(&baseline(Some(40.0)))),
            FeasibilityResult::Reachable
        ));
    }

    #[test]
    fn baseline_none_reachable() {
        assert!(matches!(
            check_feasibility(&goal_throughput(), None),
            FeasibilityResult::Reachable
        ));
    }
}
