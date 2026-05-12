use crate::engine::{IssueGroup, PhysicsBaseline};

use super::types::{InferredObjective, Objective};

pub fn infer_objective(
    groups: &[IssueGroup],
    baseline: Option<&PhysicsBaseline>,
) -> InferredObjective {
    if let Some(g) = groups.first() {
        match g.primary.rule_name {
            "under_batching" => InferredObjective {
                objective: Objective::MaxThroughput,
                reason: format!(
                    "GPU at {:.0}% of ceiling — under-batching detected",
                    baseline.and_then(|b| b.efficiency_pct).unwrap_or(0.0)
                ),
            },
            "kv_cache_pressure" => InferredObjective {
                objective: Objective::MinLatency,
                reason: "KV cache near capacity — latency at risk".to_string(),
            },
            "low_prefix_reuse" => InferredObjective {
                objective: Objective::MinLatency,
                reason: "Low prefix cache hit rate — TTFT elevated".to_string(),
            },
            "parallelism_mismatch" => InferredObjective {
                objective: Objective::MaxThroughput,
                reason: "Model exceeds single-GPU VRAM — parallelism needed".to_string(),
            },
            _ => efficiency_default(baseline),
        }
    } else {
        efficiency_default(baseline)
    }
}

fn efficiency_default(baseline: Option<&PhysicsBaseline>) -> InferredObjective {
    let eff = baseline.and_then(|b| b.efficiency_pct).unwrap_or(100.0);
    if eff < 70.0 {
        InferredObjective {
            objective: Objective::MaxThroughput,
            reason: format!("GPU at {eff:.0}% of ceiling — throughput headroom available"),
        }
    } else {
        InferredObjective {
            objective: Objective::MinLatency,
            reason: "Server near ceiling — latency optimization most impactful".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{IssueGroup, Recommendation};

    fn group(rule_name: &'static str) -> IssueGroup {
        IssueGroup {
            primary: Recommendation {
                rule_name,
                impact: 1,
                confidence: 1.0,
                action: String::new(),
                expected_impact: String::new(),
                display_lines: Vec::new(),
            },
            secondary: Vec::new(),
        }
    }

    fn baseline_eff(efficiency_pct: Option<f64>) -> PhysicsBaseline {
        PhysicsBaseline {
            decode: crate::engine::CeilingEstimate {
                lower: 1.0,
                expected: 1.0,
                upper: 1.0,
            },
            prefill: None,
            efficiency_pct,
            headroom_pct: efficiency_pct.map(|raw| 100.0 - raw.min(100.0)),
            weight_dtype_source: crate::engine::WeightDtypeSource::Fallback,
            weight_gb: 0.0,
            kv_headroom_gb: None,
            tpot_floor_ms: 1.0,
            prefill_latency_floor_ms: None,
            ridge_batch_size: 1.0,
        }
    }

    #[test]
    fn under_batching_maps_to_max_throughput() {
        let i = infer_objective(&[group("under_batching")], Some(&baseline_eff(Some(40.0))));
        assert_eq!(i.objective, Objective::MaxThroughput);
        assert!(i.reason.contains("40%"));
    }

    #[test]
    fn kv_cache_pressure_maps_to_min_latency() {
        let i = infer_objective(&[group("kv_cache_pressure")], None);
        assert_eq!(i.objective, Objective::MinLatency);
    }

    #[test]
    fn low_prefix_reuse_maps_to_min_latency() {
        let i = infer_objective(&[group("low_prefix_reuse")], None);
        assert_eq!(i.objective, Objective::MinLatency);
    }

    #[test]
    fn parallelism_mismatch_maps_to_max_throughput() {
        let i = infer_objective(&[group("parallelism_mismatch")], None);
        assert_eq!(i.objective, Objective::MaxThroughput);
    }

    #[test]
    fn unknown_rule_falls_back_to_efficiency_default() {
        let i = infer_objective(&[group("future_rule_xyz")], Some(&baseline_eff(Some(50.0))));
        assert_eq!(i.objective, Objective::MaxThroughput);
        assert!(i.reason.contains("50%"));
    }

    #[test]
    fn empty_groups_eff_below_70_max_throughput() {
        let i = infer_objective(&[], Some(&baseline_eff(Some(55.0))));
        assert_eq!(i.objective, Objective::MaxThroughput);
    }

    #[test]
    fn empty_groups_eff_at_or_above_70_min_latency() {
        let i = infer_objective(&[], Some(&baseline_eff(Some(70.0))));
        assert_eq!(i.objective, Objective::MinLatency);
        let i2 = infer_objective(&[], Some(&baseline_eff(Some(90.0))));
        assert_eq!(i2.objective, Objective::MinLatency);
    }
}
