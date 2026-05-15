use super::Recommendation;

/// r4: weights exceed GPU VRAM budget (`kv_headroom_gb < 0`).
pub fn r4_recommendation(
    kv_headroom_gb: Option<f64>,
    _tensor_parallel_size: Option<u32>,
) -> Option<Recommendation> {
    let h = kv_headroom_gb?;
    if !h.is_finite() {
        return None;
    }
    if h >= 0.0 {
        return None;
    }
    let overflow = h.abs();
    Some(Recommendation {
        rule_name: "parallelism_mismatch",
        impact: 5,
        confidence: 0.95,
        action: format!(
            "Model weights exceed GPU VRAM by {:.0}GB — increase --tensor-parallel-size",
            overflow
        ),
        expected_impact: "Model fits in memory; eliminates OOM risk".to_string(),
        display_lines: vec![
            "Parallelism Mismatch".to_string(),
            format!(
                "Cause: KV headroom {:.1}GB (threshold: ≥0GB); model weights exceed GPU VRAM by ~{:.0}GB",
                h, overflow
            ),
            String::new(),
            "Recommendation:".to_string(),
            format!(
                "  • Increase --tensor-parallel-size (weights overflow by ~{:.0}GB)",
                overflow
            ),
            String::new(),
            "Expected: Model fits in VRAM; eliminates OOM risk".to_string(),
            "Confidence: High".to_string(),
        ],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn r4_fires_when_negative_headroom() {
        let r = r4_recommendation(Some(-12.5), None).expect("fired");
        assert_eq!(r.rule_name, "parallelism_mismatch");
        assert_eq!(r.impact, 5);
        assert!((r.confidence - 0.95).abs() < 1e-9);
        assert!(r.action.contains("tensor-parallel-size"));
        let cause = r.display_lines[1].as_str();
        assert!(cause.contains("KV headroom"));
        assert!(cause.contains("threshold: ≥0GB"));
        assert!(!cause.contains("TP "));
    }

    #[test]
    fn r4_fires_when_negative_headroom_even_with_tp_configured() {
        let r = r4_recommendation(Some(-8.0), Some(2)).expect("fired");
        assert_eq!(r.rule_name, "parallelism_mismatch");
    }

    #[test]
    fn r4_suppressed_when_headroom_non_negative() {
        assert!(r4_recommendation(Some(4.0), None).is_none());
        assert!(r4_recommendation(Some(0.0), None).is_none());
    }

    #[test]
    fn r4_suppressed_when_headroom_missing_or_non_finite() {
        assert!(r4_recommendation(None, None).is_none());
        assert!(r4_recommendation(Some(f64::NAN), None).is_none());
    }
}
