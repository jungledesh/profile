use crate::engine::WeightDtypeSource;

use super::Recommendation;

/// Minimum buffer inside the vLLM-allocated block for activations and bare-minimum KV cache.
/// Source: vLLM memory profiling — weights filling 100% of the utilized block causes
/// immediate OOM on first forward pass. 3GB is a conservative lower bound.
const ACTIVATION_KV_BUFFER_GB: f64 = 3.0;

/// vLLM default gpu_memory_utilization when not explicitly set by operator.
const DEFAULT_GPU_MEMORY_UTILIZATION: f64 = 0.90;

fn min_tp(weight_gb: f64, vram_gb: f64, gpu_memory_utilization: f64) -> u32 {
    let usable = (vram_gb * gpu_memory_utilization) - ACTIVATION_KV_BUFFER_GB;
    if usable <= 0.0 {
        return 1;
    }
    (weight_gb / usable).ceil() as u32
}

fn confidence_for_source(weight_dtype_source: WeightDtypeSource) -> f64 {
    match weight_dtype_source {
        WeightDtypeSource::EnvVar => 0.95,
        WeightDtypeSource::KvCacheDtype => 0.90,
        WeightDtypeSource::Catalog => 0.90,
        WeightDtypeSource::Fallback => 0.60,
    }
}

fn confidence_label(confidence: f64) -> &'static str {
    if confidence >= 0.9 {
        "High"
    } else {
        "Medium"
    }
}

/// r4: weights exceed GPU VRAM budget (`kv_headroom_gb < 0`).
pub fn r4_recommendation(
    kv_headroom_gb: Option<f64>,
    tensor_parallel_size: Option<u32>,
    weight_gb: Option<f64>,
    vram_gb: Option<f64>,
    gpu_memory_utilization: Option<f64>,
    weight_dtype_source: WeightDtypeSource,
) -> Option<Recommendation> {
    let h = kv_headroom_gb?;
    if !h.is_finite() {
        return None;
    }
    if h >= 0.0 {
        return None;
    }
    let overflow = h.abs();
    let confidence = confidence_for_source(weight_dtype_source);
    let gpu_util = gpu_memory_utilization.unwrap_or(DEFAULT_GPU_MEMORY_UTILIZATION);
    let computed_min_tp = weight_gb.zip(vram_gb).map(|(w, v)| min_tp(w, v, gpu_util));

    let fix_line = match (tensor_parallel_size, computed_min_tp) {
        (Some(current), Some(needed)) if current < needed => format!(
            "    • Increase --tensor-parallel-size to at least {needed} (currently {current})"
        ),
        (Some(current), Some(needed)) if current >= needed => format!(
            "    • TP={current} should fit weights, but KV cache or activation memory is exhausted — reduce --max-model-len or lower --gpu-memory-utilization"
        ),
        (None, Some(needed)) => {
            format!("    • Set --tensor-parallel-size to at least {needed}")
        }
        _ => format!(
            "    • Increase --tensor-parallel-size (weights overflow by ~{overflow:.0}GB)"
        ),
    };

    let short_action = match computed_min_tp {
        Some(n) => format!("set --tensor-parallel-size to at least {n}"),
        None => format!("increase --tensor-parallel-size (weights overflow by ~{overflow:.0}GB)"),
    };

    Some(Recommendation {
        rule_name: "oom_risk",
        impact: 5,
        confidence,
        action: format!(
            "Model weights exceed GPU VRAM by {overflow:.0}GB — server will OOM without tensor parallelism"
        ),
        short_action,
        expected_impact: "Model fits in memory; eliminates OOM risk".to_string(),
        display_lines: vec![
            "[!] OOM Risk".to_string(),
            String::new(),
            "  Cause:".to_string(),
            format!("    • Model weights exceed GPU VRAM by ~{overflow:.0}GB"),
            String::new(),
            "  Fix:".to_string(),
            fix_line,
            String::new(),
            "  Expected: Model fits in VRAM; eliminates OOM risk.".to_string(),
            format!(
                "  Confidence: {}",
                confidence_label(confidence)
            ),
        ],
    })
}

/// Fires when R4 cannot evaluate due to missing VRAM or model params.
/// No traffic gate — OOM risk exists regardless of current load.
pub fn r4_advisory(
    kv_headroom_gb: Option<f64>,
    vram_gb: Option<f64>,
    weight_gb: Option<f64>,
) -> Option<Vec<String>> {
    if kv_headroom_gb.is_some() {
        return None;
    }
    if vram_gb.is_none() {
        return Some(vec![
            "[i] OOM Risk: GPU VRAM unavailable (NVML not reporting). Cannot verify model fits in memory.".to_string(),
        ]);
    }
    if weight_gb.is_none() {
        return Some(vec![
            "[i] OOM Risk: Model parameters unknown. Cannot verify model fits in memory. Add model to catalog or pass --model-params.".to_string(),
        ]);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn min_tp_h200_140gb_model() {
        // 80GB × 0.9 - 3GB buffer = 69GB usable; 140GB weights → TP=3
        assert_eq!(min_tp(140.0, 80.0, 0.9), 3);
    }

    #[test]
    fn min_tp_degenerate_usable_returns_one() {
        assert_eq!(min_tp(100.0, 2.0, 0.9), 1);
    }

    #[test]
    fn fix_shows_exact_tp_when_current_tp_known_and_insufficient() {
        let r = r4_recommendation(
            Some(-12.5),
            Some(1),
            Some(140.0),
            Some(80.0),
            Some(0.9),
            WeightDtypeSource::EnvVar,
        )
        .expect("fired");
        let text = r.display_lines.join("\n");
        assert!(text.contains("to at least 3 (currently 1)"));
    }

    #[test]
    fn fix_pivots_when_current_tp_meets_min_but_still_overflowing() {
        let r = r4_recommendation(
            Some(-8.0),
            Some(4),
            Some(140.0),
            Some(80.0),
            Some(0.9),
            WeightDtypeSource::EnvVar,
        )
        .expect("fired");
        let text = r.display_lines.join("\n");
        assert!(text.contains("TP=4 should fit weights"));
        assert!(text.contains("KV cache or activation memory is exhausted"));
    }

    #[test]
    fn confidence_low_when_dtype_fallback() {
        let r = r4_recommendation(
            Some(-12.5),
            None,
            Some(140.0),
            Some(80.0),
            Some(0.9),
            WeightDtypeSource::Fallback,
        )
        .expect("fired");
        assert!((r.confidence - 0.60).abs() < 1e-9);
        assert!(r.display_lines.join("\n").contains("Confidence: Medium"));
    }

    #[test]
    fn confidence_high_when_dtype_from_env() {
        let r = r4_recommendation(
            Some(-12.5),
            None,
            Some(140.0),
            Some(80.0),
            Some(0.9),
            WeightDtypeSource::EnvVar,
        )
        .expect("fired");
        assert!((r.confidence - 0.95).abs() < 1e-9);
        assert!(r.display_lines.join("\n").contains("Confidence: High"));
    }

    #[test]
    fn advisory_fires_when_vram_gb_missing() {
        let adv = r4_advisory(None, None, Some(70.0)).expect("advisory");
        assert!(adv[0].contains("GPU VRAM unavailable"));
    }

    #[test]
    fn advisory_fires_when_weight_gb_missing() {
        let adv = r4_advisory(None, Some(80.0), None).expect("advisory");
        assert!(adv[0].contains("Model parameters unknown"));
    }

    #[test]
    fn advisory_absent_when_kv_headroom_computed() {
        assert!(r4_advisory(Some(-5.0), None, None).is_none());
        assert!(r4_advisory(Some(4.0), None, Some(70.0)).is_none());
    }

    #[test]
    fn r4_fires_when_negative_headroom() {
        let r = r4_recommendation(
            Some(-12.5),
            None,
            None,
            None,
            None,
            WeightDtypeSource::EnvVar,
        )
        .expect("fired");
        assert_eq!(r.rule_name, "oom_risk");
        assert_eq!(r.impact, 5);
        assert!((r.confidence - 0.95).abs() < 1e-9);
        let text = r.display_lines.join("\n");
        assert!(text.contains("[!] OOM Risk"));
        assert!(text.contains("    • Model weights exceed GPU VRAM by ~12GB"));
        assert!(text.contains("weights overflow by ~12GB"));
        assert!(r
            .short_action
            .contains("increase --tensor-parallel-size (weights overflow by ~12GB)"));
    }

    #[test]
    fn short_action_includes_min_tp_when_weight_and_vram_known() {
        let r = r4_recommendation(
            Some(-12.5),
            Some(1),
            Some(140.0),
            Some(80.0),
            Some(0.9),
            WeightDtypeSource::EnvVar,
        )
        .expect("fired");
        assert_eq!(r.short_action, "set --tensor-parallel-size to at least 3");
    }

    #[test]
    fn r4_suppressed_when_headroom_non_negative() {
        assert!(
            r4_recommendation(Some(4.0), None, None, None, None, WeightDtypeSource::EnvVar,)
                .is_none()
        );
        assert!(
            r4_recommendation(Some(0.0), None, None, None, None, WeightDtypeSource::EnvVar,)
                .is_none()
        );
    }

    #[test]
    fn r4_suppressed_when_headroom_missing_or_non_finite() {
        assert!(
            r4_recommendation(None, None, None, None, None, WeightDtypeSource::EnvVar,).is_none()
        );
        assert!(r4_recommendation(
            Some(f64::NAN),
            None,
            None,
            None,
            None,
            WeightDtypeSource::EnvVar,
        )
        .is_none());
    }
}
