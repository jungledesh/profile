use crate::collectors::config::DEFAULT_GPU_MEMORY_UTILIZATION;
use crate::engine::WeightDtypeSource;
use crate::engine::baseline::ACTIVATION_KV_BUFFER_GB;

use super::Recommendation;
use super::rule_names;

fn min_tp(weight_gb: f64, vram_gb: f64, gpu_memory_utilization: f64) -> Option<u32> {
    let usable = (vram_gb * gpu_memory_utilization) - ACTIVATION_KV_BUFFER_GB;
    if usable <= 0.0 {
        return None; // GPU can't hold even the activation buffer
    }
    Some((weight_gb / usable).ceil() as u32)
}

fn confidence_for_source(weight_dtype_source: WeightDtypeSource) -> f64 {
    match weight_dtype_source {
        WeightDtypeSource::VllmInfoQuantization => 0.90,
        WeightDtypeSource::EnvVarQuantization => 0.95,
        WeightDtypeSource::EnvVar => 0.95,
        WeightDtypeSource::VllmConfig => 0.90,
        WeightDtypeSource::VllmInfoEndpoint => 0.90,
        WeightDtypeSource::Catalog => 0.90,
        WeightDtypeSource::Fallback => 0.60,
    }
}

fn confidence_label(confidence: f64) -> &'static str {
    if confidence >= 0.9 { "High" } else { "Medium" }
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
    r4_recommendation_with_request_floor(
        kv_headroom_gb,
        tensor_parallel_size,
        weight_gb,
        vram_gb,
        gpu_memory_utilization,
        weight_dtype_source,
        None,
    )
}

pub(super) fn r4_recommendation_with_request_floor(
    kv_headroom_gb: Option<f64>,
    tensor_parallel_size: Option<u32>,
    weight_gb: Option<f64>,
    vram_gb: Option<f64>,
    gpu_memory_utilization: Option<f64>,
    weight_dtype_source: WeightDtypeSource,
    request_bytes: Option<u64>,
) -> Option<Recommendation> {
    let h = kv_headroom_gb?;
    if !h.is_finite() {
        return None;
    }
    if weight_dtype_source == WeightDtypeSource::Fallback {
        return None;
    }
    if h >= 0.0 {
        let request_bytes = request_bytes?;
        let free_bytes = (h * 1e9) as u64;
        if free_bytes >= request_bytes {
            return None;
        }
        let confidence = confidence_for_source(weight_dtype_source);
        let gpu_util = gpu_memory_utilization.unwrap_or(DEFAULT_GPU_MEMORY_UTILIZATION);
        let can_raise_utilization = vram_gb.is_some_and(|vram| {
            gpu_util < 1.0 && (h + vram * (1.0 - gpu_util)) * 1e9 >= request_bytes as f64
        });
        let fix_line = if can_raise_utilization {
            "      • Raise --gpu-memory-utilization enough to hold one worst-case request"
                .to_string()
        } else {
            "      • Use a smaller model or a GPU with more VRAM".to_string()
        };
        return Some(Recommendation {
            rule_name: rule_names::OOM_RISK,
            layer: 2,
            impact: 5,
            confidence,
            display_lines: vec![
                "[!] OOM Risk".to_string(),
                String::new(),
                "    Cause:".to_string(),
                "      • Free VRAM after weights and the activation allowance cannot hold a single request's KV + state."
                    .to_string(),
                format!(
                    "      • {:.1}GB free; one worst-case request needs {:.1}GB (est).",
                    h,
                    request_bytes as f64 / 1e9
                ),
                String::new(),
                "    Fix:".to_string(),
                fix_line,
                String::new(),
                "    Expected: Model and one request fit in VRAM.".to_string(),
                format!("    Confidence: {}", confidence_label(confidence)),
            ],
        });
    }
    let overflow = h.abs();
    let confidence = confidence_for_source(weight_dtype_source);
    let gpu_util = gpu_memory_utilization.unwrap_or(DEFAULT_GPU_MEMORY_UTILIZATION);
    let unrunnable = weight_gb
        .zip(vram_gb)
        .is_some_and(|(_, v)| (v * gpu_util) - ACTIVATION_KV_BUFFER_GB <= 0.0);
    let computed_min_tp = weight_gb
        .zip(vram_gb)
        .and_then(|(w, v)| min_tp(w, v, gpu_util));

    let fix_line = if unrunnable {
        "      • This GPU has insufficient VRAM for activation memory alone. This model cannot run on this hardware.".to_string()
    } else {
        match (tensor_parallel_size, computed_min_tp) {
            (Some(current), Some(needed)) if current < needed => format!(
                "      • Increase --tensor-parallel-size to at least {needed} (currently {current})"
            ),
            (Some(current), Some(needed)) if current >= needed => format!(
                "      • TP={current} should fit weights, but KV cache or activation memory is exhausted. Reduce --max-model-len or lower --gpu-memory-utilization"
            ),
            (None, Some(needed)) => {
                format!("      • Set --tensor-parallel-size to at least {needed}")
            }
            _ => format!(
                "      • Increase --tensor-parallel-size (weights overflow by ~{overflow:.0}GB)"
            ),
        }
    };

    Some(Recommendation {
        rule_name: rule_names::OOM_RISK,
        layer: 2,
        impact: 5,
        confidence,
        display_lines: vec![
            "[!] OOM Risk".to_string(),
            String::new(),
            "    Cause:".to_string(),
            format!("      • Model weights exceed GPU VRAM by ~{overflow:.0}GB"),
            "      • Server may OOM without tensor parallelism.".to_string(),
            String::new(),
            "    Fix:".to_string(),
            fix_line,
            String::new(),
            "    Expected: Model fits in VRAM; eliminates OOM risk.".to_string(),
            format!("    Confidence: {}", confidence_label(confidence)),
        ],
    })
}

/// Informational note when R4 cannot produce a confident recommendation.
/// Covers: missing VRAM, missing model params, or negative headroom with uncertain dtype (Fallback).
/// No traffic gate, no DAG participation.
pub fn r4_advisory(
    kv_headroom_gb: Option<f64>,
    vram_gb: Option<f64>,
    weight_gb: Option<f64>,
    weight_dtype_source: WeightDtypeSource,
    tensor_parallel_size: Option<u32>,
) -> Option<Vec<String>> {
    if let Some(h) = kv_headroom_gb {
        if h < 0.0 && weight_dtype_source == WeightDtypeSource::Fallback {
            let w = weight_gb
                .map(|g| format!("{g:.0}GB"))
                .unwrap_or_else(|| "?".to_string());
            let v = vram_gb
                .map(|g| format!("{g:.0}GB"))
                .unwrap_or_else(|| "?".to_string());
            let tp = tensor_parallel_size
                .map(|t| format!("TP={t}"))
                .unwrap_or_else(|| "TP=?".to_string());
            return Some(vec![
                format!(
                    "[i] OOM Risk: Negative headroom ({h:.1}GB) computed assuming bf16. Weights {w}, VRAM {v}, {tp}."
                ),
                "    If quantized, ignore this. Otherwise verify dtype or increase TP.".to_string(),
            ]);
        }
        return None;
    }
    if vram_gb.is_none() {
        return Some(vec![
            "[i] OOM Risk: GPU VRAM unavailable (driver not reporting). Cannot verify model fits in memory.".to_string(),
        ]);
    }
    if weight_gb.is_none() {
        return Some(vec![
            "[i] OOM Risk: Model parameters unknown. Cannot verify model fits in memory. Add model to catalog.".to_string(),
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
        assert_eq!(min_tp(140.0, 80.0, 0.9), Some(3));
    }

    #[test]
    fn min_tp_returns_none_when_usable_non_positive() {
        assert!(min_tp(100.0, 2.0, 0.9).is_none());
    }

    #[test]
    fn min_tp_returns_none_when_vram_too_small_for_activation_buffer() {
        // 3GB GPU × 0.9 util = 2.7GB < 3.0GB buffer → unrunnable
        assert!(min_tp(140.0, 3.0, 0.9).is_none());
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
    fn request_floor_fires_only_when_one_request_exceeds_free_vram() {
        let with_whiteboard = r4_recommendation_with_request_floor(
            Some(0.05),
            Some(1),
            Some(20.0),
            Some(24.0),
            Some(0.9),
            WeightDtypeSource::Catalog,
            Some(60_000_000),
        )
        .expect("one request should not fit");
        let text = with_whiteboard.display_lines.join("\n");
        assert!(text.contains("cannot hold a single request's KV + state"));
        assert!(text.contains("Raise --gpu-memory-utilization"));
        assert!(!text.contains("tensor-parallel"));

        let without_whiteboard = r4_recommendation_with_request_floor(
            Some(0.05),
            Some(1),
            Some(20.0),
            Some(24.0),
            Some(0.9),
            WeightDtypeSource::Catalog,
            Some(40_000_000),
        );
        assert!(without_whiteboard.is_none());
    }

    #[test]
    fn advisory_fires_on_fallback_negative_headroom() {
        let adv = r4_advisory(
            Some(-5.0),
            Some(80.0),
            Some(140.0),
            WeightDtypeSource::Fallback,
            Some(2),
        )
        .expect("advisory");
        let text = adv.join("\n");
        assert!(text.contains("assuming bf16"));
        assert!(text.contains("Weights 140GB, VRAM 80GB, TP=2."));
    }

    #[test]
    fn confidence_high_when_dtype_from_vllm_config() {
        let r = r4_recommendation(
            Some(-12.5),
            None,
            Some(140.0),
            Some(80.0),
            Some(0.9),
            WeightDtypeSource::VllmConfig,
        )
        .expect("fired");
        assert!((r.confidence - 0.90).abs() < 1e-9);
        assert!(r.display_lines.join("\n").contains("Confidence: High"));
    }

    #[test]
    fn confidence_high_when_dtype_from_vllm_info_endpoint() {
        let r = r4_recommendation(
            Some(-12.5),
            None,
            Some(140.0),
            Some(80.0),
            Some(0.9),
            WeightDtypeSource::VllmInfoEndpoint,
        )
        .expect("fired");
        assert!((r.confidence - 0.90).abs() < 1e-9);
        assert!(r.display_lines.join("\n").contains("Confidence: High"));
    }

    #[test]
    fn confidence_high_when_dtype_from_vllm_info_quantization() {
        let r = r4_recommendation(
            Some(-12.5),
            None,
            Some(140.0),
            Some(80.0),
            Some(0.9),
            WeightDtypeSource::VllmInfoQuantization,
        )
        .expect("fired");
        assert!((r.confidence - 0.90).abs() < 1e-9);
        assert!(r.display_lines.join("\n").contains("Confidence: High"));
    }

    #[test]
    fn confidence_high_when_dtype_from_env_quantization() {
        let r = r4_recommendation(
            Some(-12.5),
            None,
            Some(140.0),
            Some(80.0),
            Some(0.9),
            WeightDtypeSource::EnvVarQuantization,
        )
        .expect("fired");
        assert!((r.confidence - 0.95).abs() < 1e-9);
        assert!(r.display_lines.join("\n").contains("Confidence: High"));
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
        let adv =
            r4_advisory(None, None, Some(70.0), WeightDtypeSource::EnvVar, None).expect("advisory");
        assert!(adv[0].contains("GPU VRAM unavailable"));
    }

    #[test]
    fn advisory_fires_when_weight_gb_missing() {
        let adv =
            r4_advisory(None, Some(80.0), None, WeightDtypeSource::EnvVar, None).expect("advisory");
        assert!(adv[0].contains("Model parameters unknown"));
    }

    #[test]
    fn advisory_absent_when_kv_headroom_computed() {
        assert!(r4_advisory(Some(-5.0), None, None, WeightDtypeSource::EnvVar, None).is_none());
        assert!(
            r4_advisory(Some(4.0), None, Some(70.0), WeightDtypeSource::EnvVar, None).is_none()
        );
    }

    #[test]
    fn advisory_absent_when_fallback_but_positive_headroom() {
        assert!(
            r4_advisory(
                Some(4.0),
                Some(80.0),
                Some(140.0),
                WeightDtypeSource::Fallback,
                Some(2),
            )
            .is_none()
        );
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
        assert_eq!(r.rule_name, rule_names::OOM_RISK);
        assert_eq!(r.impact, 5);
        assert!((r.confidence - 0.95).abs() < 1e-9);
        let text = r.display_lines.join("\n");
        assert!(text.contains("[!] OOM Risk"));
        assert!(text.contains("      • Model weights exceed GPU VRAM by ~12GB"));
        assert!(text.contains("Server may OOM without tensor parallelism"));
        assert!(text.contains("weights overflow by ~12GB"));
    }

    #[test]
    fn r4_request_floor_declines_on_fallback_dtype() {
        // Same numbers as request_floor_fires_on_known_dtype; only dtype source differs.
        assert!(
            r4_recommendation_with_request_floor(
                Some(0.05),
                Some(1),
                Some(20.0),
                Some(24.0),
                Some(0.9),
                WeightDtypeSource::Fallback,
                Some(60_000_000),
            )
            .is_none()
        );
    }

    #[test]
    fn r4_request_floor_fires_on_known_dtype() {
        let r = r4_recommendation_with_request_floor(
            Some(0.05),
            Some(1),
            Some(20.0),
            Some(24.0),
            Some(0.9),
            WeightDtypeSource::EnvVar,
            Some(60_000_000),
        )
        .expect("known dtype should fire on identical numbers");
        let text = r.display_lines.join("\n");
        assert!(text.contains("cannot hold a single request's KV + state"));
        assert!(text.contains("Raise --gpu-memory-utilization"));
    }

    #[test]
    fn r4_recommendation_none_when_fallback_negative() {
        assert!(
            r4_recommendation(
                Some(-12.5),
                None,
                Some(140.0),
                Some(80.0),
                Some(0.9),
                WeightDtypeSource::Fallback,
            )
            .is_none()
        );
    }

    #[test]
    fn r4_recommendation_fires_when_non_fallback_negative() {
        assert!(
            r4_recommendation(
                Some(-12.5),
                None,
                Some(140.0),
                Some(80.0),
                Some(0.9),
                WeightDtypeSource::EnvVar,
            )
            .is_some()
        );
    }

    #[test]
    fn display_includes_min_tp_when_weight_and_vram_known() {
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
        assert!(text.contains("Increase --tensor-parallel-size to at least 3"));
        assert!(text.contains("Server may OOM without tensor parallelism"));
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
        assert!(
            r4_recommendation(
                Some(f64::NAN),
                None,
                None,
                None,
                None,
                WeightDtypeSource::EnvVar,
            )
            .is_none()
        );
    }
}
