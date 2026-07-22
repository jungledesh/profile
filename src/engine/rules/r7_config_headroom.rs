use crate::collectors::RawSnapshot;

use super::{BindingWall, KvBoundSource, RecommendedSeqs, recommended_seqs, resolve_kv_bound};

/// Fire when max_num_seqs is less than 90% of hardware-recommended capacity.
const CONFIG_HEADROOM_RATIO: f64 = 0.90;

/// Server must be working, not idle.
const OCCUPANCY_FLOOR: f64 = 0.50;

/// No backlog; R5 owns that signal. Below 1.0 is gauge noise, not a real queue.
const WAITING_CEILING: f64 = 1.0;

#[derive(Debug, Clone, PartialEq)]
pub struct ConfigHeadroomDetail {
    pub max_num_seqs: u32,
    pub recommended_seqs: u32,
    pub ridge_batch_size: Option<f64>,
    pub occupancy_pct: f64,
    pub running: f64,
}

/// R7 confidence keyed to the binding wall's source (Decided #3: Observed > derived
/// > empirical). Ridge-bound (no memory source) is a physics ceiling: High.
pub(super) fn r7_confidence(rec: Option<&RecommendedSeqs>) -> f64 {
    match rec {
        None => 0.6,
        Some(r) if r.empirical => 0.5,
        Some(r) => match r.source {
            Some(KvBoundSource::Derived | KvBoundSource::DerivedHybrid) => 0.6,
            // Observed memory or ridge (source None): firm ceiling.
            _ => 0.8,
        },
    }
}

/// Per-window fire: resolve the KV bound Observed else derived else this window's
/// empirical, then take the two-wall min with ridge via the shared helper. The
/// displayed recommendation is overridden at report time with the run-level
/// resolution so R5 and R7 print one number.
pub fn rule7_config_headroom(
    snapshot: &RawSnapshot,
    config_max_num_seqs: Option<u32>,
    ridge_batch_size: Option<f64>,
    derived_kv: Option<u32>,
    is_hybrid: bool,
) -> Option<ConfigHeadroomDetail> {
    let max_n = snapshot
        .vllm
        .max_num_seqs
        .or(config_max_num_seqs)
        .filter(|&n| n > 0)?;
    let run = snapshot
        .vllm
        .num_requests_running
        .filter(|v| v.is_finite() && *v > 0.0)?;
    let wait = snapshot
        .vllm
        .num_requests_waiting
        .filter(|v| v.is_finite())?;

    let (kv_bound, kv_source) = resolve_kv_bound(
        snapshot.vllm.cache_config.kv_cache_max_concurrency,
        derived_kv,
        is_hybrid,
        Some(run),
        snapshot.vllm.kv_cache_usage_perc,
    );
    let rec = recommended_seqs(ridge_batch_size, kv_bound, kv_source, Some(max_n))?;
    if f64::from(max_n) >= f64::from(rec.target) * CONFIG_HEADROOM_RATIO {
        return None;
    }

    let occupancy = run / f64::from(max_n);
    if occupancy < OCCUPANCY_FLOOR {
        return None;
    }
    if wait > WAITING_CEILING {
        return None;
    }

    Some(ConfigHeadroomDetail {
        max_num_seqs: max_n,
        recommended_seqs: rec.target,
        ridge_batch_size: ridge_batch_size.filter(|r| r.is_finite() && *r > 0.0),
        occupancy_pct: occupancy * 100.0,
        running: run,
    })
}

fn median_u32(mut values: Vec<u32>) -> Option<u32> {
    if values.is_empty() {
        return None;
    }
    values.sort_unstable();
    Some(values[values.len() / 2])
}

pub(super) fn aggregate_r7_detail(details: &[ConfigHeadroomDetail]) -> ConfigHeadroomDetail {
    debug_assert!(
        !details.is_empty(),
        "aggregate_r7_detail called with empty details"
    );
    let n = details.len() as f64;
    let occupancy_pct = details.iter().map(|d| d.occupancy_pct).sum::<f64>() / n;
    let running = details.iter().map(|d| d.running).sum::<f64>() / n;
    // occupancy_pct and running: mean over fired windows. max_num_seqs and ridge are static.
    // recommended_seqs varies per window; median over fired windows. The displayed
    // value is overridden at report time with the run-level resolution.
    let first = &details[0];
    let recommended_seqs = median_u32(details.iter().map(|d| d.recommended_seqs).collect())
        .unwrap_or(first.recommended_seqs);
    ConfigHeadroomDetail {
        max_num_seqs: first.max_num_seqs,
        recommended_seqs,
        ridge_batch_size: first.ridge_batch_size,
        occupancy_pct,
        running,
    }
}

fn confidence_label(conf: f64) -> &'static str {
    if conf >= 0.8 {
        "High"
    } else if conf >= 0.6 {
        "Medium"
    } else {
        "Low"
    }
}

/// Recommended line binder label. Ridge number is on its own line (do not repeat).
/// Memory cap appears nowhere else in the block, so show it. Empirical: "(est)" only.
fn recommended_binder_suffix(rec: &RecommendedSeqs) -> String {
    if rec.empirical {
        return " (est)".to_string();
    }
    match rec.binder {
        BindingWall::Ridge | BindingWall::Config => " (bound by compute ridge)".to_string(),
        BindingWall::Memory { cap } => match rec.source {
            Some(KvBoundSource::Observed) => {
                format!(" (bound by memory limit {cap}, vLLM-reported)")
            }
            Some(KvBoundSource::Derived) | Some(KvBoundSource::DerivedHybrid) => {
                format!(" (at least {cap} worst-case requests fit (est))")
            }
            // Empirical handled above; None on a memory binder is a defensive fallback.
            Some(KvBoundSource::Empirical) | None => {
                format!(" (bound by memory limit {cap})")
            }
        },
    }
}

fn headroom_available_cause_line(ridge_resolved: bool, memory_resolved: bool) -> &'static str {
    match (ridge_resolved, memory_resolved) {
        (true, true) => "      Compute and KV memory headroom available.",
        (true, false) => "      Compute headroom available; memory bound unmeasured.",
        (false, true) => "      KV memory headroom available; compute ridge unknown.",
        (false, false) => "      Hardware capacity bounds unmeasured.",
    }
}

pub(super) fn format_config_headroom_window_issue(
    d: &ConfigHeadroomDetail,
    seen_pct: u32,
    confidence: f64,
    rec: Option<&RecommendedSeqs>,
    memory_bound_resolved: bool,
) -> Vec<String> {
    let cap_pct = (f64::from(d.max_num_seqs) / f64::from(d.recommended_seqs)) * 100.0;
    let ridge_str = d
        .ridge_batch_size
        .map(|r| format!("{r:.0}"))
        .unwrap_or_else(|| "-".to_string());
    let recommended = match rec {
        Some(r) => format!(
            "    Recommended   {}{}",
            d.recommended_seqs,
            recommended_binder_suffix(r)
        ),
        None => format!("    Recommended   {}", d.recommended_seqs),
    };
    let empirical = rec.is_some_and(|r| r.empirical);
    let mut lines = vec![
        "[!] Configured Batch Limit".to_string(),
        String::new(),
        format!("    Config max    {}", d.max_num_seqs),
        format!("    Ridge batch   {ridge_str}"),
        recommended,
        String::new(),
        "    Cause:".to_string(),
        format!(
            "      --max-num-seqs={} caps batch size at {:.0}% of hardware capacity.",
            d.max_num_seqs, cap_pct
        ),
        headroom_available_cause_line(d.ridge_batch_size.is_some(), memory_bound_resolved)
            .to_string(),
        String::new(),
        "    Fix:".to_string(),
        format!("      • Raise --max-num-seqs to {}.", d.recommended_seqs),
    ];
    if empirical {
        lines.push("      • Monitor KV cache when scaling up.".to_string());
    }
    lines.extend([
        String::new(),
        "    Expected: Higher decode throughput when traffic concurrency increases."
            .to_string(),
        "    Watch: Higher concurrency increases prefill load. Monitor decode latency after applying."
            .to_string(),
        format!("    Confidence: {}", confidence_label(confidence)),
    ]);
    super::with_seen_pct(lines, seen_pct)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collectors::{CacheConfigLabels, VllmRawMetrics};
    use std::time::SystemTime;

    fn snap(
        running: f64,
        max_num_seqs: u32,
        waiting: f64,
        kv_cache_usage_perc: Option<f64>,
    ) -> RawSnapshot {
        observed_snap(running, max_num_seqs, waiting, kv_cache_usage_perc, None)
    }

    fn observed_snap(
        running: f64,
        max_num_seqs: u32,
        waiting: f64,
        kv_cache_usage_perc: Option<f64>,
        kv_cache_max_concurrency: Option<f64>,
    ) -> RawSnapshot {
        let t = SystemTime::UNIX_EPOCH;
        RawSnapshot {
            gpu_observed_at: t,
            vllm_observed_at: t,
            timestamp: t,
            vllm: VllmRawMetrics {
                num_requests_running: Some(running),
                num_requests_waiting: Some(waiting),
                max_num_seqs: Some(max_num_seqs),
                kv_cache_usage_perc,
                generation_tokens_per_sec: Some(100.0),
                window_duration_secs: Some(2.0),
                cache_config: CacheConfigLabels {
                    kv_cache_max_concurrency,
                    ..Default::default()
                },
                ..Default::default()
            },
            gpus: vec![],
        }
    }

    #[test]
    fn ridge_binds_below_empirical_kv() {
        // ridge 153 < empirical ~606; ridge binds, no empirical cap, target 122.
        let s = snap(20.0, 32, 0.0, Some(3.3));
        let d = rule7_config_headroom(&s, None, Some(153.0), None, false).expect("fired");
        assert_eq!(d.recommended_seqs, 122);
        assert_eq!(d.ridge_batch_size, Some(153.0));
    }

    #[test]
    fn empirical_memory_binder_is_step_capped() {
        // empirical 62.5 < ridge 153; memory (empirical) binds; margin 50 capped to 2x20=40.
        let s = snap(50.0, 20, 0.0, Some(80.0));
        let d = rule7_config_headroom(&s, None, Some(153.0), None, false).expect("fired");
        assert_eq!(d.recommended_seqs, 40);
    }

    #[test]
    fn empirical_noise_floor_no_kv_bound() {
        // KV 0.5% is below the 1% noise floor; only ridge remains.
        let s = snap(20.0, 32, 0.0, Some(0.5));
        let d = rule7_config_headroom(&s, None, Some(153.0), None, false).expect("fired");
        assert_eq!(d.recommended_seqs, 122);
    }

    #[test]
    fn observed_first_uses_reported_concurrency() {
        // Observed 120 present, beats derived and empirical. min(153, 120) = 120, target 96.
        let s = observed_snap(20.0, 32, 0.0, Some(3.3), Some(120.0));
        let d = rule7_config_headroom(&s, None, Some(153.0), Some(240), false).expect("fired");
        assert_eq!(d.recommended_seqs, 96);
    }

    #[test]
    fn derived_used_when_observed_absent() {
        // No observed; derived 120 beats empirical. min(153, 120) = 120, target 96.
        let s = snap(20.0, 32, 0.0, Some(3.3));
        let d = rule7_config_headroom(&s, None, Some(153.0), Some(120), false).expect("fired");
        assert_eq!(d.recommended_seqs, 96);
    }

    #[test]
    fn kv_unavailable_uses_ridge_only() {
        let s = snap(20.0, 32, 0.0, None);
        let d = rule7_config_headroom(&s, None, Some(153.0), None, false).expect("fired");
        assert_eq!(d.recommended_seqs, 122);
    }

    #[test]
    fn fires_when_config_well_below_recommended() {
        let s = snap(20.0, 32, 0.0, Some(3.3));
        let d = rule7_config_headroom(&s, None, Some(153.0), None, false).expect("fired");
        assert_eq!(d.max_num_seqs, 32);
        assert_eq!(d.recommended_seqs, 122);
    }

    #[test]
    fn fires_when_config_at_80_pct_of_recommended() {
        // ridge 153 binds, target 122; 98 < 122 * 0.90 = 109.8 → fires.
        let s = snap(95.0, 98, 0.0, Some(15.6));
        let d = rule7_config_headroom(&s, None, Some(153.0), None, false).expect("fired");
        assert_eq!(d.recommended_seqs, 122);
    }

    #[test]
    fn does_not_fire_when_config_near_recommended() {
        let s = snap(20.0, 115, 0.0, Some(3.3));
        assert!(rule7_config_headroom(&s, None, Some(153.0), None, false).is_none());
    }

    #[test]
    fn does_not_fire_when_waiting_nonzero() {
        let s = snap(20.0, 32, 5.0, Some(3.3));
        assert!(rule7_config_headroom(&s, None, Some(153.0), None, false).is_none());
    }

    #[test]
    fn does_not_fire_when_occupancy_low() {
        let s = snap(2.0, 32, 0.0, Some(3.3));
        assert!(rule7_config_headroom(&s, None, Some(153.0), None, false).is_none());
    }

    #[test]
    fn does_not_fire_when_no_wall_known() {
        // Spec pin: no wall resolved → R7 does not fire.
        let s = snap(20.0, 32, 0.0, None);
        assert!(rule7_config_headroom(&s, None, None, None, false).is_none());
    }

    #[test]
    fn ridge_stored_as_none_when_only_kv_drives_recommendation() {
        // No ridge; empirical 62.5 binds, target min(50, 2x20=40) = 40.
        let s = snap(50.0, 20, 0.0, Some(80.0));
        let d = rule7_config_headroom(&s, None, None, None, false).expect("fired on kv only");
        assert!(d.ridge_batch_size.is_none());
        assert_eq!(d.recommended_seqs, 40);
    }

    #[test]
    fn fractional_waiting_below_ceiling_does_not_suppress() {
        let s = snap(20.0, 32, 0.5, Some(3.3));
        assert!(rule7_config_headroom(&s, None, Some(153.0), None, false).is_some());
    }

    #[test]
    fn confidence_keyed_to_source() {
        use crate::engine::rules::{BindingWall, KvBoundSource, RecommendedSeqs};
        let observed = RecommendedSeqs {
            target: 96,
            wall: 120.0,
            binder: BindingWall::Memory { cap: 120 },
            source: Some(KvBoundSource::Observed),
            empirical: false,
        };
        let derived = RecommendedSeqs {
            source: Some(KvBoundSource::Derived),
            ..observed
        };
        let empirical = RecommendedSeqs {
            source: Some(KvBoundSource::Empirical),
            empirical: true,
            ..observed
        };
        let ridge = RecommendedSeqs {
            binder: BindingWall::Ridge,
            source: None,
            ..observed
        };
        assert_eq!(r7_confidence(Some(&observed)), 0.8);
        assert_eq!(r7_confidence(Some(&derived)), 0.6);
        assert_eq!(r7_confidence(Some(&empirical)), 0.5);
        assert_eq!(r7_confidence(Some(&ridge)), 0.8);
        assert_eq!(r7_confidence(None), 0.6);
    }

    #[test]
    fn format_shows_dash_when_ridge_unavailable() {
        let d = ConfigHeadroomDetail {
            max_num_seqs: 32,
            recommended_seqs: 122,
            ridge_batch_size: None,
            occupancy_pct: 62.5,
            running: 20.0,
        };
        let text = format_config_headroom_window_issue(&d, 100, 0.6, None, false).join("\n");
        assert!(text.contains("Ridge batch   -"));
        assert!(text.contains("Hardware capacity bounds unmeasured."));
        assert!(!text.contains("Compute and KV memory headroom available."));
        assert!(text.contains("Confidence: Medium"));
        assert!(text.contains("Watch: Higher concurrency increases prefill load"));
    }

    #[test]
    fn format_ridge_binds_names_binder_without_repeating_number() {
        let d = ConfigHeadroomDetail {
            max_num_seqs: 32,
            recommended_seqs: 122,
            ridge_batch_size: Some(153.0),
            occupancy_pct: 62.5,
            running: 20.0,
        };
        let rec = RecommendedSeqs {
            target: 122,
            wall: 153.0,
            binder: BindingWall::Ridge,
            source: None,
            empirical: false,
        };
        let text = format_config_headroom_window_issue(&d, 100, 0.8, Some(&rec), false).join("\n");
        assert!(text.contains("Recommended   122 (bound by compute ridge)"));
        assert!(text.contains("Compute headroom available; memory bound unmeasured."));
        assert!(!text.contains("bound by compute ridge ~"));
        assert!(!text.contains("bound by compute ridge 153"));
        assert!(text.contains("Raise --max-num-seqs to 122."));
        assert!(!text.contains("at least"));
        assert!(!text.contains("Monitor KV cache"));
    }

    #[test]
    fn format_memory_observed_names_cap_vllm_reported() {
        let d = ConfigHeadroomDetail {
            max_num_seqs: 32,
            recommended_seqs: 96,
            ridge_batch_size: Some(153.0),
            occupancy_pct: 62.5,
            running: 20.0,
        };
        let rec = RecommendedSeqs {
            target: 96,
            wall: 120.0,
            binder: BindingWall::Memory { cap: 120 },
            source: Some(KvBoundSource::Observed),
            empirical: false,
        };
        let text = format_config_headroom_window_issue(&d, 100, 0.8, Some(&rec), true).join("\n");
        assert!(text.contains("Recommended   96 (bound by memory limit 120, vLLM-reported)"));
        assert!(text.contains("Compute and KV memory headroom available."));
        assert!(!text.contains("~120"));
        assert!(!text.contains("(est)"));
        assert!(text.contains("Raise --max-num-seqs to 96."));
        assert!(!text.contains("at least"));
    }

    #[test]
    fn format_memory_derived_tilde_and_est() {
        let d = ConfigHeadroomDetail {
            max_num_seqs: 32,
            recommended_seqs: 96,
            ridge_batch_size: Some(153.0),
            occupancy_pct: 62.5,
            running: 20.0,
        };
        let rec = RecommendedSeqs {
            target: 96,
            wall: 120.0,
            binder: BindingWall::Memory { cap: 120 },
            source: Some(KvBoundSource::Derived),
            empirical: false,
        };
        let text = format_config_headroom_window_issue(&d, 100, 0.6, Some(&rec), true).join("\n");
        assert!(text.contains("Recommended   96 (at least 120 worst-case requests fit (est))"));
        assert!(text.contains("Compute and KV memory headroom available."));
        assert!(!text.contains("vLLM-reported"));
        assert!(text.contains("Raise --max-num-seqs to 96."));
    }

    #[test]
    fn format_empirical_est_only_with_monitor_and_low() {
        let d = ConfigHeadroomDetail {
            max_num_seqs: 32,
            recommended_seqs: 64,
            ridge_batch_size: Some(153.0),
            occupancy_pct: 100.0,
            running: 32.0,
        };
        let rec = RecommendedSeqs {
            target: 64,
            wall: 400.0,
            binder: BindingWall::Memory { cap: 400 },
            source: Some(KvBoundSource::Empirical),
            empirical: true,
        };
        // Empirical is not a resolved wall for the cause line.
        let text = format_config_headroom_window_issue(&d, 100, 0.5, Some(&rec), false).join("\n");
        assert!(text.contains("Compute headroom available; memory bound unmeasured."));
        assert!(!text.contains("Compute and KV memory headroom available."));
        assert!(!text.contains("KV memory headroom available"));
        assert!(text.contains("Recommended   64 (est)"));
        assert!(!text.contains("bound by"));
        assert!(!text.contains("400"));
        assert!(text.contains("Raise --max-num-seqs to 64."));
        assert!(text.contains("Monitor KV cache when scaling up."));
        assert!(text.contains("Confidence: Low"));
        assert!(!text.contains("at least"));
    }

    #[test]
    fn occupancy_pct_computed_correctly() {
        let s = snap(20.0, 32, 0.0, Some(3.3));
        let d = rule7_config_headroom(&s, None, Some(153.0), None, false).expect("fired");
        assert!((d.occupancy_pct - (20.0 / 32.0 * 100.0)).abs() < 0.1);
    }
}
