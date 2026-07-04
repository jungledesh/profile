use crate::collectors::RawSnapshot;

/// Fire when max_num_seqs is less than 90% of hardware-recommended capacity.
const CONFIG_HEADROOM_RATIO: f64 = 0.90;

/// Server must be working, not idle.
const OCCUPANCY_FLOOR: f64 = 0.50;

/// No backlog; R5 owns that signal. Below 1.0 is gauge noise, not a real queue.
const WAITING_CEILING: f64 = 1.0;

/// Safety margin on recommended max_num_seqs.
const RECOMMENDED_SEQS_SAFETY_MARGIN: f64 = 0.80;

#[derive(Debug, Clone, PartialEq)]
pub struct ConfigHeadroomDetail {
    pub max_num_seqs: u32,
    pub recommended_seqs: u32,
    pub ridge_batch_size: Option<f64>,
    pub empirical_kv_seqs: Option<u32>,
    pub occupancy_pct: f64,
    pub running: f64,
}

/// Empirical KV capacity estimate from live metrics.
/// running / kv_usage_fraction gives the number of sequences that would fill KV to 100%.
/// Returns None if either metric is missing or kv_usage is too small to extrapolate reliably.
fn empirical_kv_max(running: f64, kv_cache_usage_perc: Option<f64>) -> Option<f64> {
    let kv_frac = kv_cache_usage_perc.filter(|v| v.is_finite() && *v > 1.0)?; // below 1% is noise
    let kv_as_fraction = kv_frac / 100.0;
    Some(running / kv_as_fraction)
}

fn recommended_seqs(ridge_batch_size: Option<f64>, empirical_kv: Option<f64>) -> Option<u32> {
    let ridge = ridge_batch_size.filter(|r| r.is_finite() && *r > 0.0);
    let kv = empirical_kv.filter(|k| k.is_finite() && *k > 0.0);
    let raw = match (ridge, kv) {
        (Some(r), Some(k)) => r.min(k),
        (Some(r), None) => r,
        (None, Some(k)) => k,
        (None, None) => return None,
    };
    let margined = (raw * RECOMMENDED_SEQS_SAFETY_MARGIN).floor();
    u32::try_from(margined as u64).ok().filter(|&n| n > 0)
}

pub fn rule7_config_headroom(
    snapshot: &RawSnapshot,
    config_max_num_seqs: Option<u32>,
    ridge_batch_size: Option<f64>,
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

    let emp_kv = empirical_kv_max(run, snapshot.vllm.kv_cache_usage_perc);
    let recommended = recommended_seqs(ridge_batch_size, emp_kv)?;
    if f64::from(max_n) >= f64::from(recommended) * CONFIG_HEADROOM_RATIO {
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
        recommended_seqs: recommended,
        ridge_batch_size: ridge_batch_size.filter(|r| r.is_finite() && *r > 0.0),
        empirical_kv_seqs: emp_kv.and_then(|v| u32::try_from(v.floor() as u64).ok()),
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

fn median_option_u32(values: &[Option<u32>]) -> Option<u32> {
    let mut present: Vec<u32> = values.iter().filter_map(|v| *v).collect();
    if present.is_empty() {
        None
    } else {
        present.sort_unstable();
        Some(present[present.len() / 2])
    }
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
    // recommended_seqs and empirical_kv vary per window; median over fired windows.
    let first = &details[0];
    let recommended_seqs = median_u32(details.iter().map(|d| d.recommended_seqs).collect())
        .unwrap_or(first.recommended_seqs);
    let empirical_kv_seqs = median_option_u32(
        &details
            .iter()
            .map(|d| d.empirical_kv_seqs)
            .collect::<Vec<_>>(),
    );
    ConfigHeadroomDetail {
        max_num_seqs: first.max_num_seqs,
        recommended_seqs,
        ridge_batch_size: first.ridge_batch_size,
        empirical_kv_seqs,
        occupancy_pct,
        running,
    }
}

fn confidence_label(conf: f64) -> &'static str {
    if conf >= 0.8 {
        "High"
    } else if conf >= 0.6 {
        "Moderate"
    } else {
        "Low"
    }
}

pub(super) fn format_config_headroom_window_issue(
    d: &ConfigHeadroomDetail,
    seen_pct: u32,
    confidence: f64,
) -> Vec<String> {
    let cap_pct = (f64::from(d.max_num_seqs) / f64::from(d.recommended_seqs)) * 100.0;
    let ridge_str = d
        .ridge_batch_size
        .map(|r| format!("{r:.0}"))
        .unwrap_or_else(|| "-".to_string());
    super::with_seen_pct(
        vec![
            "[!] Configured Batch Limit".to_string(),
            String::new(),
            format!("  Config max    {}", d.max_num_seqs),
            format!("  Ridge batch   {ridge_str}"),
            format!("  Recommended   {}", d.recommended_seqs),
            String::new(),
            "  Cause:".to_string(),
            format!(
                "    --max-num-seqs={} caps batch size at {:.0}% of hardware capacity.",
                d.max_num_seqs, cap_pct
            ),
            "    Compute and KV memory headroom available.".to_string(),
            String::new(),
            "  Fix:".to_string(),
            format!(
                "    • Raise --max-num-seqs to at least {}.",
                d.recommended_seqs
            ),
            String::new(),
            "  Expected: Higher decode throughput when traffic concurrency increases.".to_string(),
            format!("  Confidence: {}", confidence_label(confidence)),
        ],
        seen_pct,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collectors::VllmRawMetrics;
    use std::time::SystemTime;

    fn snap(
        running: f64,
        max_num_seqs: u32,
        waiting: f64,
        kv_cache_usage_perc: Option<f64>,
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
                ..Default::default()
            },
            gpus: vec![],
            nvml_host_gpu_count: None,
        }
    }

    #[test]
    fn empirical_kv_max_basic() {
        let kv = empirical_kv_max(20.0, Some(3.3)).expect("kv");
        assert!((kv - (20.0 / 0.033)).abs() < 1.0);
        let s = snap(20.0, 32, 0.0, Some(3.3));
        let d = rule7_config_headroom(&s, None, Some(153.0)).expect("fired");
        assert_eq!(d.recommended_seqs, 122);
        assert_eq!(d.empirical_kv_seqs, Some(606));
        assert_eq!(d.ridge_batch_size, Some(153.0));
    }

    #[test]
    fn empirical_kv_max_below_ridge() {
        let kv = empirical_kv_max(50.0, Some(80.0)).expect("kv");
        assert!((kv - 62.5).abs() < 0.01);
        let s = snap(50.0, 20, 0.0, Some(80.0));
        let d = rule7_config_headroom(&s, None, Some(153.0)).expect("fired");
        assert_eq!(d.recommended_seqs, 50);
    }

    #[test]
    fn empirical_kv_max_noise_floor() {
        assert!(empirical_kv_max(20.0, Some(0.5)).is_none());
    }

    #[test]
    fn safety_margin_applied() {
        let s = snap(20.0, 32, 0.0, Some(3.3));
        let d = rule7_config_headroom(&s, None, Some(153.0)).expect("fired");
        let raw = 153.0_f64.min(20.0 / 0.033);
        assert_eq!(
            d.recommended_seqs,
            (raw * RECOMMENDED_SEQS_SAFETY_MARGIN).floor() as u32
        );
    }

    #[test]
    fn kv_unavailable_uses_ridge_only() {
        let s = snap(20.0, 32, 0.0, None);
        let d = rule7_config_headroom(&s, None, Some(153.0)).expect("fired");
        assert_eq!(d.recommended_seqs, 122);
        assert!(d.empirical_kv_seqs.is_none());
    }

    #[test]
    fn fires_when_config_well_below_recommended() {
        let s = snap(20.0, 32, 0.0, Some(3.3));
        let d = rule7_config_headroom(&s, None, Some(153.0)).expect("fired");
        assert_eq!(d.max_num_seqs, 32);
        assert_eq!(d.recommended_seqs, 122);
    }

    #[test]
    fn fires_when_config_at_80_pct_of_recommended() {
        // max_num_seqs=98, ridge=153, empirical_kv large
        // recommended = min(153, large) * 0.80 = 122
        // 98 < 122 * 0.90 = 109.8 → fires
        let s = snap(95.0, 98, 0.0, Some(15.6));
        let d = rule7_config_headroom(&s, None, Some(153.0)).expect("fired");
        assert_eq!(d.recommended_seqs, 122);
    }

    #[test]
    fn does_not_fire_when_config_near_recommended() {
        let s = snap(20.0, 115, 0.0, Some(3.3));
        assert!(rule7_config_headroom(&s, None, Some(153.0)).is_none());
    }

    #[test]
    fn does_not_fire_when_waiting_nonzero() {
        let s = snap(20.0, 32, 5.0, Some(3.3));
        assert!(rule7_config_headroom(&s, None, Some(153.0)).is_none());
    }

    #[test]
    fn does_not_fire_when_occupancy_low() {
        let s = snap(2.0, 32, 0.0, Some(3.3));
        assert!(rule7_config_headroom(&s, None, Some(153.0)).is_none());
    }

    #[test]
    fn does_not_fire_when_ridge_unavailable() {
        let s = snap(20.0, 32, 0.0, None);
        assert!(rule7_config_headroom(&s, None, None).is_none());
    }

    #[test]
    fn uses_min_of_ridge_and_empirical_kv() {
        let s = snap(50.0, 20, 0.0, Some(80.0));
        let d = rule7_config_headroom(&s, None, Some(200.0)).expect("fired");
        assert_eq!(d.recommended_seqs, 50);
    }

    #[test]
    fn ridge_stored_as_none_when_only_kv_drives_recommendation() {
        let s = snap(50.0, 20, 0.0, Some(80.0));
        let d = rule7_config_headroom(&s, None, None).expect("fired on kv only");
        assert!(d.ridge_batch_size.is_none());
        assert_eq!(d.recommended_seqs, 50);
    }

    #[test]
    fn fractional_waiting_below_ceiling_does_not_suppress() {
        let s = snap(20.0, 32, 0.5, Some(3.3));
        assert!(rule7_config_headroom(&s, None, Some(153.0)).is_some());
    }

    #[test]
    fn format_shows_dash_when_ridge_unavailable() {
        let d = ConfigHeadroomDetail {
            max_num_seqs: 32,
            recommended_seqs: 122,
            ridge_batch_size: None,
            empirical_kv_seqs: Some(606),
            occupancy_pct: 62.5,
            running: 20.0,
        };
        let text = format_config_headroom_window_issue(&d, 100, 0.6).join("\n");
        assert!(text.contains("Ridge batch   -"));
        assert!(text.contains("Confidence: Moderate"));
    }

    #[test]
    fn occupancy_pct_computed_correctly() {
        let s = snap(20.0, 32, 0.0, Some(3.3));
        let d = rule7_config_headroom(&s, None, Some(153.0)).expect("fired");
        assert!((d.occupancy_pct - (20.0 / 32.0 * 100.0)).abs() < 0.1);
    }
}
