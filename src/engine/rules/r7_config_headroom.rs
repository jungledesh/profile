use crate::collectors::RawSnapshot;

use super::with_seen_pct;

/// Fire when max_num_seqs is less than half of hardware-recommended capacity.
const CONFIG_HEADROOM_RATIO: f64 = 0.5;

/// Server must be working, not idle.
const OCCUPANCY_FLOOR: f64 = 0.50;

/// No backlog; R5 owns that signal.
const WAITING_CEILING: f64 = 0.0;

#[derive(Debug, Clone, PartialEq)]
pub struct ConfigHeadroomDetail {
    pub max_num_seqs: u32,
    pub recommended_seqs: u32,
    pub ridge_batch_size: f64,
    pub kv_affordable_seqs: Option<u32>,
    pub occupancy_pct: f64,
    pub running: f64,
}

fn recommended_seqs(ridge_batch_size: Option<f64>, kv_max_seqs: Option<u32>) -> Option<u32> {
    match (
        ridge_batch_size.filter(|r| r.is_finite() && *r > 0.0),
        kv_max_seqs.filter(|&n| n > 0),
    ) {
        (Some(ridge), Some(kv)) => Some((ridge.round() as u32).min(kv)),
        (Some(ridge), None) => u32::try_from(ridge.round() as u64).ok().filter(|&n| n > 0),
        (None, Some(kv)) => Some(kv),
        (None, None) => None,
    }
}

pub fn rule7_config_headroom(
    snapshot: &RawSnapshot,
    config_max_num_seqs: Option<u32>,
    ridge_batch_size: Option<f64>,
    kv_max_seqs: Option<u32>,
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

    let recommended = recommended_seqs(ridge_batch_size, kv_max_seqs)?;
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

    let ridge = ridge_batch_size
        .filter(|r| r.is_finite() && *r > 0.0)
        .unwrap_or(0.0);

    Some(ConfigHeadroomDetail {
        max_num_seqs: max_n,
        recommended_seqs: recommended,
        ridge_batch_size: ridge,
        kv_affordable_seqs: kv_max_seqs.filter(|&n| n > 0),
        occupancy_pct: occupancy * 100.0,
        running: run,
    })
}

pub(super) fn aggregate_r7_detail(details: &[ConfigHeadroomDetail]) -> ConfigHeadroomDetail {
    debug_assert!(
        !details.is_empty(),
        "aggregate_r7_detail called with empty details"
    );
    let n = details.len() as f64;
    let occupancy_pct = details.iter().map(|d| d.occupancy_pct).sum::<f64>() / n;
    let running = details.iter().map(|d| d.running).sum::<f64>() / n;
    // Config-derived fields (max_num_seqs, recommended_seqs, ridge, kv_affordable) are
    // identical across windows. Take from first; only occupancy and running vary.
    let first = &details[0];
    ConfigHeadroomDetail {
        max_num_seqs: first.max_num_seqs,
        recommended_seqs: first.recommended_seqs,
        ridge_batch_size: first.ridge_batch_size,
        kv_affordable_seqs: first.kv_affordable_seqs,
        occupancy_pct,
        running,
    }
}

pub(super) fn format_config_headroom_window_issue(
    d: &ConfigHeadroomDetail,
    seen_pct: u32,
) -> Vec<String> {
    let cap_pct = (f64::from(d.max_num_seqs) / f64::from(d.recommended_seqs)) * 100.0;
    with_seen_pct(
        vec![
            format!(
                "[!] Config Headroom: --max-num-seqs={} caps batch size at {:.0}% of hardware capacity.",
                d.max_num_seqs, cap_pct
            ),
            format!(
                "  Raise to {}. Decode and KV constraints satisfied.",
                d.recommended_seqs
            ),
        ],
        seen_pct,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collectors::VllmRawMetrics;
    use std::time::SystemTime;

    fn snap(running: f64, max_num_seqs: u32, waiting: f64) -> RawSnapshot {
        let t = SystemTime::UNIX_EPOCH;
        RawSnapshot {
            gpu_observed_at: t,
            vllm_observed_at: t,
            timestamp: t,
            vllm: VllmRawMetrics {
                num_requests_running: Some(running),
                num_requests_waiting: Some(waiting),
                max_num_seqs: Some(max_num_seqs),
                generation_tokens_per_sec: Some(100.0),
                window_duration_secs: Some(2.0),
                ..Default::default()
            },
            gpus: vec![],
            nvml_host_gpu_count: None,
        }
    }

    #[test]
    fn fires_when_config_well_below_recommended() {
        let s = snap(20.0, 32, 0.0);
        let d = rule7_config_headroom(&s, None, Some(153.0), Some(96)).expect("fired");
        assert_eq!(d.max_num_seqs, 32);
        assert_eq!(d.recommended_seqs, 96);
        assert_eq!(d.kv_affordable_seqs, Some(96));
    }

    #[test]
    fn does_not_fire_when_config_near_recommended() {
        let s = snap(20.0, 80, 0.0);
        assert!(rule7_config_headroom(&s, None, Some(153.0), Some(96)).is_none());
    }

    #[test]
    fn does_not_fire_when_waiting_nonzero() {
        let s = snap(20.0, 32, 5.0);
        assert!(rule7_config_headroom(&s, None, Some(153.0), Some(96)).is_none());
    }

    #[test]
    fn does_not_fire_when_occupancy_low() {
        let s = snap(2.0, 32, 0.0);
        assert!(rule7_config_headroom(&s, None, Some(153.0), Some(96)).is_none());
    }

    #[test]
    fn does_not_fire_when_ridge_unavailable() {
        let s = snap(20.0, 32, 0.0);
        assert!(rule7_config_headroom(&s, None, None, None).is_none());
    }

    #[test]
    fn uses_min_of_ridge_and_kv() {
        let s = snap(20.0, 32, 0.0);
        let d = rule7_config_headroom(&s, None, Some(200.0), Some(80)).expect("fired");
        assert_eq!(d.recommended_seqs, 80);
    }

    #[test]
    fn uses_ridge_when_kv_unavailable() {
        let s = snap(20.0, 32, 0.0);
        let d = rule7_config_headroom(&s, None, Some(153.0), None).expect("fired");
        assert_eq!(d.recommended_seqs, 153);
    }

    #[test]
    fn occupancy_pct_computed_correctly() {
        let s = snap(20.0, 32, 0.0);
        let d = rule7_config_headroom(&s, None, Some(153.0), Some(96)).expect("fired");
        assert!((d.occupancy_pct - (20.0 / 32.0 * 100.0)).abs() < 0.1);
    }
}
