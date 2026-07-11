//! Shared polling aggregation for GPU backends.
//! Vendor-neutral: no I/O, no driver calls.

/// Single-tick GPU sample. Populated by vendor-specific polling code.
#[derive(Default)]
pub(super) struct GpuPoll {
    pub(super) util_gpu: Option<u32>,
    pub(super) util_mem: Option<u32>,
    pub(super) power_watts: Option<f64>,
    pub(super) vram_used_mb: Option<u64>,
    pub(super) vram_total_mb: Option<u64>,
    pub(super) temperature_c: Option<f64>,
    pub(super) sm_clock_mhz: Option<u32>,
}

/// Window-aggregated GPU metrics. Aggregation rules:
/// - util, power: mean across polls
/// - vram_used, temperature, sm_clock: last poll
/// - vram_peak, temperature_peak: max across polls
/// - vram_total: last poll (constant)
#[derive(Debug, PartialEq)]
pub(super) struct AggregatedPolls {
    pub(super) gpu_util_pct: Option<f64>,
    pub(super) mem_util_pct: Option<f64>,
    pub(super) power_watts: Option<f64>,
    pub(super) vram_used_mb: Option<u64>,
    pub(super) vram_peak_mb: Option<u64>,
    pub(super) vram_total_mb: Option<u64>,
    pub(super) temperature_c: Option<f64>,
    pub(super) temperature_peak_c: Option<f64>,
    pub(super) sm_clock_mhz: Option<u32>,
}

pub(super) fn aggregate_polls(polls: &[GpuPoll]) -> AggregatedPolls {
    let mut sum_gpu = 0.0f64;
    let mut sum_mem = 0.0f64;
    let mut n_util = 0u32;
    let mut sum_power = 0.0f64;
    let mut n_power = 0u32;

    let mut vram_used_mb = None;
    let mut vram_peak_mb: Option<u64> = None;
    let mut vram_total_mb = None;
    let mut temperature_c = None;
    let mut temperature_peak_c: Option<f64> = None;
    let mut sm_clock_mhz = None;

    for p in polls {
        if let (Some(g), Some(m)) = (p.util_gpu, p.util_mem) {
            sum_gpu += f64::from(g);
            sum_mem += f64::from(m);
            n_util += 1;
        }

        if let Some(w) = p.power_watts {
            sum_power += w;
            n_power += 1;
        }

        if let Some(u) = p.vram_used_mb {
            vram_used_mb = Some(u);
            vram_peak_mb = Some(match vram_peak_mb {
                Some(pk) => pk.max(u),
                None => u,
            });
        }
        if let Some(t) = p.vram_total_mb {
            vram_total_mb = Some(t);
        }
        if let Some(t) = p.temperature_c.filter(|x| x.is_finite()) {
            temperature_c = Some(t);
            temperature_peak_c = Some(match temperature_peak_c {
                Some(pk) => pk.max(t),
                None => t,
            });
        }
        if let Some(c) = p.sm_clock_mhz {
            sm_clock_mhz = Some(c);
        }
    }

    let gpu_util_pct = (n_util > 0).then_some(sum_gpu / f64::from(n_util));
    let mem_util_pct = (n_util > 0).then_some(sum_mem / f64::from(n_util));
    let power_watts = (n_power > 0).then_some(sum_power / f64::from(n_power));

    AggregatedPolls {
        gpu_util_pct,
        mem_util_pct,
        power_watts,
        vram_used_mb,
        vram_peak_mb,
        vram_total_mb,
        temperature_c,
        temperature_peak_c,
        sm_clock_mhz,
    }
}

/// Device indices to poll. Empty env input means "all GPUs on host."
/// Scope vs TP is validated after collection in `validate_tensor_parallel_scope`.
pub(super) fn resolve_device_indices(env_indices: Vec<u32>, host_device_count: u32) -> Vec<u32> {
    if env_indices.is_empty() {
        (0..host_device_count).collect()
    } else {
        env_indices
    }
}

#[cfg(test)]
pub(super) fn sample_poll(
    ug: u32,
    um: u32,
    p: f64,
    vu: u64,
    vt: u64,
    temp: f64,
    sm: u32,
) -> GpuPoll {
    GpuPoll {
        util_gpu: Some(ug),
        util_mem: Some(um),
        power_watts: Some(p),
        vram_used_mb: Some(vu),
        vram_total_mb: Some(vt),
        temperature_c: Some(temp),
        sm_clock_mhz: Some(sm),
    }
}

#[cfg(test)]
mod resolve_device_indices_tests {
    use super::resolve_device_indices;

    #[test]
    fn empty_returns_all_host_gpus() {
        assert_eq!(
            resolve_device_indices(vec![], 8),
            vec![0, 1, 2, 3, 4, 5, 6, 7]
        );
    }

    #[test]
    fn explicit_overrides_host_gpus() {
        assert_eq!(resolve_device_indices(vec![2, 3], 8), vec![2, 3]);
    }

    #[test]
    fn subset_of_host() {
        assert_eq!(resolve_device_indices(vec![0, 1], 8), vec![0, 1]);
    }

    #[test]
    fn larger_than_tp_is_collected_whole_for_dilution_abort() {
        assert_eq!(
            resolve_device_indices(vec![0, 1, 2, 3], 8),
            vec![0, 1, 2, 3]
        );
    }
}

#[cfg(test)]
mod aggregate_polls_tests {
    use super::*;

    const SAMPLE_COUNT: usize = 9;

    #[test]
    fn identical_polls_equals_sample_values() {
        let polls: Vec<_> = (0..SAMPLE_COUNT)
            .map(|_| sample_poll(80, 20, 300.0, 1000, 8000, 55.0, 2100))
            .collect();
        let a = aggregate_polls(&polls);
        assert_eq!(a.gpu_util_pct, Some(80.0));
        assert_eq!(a.mem_util_pct, Some(20.0));
        assert_eq!(a.power_watts, Some(300.0));
        assert_eq!(a.vram_used_mb, Some(1000));
        assert_eq!(a.vram_peak_mb, Some(1000));
        assert_eq!(a.vram_total_mb, Some(8000));
        assert_eq!(a.temperature_c, Some(55.0));
        assert_eq!(a.temperature_peak_c, Some(55.0));
        assert_eq!(a.sm_clock_mhz, Some(2100));
    }

    #[test]
    fn means_util_and_power_last_for_rest() {
        let polls = vec![
            sample_poll(0, 0, 100.0, 100, 8000, 40.0, 1000),
            sample_poll(100, 50, 200.0, 200, 8000, 50.0, 2000),
        ];
        let a = aggregate_polls(&polls);
        assert_eq!(a.gpu_util_pct, Some(50.0));
        assert_eq!(a.mem_util_pct, Some(25.0));
        assert_eq!(a.power_watts, Some(150.0));
        assert_eq!(a.vram_used_mb, Some(200));
        assert_eq!(a.vram_peak_mb, Some(200));
        assert_eq!(a.vram_total_mb, Some(8000));
        assert_eq!(a.temperature_c, Some(50.0));
        assert_eq!(a.temperature_peak_c, Some(50.0));
        assert_eq!(a.sm_clock_mhz, Some(2000));
    }

    #[test]
    fn vram_peak_is_max_across_polls() {
        let polls = vec![
            sample_poll(80, 20, 300.0, 50 * 1024, 8000, 55.0, 2100),
            sample_poll(80, 20, 300.0, 10 * 1024, 8000, 55.0, 2100),
        ];
        let a = aggregate_polls(&polls);
        assert_eq!(a.vram_used_mb, Some(10 * 1024));
        assert_eq!(a.vram_peak_mb, Some(50 * 1024));
    }

    #[test]
    fn temperature_peak_is_max_across_polls() {
        let polls = vec![
            sample_poll(80, 20, 300.0, 1000, 8000, 72.0, 2100),
            sample_poll(80, 20, 300.0, 1000, 8000, 65.0, 2100),
        ];
        let a = aggregate_polls(&polls);
        assert_eq!(a.temperature_c, Some(65.0));
        assert_eq!(a.temperature_peak_c, Some(72.0));
    }

    #[test]
    fn skips_ticks_without_util_pair() {
        let polls = vec![
            GpuPoll {
                power_watts: Some(100.0),
                ..Default::default()
            },
            sample_poll(50, 25, 200.0, 1, 2, 3.0, 4),
        ];
        let a = aggregate_polls(&polls);
        assert_eq!(a.gpu_util_pct, Some(50.0));
        assert_eq!(a.mem_util_pct, Some(25.0));
        assert_eq!(a.power_watts, Some(150.0));
    }

    #[test]
    fn empty_is_all_none() {
        assert_eq!(
            aggregate_polls(&[]),
            AggregatedPolls {
                gpu_util_pct: None,
                mem_util_pct: None,
                power_watts: None,
                vram_used_mb: None,
                vram_peak_mb: None,
                vram_total_mb: None,
                temperature_c: None,
                temperature_peak_c: None,
                sm_clock_mhz: None,
            }
        );
    }
}
