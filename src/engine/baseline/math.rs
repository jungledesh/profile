/// Conservative activation memory buffer vLLM reserves inside the allocated VRAM block.
pub const ACTIVATION_KV_BUFFER_GB: f64 = 3.0;

pub fn prefill_ceiling_tps(peak_flops_tc_tflops: f64, param_count: u64, seq_len: u32) -> f64 {
    (peak_flops_tc_tflops * 1e12_f64) / (2.0 * param_count as f64 * seq_len as f64)
}

pub fn decode_ceiling_tps(peak_bw_gbps: f64, param_count: u64, bits_per_param: u8) -> f64 {
    (peak_bw_gbps * 1e9_f64 * 8.0) / (param_count as f64 * bits_per_param as f64)
}

pub fn efficiency_pct(actual_tps: f64, decode_ceiling: f64) -> f64 {
    (actual_tps / decode_ceiling) * 100.0
}

pub fn weight_gb(param_count: u64, bits_per_param: u8) -> f64 {
    (param_count as f64 * bits_per_param as f64) / (8.0 * 1e9)
}

/// Theoretical minimum latency (ms) for one token at the given ceiling.
/// decode ceiling → tpot floor; prefill ceiling → prefill latency floor.
pub fn latency_floor_ms(ceiling_tps: f64) -> f64 {
    1000.0 / ceiling_tps
}

/// Batch size at which decode transitions from memory-BW-bound to compute-bound.
/// Below this: throughput limited by peak_bw. At or above: limited by peak_flops.
pub fn ridge_batch_size(peak_flops_tc_tflops: f64, peak_bw_gbps: f64, bits_per_param: u8) -> f64 {
    (peak_flops_tc_tflops * 1e12 * bits_per_param as f64) / (peak_bw_gbps * 1e9 * 16.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefill_happy_path() {
        let tps = prefill_ceiling_tps(67.0, 70_000_000_000, 2048);
        assert!(tps.is_finite());
        assert!((tps - 0.233_677_455_357_142_85).abs() < 1e-15);
    }

    #[test]
    fn decode_happy_path() {
        let tps = decode_ceiling_tps(3350.0, 8_000_000_000, 16);
        assert!((tps - 209.375).abs() < 1e-6);
    }

    #[test]
    fn efficiency_happy_path() {
        let pct = efficiency_pct(10.0, 20.0);
        assert!((pct - 50.0).abs() < 1e-9);
    }

    #[test]
    fn prefill_zero_param_count_returns_infinity() {
        let tps = prefill_ceiling_tps(67.0, 0, 2048);
        assert!(tps.is_infinite());
    }

    #[test]
    fn prefill_zero_seq_len_returns_infinity() {
        let tps = prefill_ceiling_tps(67.0, 70_000_000_000, 0);
        assert!(tps.is_infinite());
    }

    #[test]
    fn decode_zero_param_count_returns_infinity() {
        let tps = decode_ceiling_tps(3350.0, 0, 16);
        assert!(tps.is_infinite());
    }

    #[test]
    fn weight_gb_happy_path() {
        // 8B params × 16 bits = 16GB
        assert!((weight_gb(8_000_000_000, 16) - 16.0).abs() < 1e-6);
        // 70B params × 16 bits = 140GB
        assert!((weight_gb(70_000_000_000, 16) - 140.0).abs() < 1e-6);
    }

    #[test]
    fn latency_floor_ms_happy_path() {
        // decode ceiling ~209.375 tok/s → tpot floor ~4.776ms
        let floor = latency_floor_ms(209.375);
        assert!((floor - 4.776).abs() < 0.001);
        // decode ceiling ~23.93 tok/s → tpot floor ~41.8ms
        let floor2 = latency_floor_ms(23.928_571_428_571_43);
        assert!((floor2 - 41.8).abs() < 0.1);
    }

    #[test]
    fn ridge_batch_size_h100_sxm_bf16() {
        // (67e12 × 16) / (3350e9 × 16) = 20.0
        let r = ridge_batch_size(67.0, 3350.0, 16);
        assert!((r - 20.0).abs() < 0.05);
    }

    #[test]
    fn ridge_batch_size_a100_80gb_bf16() {
        // (19.5e12 × 16) / (2039e9 × 16) = 9.564
        let r = ridge_batch_size(19.5, 2039.0, 16);
        assert!((r - 9.564).abs() < 0.05);
    }

    #[test]
    fn ridge_batch_size_l40s_fp8() {
        // (91.6e12 × 8) / (864e9 × 16) = 53.009
        let r = ridge_batch_size(91.6, 864.0, 8);
        assert!((r - 53.009).abs() < 0.1);
    }

    #[test]
    fn decode_ceiling_awq_4bit_gives_4x_higher_ceiling_than_bf16() {
        let bf16 = decode_ceiling_tps(3350.0, 70_000_000_000, 16);
        let awq = decode_ceiling_tps(3350.0, 70_000_000_000, 4);
        assert!((awq / bf16 - 4.0).abs() < 1e-9);
    }

    #[test]
    fn non_finite_inputs_propagate() {
        let nan_prefill = prefill_ceiling_tps(f64::NAN, 70_000_000_000, 2048);
        assert!(nan_prefill.is_nan());

        let inf_decode = decode_ceiling_tps(f64::INFINITY, 70_000_000_000, 16);
        assert!(inf_decode.is_infinite());

        let nan_eff = efficiency_pct(f64::NAN, 20.0);
        assert!(nan_eff.is_nan());
    }
}
