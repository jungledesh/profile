pub fn prefill_ceiling_tps(peak_flops_f32_tflops: f64, param_count: u64, seq_len: u32) -> f64 {
    (peak_flops_f32_tflops * 1e12_f64) / (6.0 * param_count as f64 * seq_len as f64)
}

pub fn decode_ceiling_tps(peak_bw_gbps: f64, param_count: u64, bytes_per_param: u8) -> f64 {
    (peak_bw_gbps * 1e9_f64) / (param_count as f64 * bytes_per_param as f64)
}

pub fn efficiency_pct(actual_tps: f64, decode_ceiling: f64) -> f64 {
    (actual_tps / decode_ceiling) * 100.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefill_happy_path() {
        let tps = prefill_ceiling_tps(67.0, 70_000_000_000, 2048);
        assert!(tps.is_finite());
        assert!(tps > 0.0);
    }

    #[test]
    fn decode_happy_path() {
        let tps = decode_ceiling_tps(3350.0, 8_000_000_000, 2);
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
        let tps = decode_ceiling_tps(3350.0, 0, 2);
        assert!(tps.is_infinite());
    }

    #[test]
    fn non_finite_inputs_propagate() {
        let nan_prefill = prefill_ceiling_tps(f64::NAN, 70_000_000_000, 2048);
        assert!(nan_prefill.is_nan());

        let inf_decode = decode_ceiling_tps(f64::INFINITY, 70_000_000_000, 2);
        assert!(inf_decode.is_infinite());

        let nan_eff = efficiency_pct(f64::NAN, 20.0);
        assert!(nan_eff.is_nan());
    }
}
