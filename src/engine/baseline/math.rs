/// Conservative activation memory buffer vLLM reserves inside the allocated VRAM block.
pub const ACTIVATION_KV_BUFFER_GB: f64 = 3.0;

/// Prefill operations per second at the compute ceiling.
///
/// Accounts for both linear layer FLOPs (2 × params × seq_len) and
/// quadratic attention FLOPs (coeff × num_layers × seq_len²).
///
/// `attn_coeff`: per-layer attention FLOPs coefficient for the seq_len² term.
///   Standard MHA/GQA: 2 × hidden_dim.
///   Non-standard (MLA, interleaved): architecture-specific value from catalog.
///   0 → skip attention correction (linear-only fallback).
pub fn prefill_ops_per_sec(
    peak_flops_tc_tflops: f64,
    param_count: u64,
    seq_len: u32,
    num_layers: u32,
    attn_coeff: u64,
) -> f64 {
    let linear = 2.0 * param_count as f64 * seq_len as f64;
    let attention = attn_coeff as f64 * num_layers as f64 * (seq_len as f64).powi(2);
    let total_flops = linear + attention;
    if total_flops == 0.0 {
        return f64::INFINITY;
    }
    (peak_flops_tc_tflops * 1e12) / total_flops
}

pub fn decode_ceiling_tps(peak_bw_gbps: f64, param_count: u64, bits_per_param: u8) -> f64 {
    (peak_bw_gbps * 1e9_f64 * 8.0) / (param_count as f64 * bits_per_param as f64)
}

pub fn efficiency_pct(actual_tps: f64, decode_ceiling: f64) -> f64 {
    (actual_tps / decode_ceiling) * 100.0
}

/// Efficiency relative to what max_num_seqs allows, not ridge.
/// Below ridge: config_ceiling = decode_ceiling_tps * max_num_seqs.
/// At/above ridge: config_ceiling = decode_ceiling_tps * ridge (physics cap).
pub fn config_relative_efficiency_pct(
    actual_tps: f64,
    decode_ceiling_tps: f64,
    max_num_seqs: u32,
    ridge: f64,
) -> f64 {
    let effective_batch = f64::from(max_num_seqs).min(ridge);
    let config_ceiling = decode_ceiling_tps * effective_batch;
    if !config_ceiling.is_finite() || config_ceiling <= 0.0 {
        return 0.0;
    }
    (actual_tps / config_ceiling) * 100.0
}

pub fn weight_gb(param_count: u64, bits_per_param: u8) -> f64 {
    (param_count as f64 * bits_per_param as f64) / (8.0 * 1e9)
}

/// Theoretical minimum latency (ms) for one token at the given ceiling.
/// decode ceiling → tpot floor; prefill ceiling → prefill latency floor.
pub fn latency_floor_ms(ceiling_tps: f64) -> f64 {
    1000.0 / ceiling_tps
}

/// KV cache bytes per element from kv_cache_dtype.
/// "fp8" → 1; bf16/fp16 → 2; "auto"/None → falls back to weight_bytes.
/// vLLM does not support fp32 KV cache, so 2 is the correct non-fp8 default.
pub fn kv_bytes_per_element(kv_cache_dtype: Option<&str>, weight_bytes: u8) -> u8 {
    match kv_cache_dtype {
        Some(d) if d.trim().to_ascii_lowercase().contains("fp8") => 1,
        Some(d)
            if {
                let d = d.trim().to_ascii_lowercase();
                d.contains("fp16") || d.contains("bf16") || d.contains("float16") || d == "half"
            } =>
        {
            2
        }
        _ => weight_bytes,
    }
}

/// Maximum concurrent sequences the KV budget supports at the given context length.
///
/// Formula: `kv_budget_bytes / (max_model_len × 2 × num_layers × num_kv_heads × head_dim × bytes_per_elem)`
///
/// Returns `None` if any input is zero or the budget is too small for even one sequence.
/// This is an upper bound - vLLM block fragmentation may reduce it slightly in practice.
pub fn kv_max_concurrent_seqs(
    kv_headroom_gb: f64,
    max_model_len: u32,
    num_layers: u32,
    num_kv_heads: u32,
    head_dim: u32,
    bytes_per_elem: u8,
) -> Option<u32> {
    if max_model_len == 0
        || num_layers == 0
        || num_kv_heads == 0
        || head_dim == 0
        || bytes_per_elem == 0
        || kv_headroom_gb <= 0.0
    {
        return None;
    }
    // 2× for K and V tensors separately
    let bytes_per_token = 2u64
        .checked_mul(num_layers as u64)?
        .checked_mul(num_kv_heads as u64)?
        .checked_mul(head_dim as u64)?
        .checked_mul(bytes_per_elem as u64)?;
    let kv_budget_bytes = (kv_headroom_gb * 1e9) as u64;
    let max_seqs = kv_budget_bytes
        .checked_div(bytes_per_token)?
        .checked_div(max_model_len as u64)?;
    u32::try_from(max_seqs).ok().filter(|&n| n > 0)
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
    fn prefill_with_attention_flops_short_context() {
        // Llama 3 8B: 32 layers, hidden_dim=4096, attn_coeff = 2*4096 = 8192
        let with = prefill_ops_per_sec(67.0, 8_000_000_000, 512, 32, 8192);
        let without = prefill_ops_per_sec(67.0, 8_000_000_000, 512, 0, 0);
        let diff_pct = ((without - with) / without) * 100.0;
        assert!(
            diff_pct < 1.0,
            "at 512 tokens, attention correction should be <1%, got {diff_pct:.2}%"
        );
    }

    #[test]
    fn prefill_with_attention_flops_long_context() {
        let with = prefill_ops_per_sec(67.0, 8_000_000_000, 131072, 32, 8192);
        let without = prefill_ops_per_sec(67.0, 8_000_000_000, 131072, 0, 0);
        assert!(
            with < without * 0.6,
            "at 128k, attention should reduce ceiling significantly"
        );
        assert!(
            with > without * 0.2,
            "sanity: shouldn't reduce by more than 5x"
        );
    }

    #[test]
    fn prefill_attn_coeff_zero_matches_linear_only() {
        let linear = prefill_ops_per_sec(67.0, 8_000_000_000, 2048, 0, 0);
        let zero_coeff = prefill_ops_per_sec(67.0, 8_000_000_000, 2048, 32, 0);
        assert!((linear - zero_coeff).abs() < 1e-6);
    }

    #[test]
    fn prefill_mla_coeff_produces_lower_ceiling_than_hidden_dim() {
        let standard = prefill_ops_per_sec(67.0, 37_000_000_000, 4096, 61, 14336);
        let mla = prefill_ops_per_sec(67.0, 37_000_000_000, 4096, 61, 139264);
        assert!(mla < standard, "MLA coeff should produce lower ceiling");
    }

    #[test]
    fn prefill_zero_param_count_returns_infinity() {
        let tps = prefill_ops_per_sec(67.0, 0, 2048, 0, 0);
        assert!(tps.is_infinite());
    }

    #[test]
    fn prefill_zero_seq_len_returns_infinity() {
        let tps = prefill_ops_per_sec(67.0, 8_000_000_000, 0, 32, 8192);
        assert!(tps.is_infinite());
    }

    #[test]
    fn prefill_zero_params_and_zero_coeff_returns_infinity() {
        let tps = prefill_ops_per_sec(67.0, 0, 2048, 0, 0);
        assert!(tps.is_infinite());
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
    fn decode_zero_param_count_returns_infinity() {
        let tps = decode_ceiling_tps(3350.0, 0, 16);
        assert!(tps.is_infinite());
    }

    #[test]
    fn config_relative_efficiency_below_ridge() {
        // max_num_seqs=32, ridge=153, decode_ceiling=127
        // config_ceiling = 127 * 32 = 4064
        // actual 386 / 4064 = 9.5%
        let pct = config_relative_efficiency_pct(386.0, 127.0, 32, 153.0);
        assert!((pct - 9.5).abs() < 0.1);
    }

    #[test]
    fn config_relative_efficiency_above_ridge_caps_at_ridge() {
        // max_num_seqs=256, ridge=153, decode_ceiling=127
        // config_ceiling = 127 * 153 = 19431 (capped at ridge)
        // actual 5000 / 19431 = 25.7%
        let pct = config_relative_efficiency_pct(5000.0, 127.0, 256, 153.0);
        assert!((pct - 25.7).abs() < 0.2);
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
    fn kv_bytes_per_element_fp8() {
        assert_eq!(kv_bytes_per_element(Some("fp8"), 2), 1);
        assert_eq!(kv_bytes_per_element(Some("FP8"), 2), 1);
    }

    #[test]
    fn kv_bytes_per_element_bf16() {
        assert_eq!(kv_bytes_per_element(Some("bf16"), 2), 2);
        assert_eq!(kv_bytes_per_element(Some("fp16"), 2), 2);
        assert_eq!(kv_bytes_per_element(Some("half"), 2), 2);
    }

    #[test]
    fn kv_bytes_per_element_auto_falls_back_to_weight_bytes() {
        assert_eq!(kv_bytes_per_element(Some("auto"), 2), 2);
        assert_eq!(kv_bytes_per_element(None, 2), 2);
        // fp8 weights → fp8 KV fallback
        assert_eq!(kv_bytes_per_element(None, 1), 1);
    }

    #[test]
    fn kv_max_concurrent_seqs_a100_llama3_70b() {
        // A100 80GB, Llama-3 70B BF16: ~18GB KV, 8192 tokens, 80 layers, 8 KV heads, head_dim=128
        // bytes_per_token = 2×80×8×128×2 = 327680 = 320KB
        // max_seqs = 18e9 / 320KB / 8192 ≈ 6.8 → 6
        let result = kv_max_concurrent_seqs(18.0, 8192, 80, 8, 128, 2);
        assert!(result.is_some());
        let n = result.unwrap();
        // Llama 70B at 8192 tokens fits only a handful of seqs - expect 5–8 range
        assert!((5..=8).contains(&n), "expected 5–8, got {n}");
    }

    #[test]
    fn kv_max_concurrent_seqs_fp8_doubles_capacity() {
        let bf16 = kv_max_concurrent_seqs(20.0, 4096, 32, 8, 128, 2).unwrap();
        let fp8 = kv_max_concurrent_seqs(20.0, 4096, 32, 8, 128, 1).unwrap();
        assert_eq!(fp8, bf16 * 2);
    }

    #[test]
    fn kv_max_concurrent_seqs_none_on_zero_inputs() {
        assert!(kv_max_concurrent_seqs(0.0, 4096, 32, 8, 128, 2).is_none());
        assert!(kv_max_concurrent_seqs(20.0, 0, 32, 8, 128, 2).is_none());
        assert!(kv_max_concurrent_seqs(20.0, 4096, 0, 8, 128, 2).is_none());
        assert!(kv_max_concurrent_seqs(20.0, 4096, 32, 0, 128, 2).is_none());
        assert!(kv_max_concurrent_seqs(20.0, 4096, 32, 8, 0, 2).is_none());
        assert!(kv_max_concurrent_seqs(20.0, 4096, 32, 8, 128, 0).is_none());
    }

    #[test]
    fn decode_ceiling_awq_4bit_gives_4x_higher_ceiling_than_bf16() {
        let bf16 = decode_ceiling_tps(3350.0, 70_000_000_000, 16);
        let awq = decode_ceiling_tps(3350.0, 70_000_000_000, 4);
        assert!((awq / bf16 - 4.0).abs() < 1e-9);
    }

    #[test]
    fn non_finite_inputs_propagate() {
        let nan_prefill = prefill_ops_per_sec(f64::NAN, 70_000_000_000, 2048, 32, 8192);
        assert!(nan_prefill.is_nan());

        let inf_decode = decode_ceiling_tps(f64::INFINITY, 70_000_000_000, 16);
        assert!(inf_decode.is_infinite());

        let nan_eff = efficiency_pct(f64::NAN, 20.0);
        assert!(nan_eff.is_nan());
    }
}
