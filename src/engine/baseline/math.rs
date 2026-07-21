use crate::context::ModelArch;

/// Conservative activation memory buffer vLLM reserves inside the allocated VRAM block.
pub const ACTIVATION_KV_BUFFER_GB: f64 = 3.0;

/// Prefill throughput at the compute ceiling, in full prompts per second.
///
/// One "op" is one full forward pass over a prompt of length `seq_len`
/// (linear GEMMs + attention). This is **prompts/s**, not tokens/s.
/// Token throughput ≈ `prompts/s × seq_len`. Callers that need tok/s must
/// multiply; display sites must not label this value as tok/s.
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

/// Coarse preflight weight estimate for GPU-assignment threshold only. Runs
/// before live config exists, so it uses the catalog's default_weight_dtype
/// directly, NOT the full resolve_bits_per_param chain. Intentionally separate
/// from the baseline resolver; they answer different questions (fits-on-boot
/// vs runtime ceiling). Do not unify.
///
/// Uses total param_count (not active): for MoE this overstates weight, which
/// is the SAFE direction for a fits-on-boot gate (conservative = won't
/// under-provision). Not a physics figure.
pub fn catalog_model_weight_gb(model_id: &str) -> Option<f64> {
    let entry = crate::context::model_catalog::lookup_model(model_id)?;
    let bits: u8 = match entry.default_weight_dtype {
        "fp8" | "e4m3" | "e5m2" => 8,
        "fp16" | "bf16" => 16,
        "fp32" => 32,
        // Unknown dtype → bf16 fallback, matches baseline resolver; catalog
        // dtypes are tested so unreachable in practice, kept for exhaustiveness.
        _ => 16,
    };
    Some(weight_gb(entry.param_count, bits))
}

/// Theoretical minimum latency (ms) for one unit of work at the given ceiling.
/// decode ceiling (tok/s) → tpot floor; prefill ceiling (prompts/s) → ms per full prompt.
pub fn latency_floor_ms(ceiling_tps: f64) -> f64 {
    1000.0 / ceiling_tps
}

/// KV cache bytes per element from kv_cache_dtype.
/// "fp8" → 1; bf16/fp16 → 2; "auto"/None → falls back to weight_bytes.
/// vLLM does not support fp32 KV cache, so 2 is the correct non-fp8 default.
pub fn kv_bytes_per_element(kv_cache_dtype: Option<&str>, weight_bytes: u8) -> u8 {
    match kv_cache_dtype {
        Some(d)
            if {
                let d = d.trim().to_ascii_lowercase();
                d.contains("fp8") || d.contains("e4m3") || d.contains("e5m2")
            } =>
        {
            1
        }
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

/// Bytes one worst-case request costs in GPU memory. Three currencies:
/// full-attention transcript, window-capped transcript, fixed whiteboard.
/// `None` means a currency this model uses cannot be priced.
pub fn bytes_per_seq(arch: &ModelArch, max_model_len: u32, kv_dtype_bytes: u8) -> Option<u64> {
    if max_model_len == 0 || kv_dtype_bytes == 0 {
        return None;
    }

    let kv_layers = arch.num_kv_layers.or(arch.num_layers)?;
    let num_kv_heads = arch.num_kv_heads?;
    let head_dim = arch.head_dim?;
    let (swa_layers, swa_window) = match (arch.num_swa_layers, arch.swa_window) {
        (None, None) => (0, 0),
        (Some(layers), Some(window)) => (layers, window),
        _ => return None,
    };
    let full_layers = kv_layers.checked_sub(swa_layers)?;
    let per_token_per_layer = 2u64
        .checked_mul(u64::from(num_kv_heads))?
        .checked_mul(u64::from(head_dim))?
        .checked_mul(u64::from(kv_dtype_bytes))?;
    let transcript = u64::from(full_layers)
        .checked_mul(per_token_per_layer)?
        .checked_mul(u64::from(max_model_len))?;
    let window_part = u64::from(swa_layers)
        .checked_mul(per_token_per_layer)?
        .checked_mul(u64::from(max_model_len.min(swa_window)))?;

    let has_linear_state = arch.linear_num_layers.is_some()
        || arch.linear_key_heads.is_some()
        || arch.linear_value_heads.is_some()
        || arch.linear_key_head_dim.is_some()
        || arch.linear_value_head_dim.is_some()
        || arch.linear_conv_kernel_dim.is_some();
    let whiteboard = if has_linear_state {
        catalog_hybrid_state_bytes(
            arch.linear_num_layers?,
            arch.linear_key_heads?,
            arch.linear_value_heads?,
            arch.linear_key_head_dim?,
            arch.linear_value_head_dim?,
            arch.linear_conv_kernel_dim?,
            state_dtype_bytes(arch.state_dtype.as_deref())?,
        )?
    } else {
        0
    };

    transcript.checked_add(window_part)?.checked_add(whiteboard)
}

/// Batch size at which decode transitions from memory-BW-bound to compute-bound.
/// Below this: throughput limited by peak_bw. At or above: limited by peak_flops.
pub fn ridge_batch_size(peak_flops_tc_tflops: f64, peak_bw_gbps: f64, bits_per_param: u8) -> f64 {
    (peak_flops_tc_tflops * 1e12 * bits_per_param as f64) / (peak_bw_gbps * 1e9 * 16.0)
}

/// Project concurrency at a hypothetical `max_model_len` from observed block geometry.
///
/// Derivation (whiteboard cost of non-attention state, in pages):
/// ```text
/// state_pages = round(num_gpu_blocks / observed_concurrency)
///             − ceil(current_max_len / block_size)   // ≥ 0
/// result      = num_gpu_blocks / (ceil(target_max_len / block_size) + state_pages)
/// ```
///
/// Source: H100 ladder 2026-07-17, five configs, zero residual. Dense models:
/// `state_pages = 0` falls out naturally when `blocks / concurrency == attn pages`.
///
/// All inputs come from `cache_config_info` labels. Returns `None` if any are
/// missing or degenerate. Assumptions for callers: block geometry is constant
/// across `max_model_len` changes (ladder-proven); not proven constant across
/// `gpu-memory-utilization` or vLLM versions. `mamba_cache_mode` changes shift
/// `state_pages` (measured 3→6 none→align) — mode-change counterfactuals stay
/// directional, no number.
pub fn counterfactual_concurrency(
    target_max_len: u32,
    block_size: u32,
    num_gpu_blocks: u32,
    observed_concurrency: f64,
    current_max_len: u32,
) -> Option<f64> {
    if target_max_len == 0 || block_size == 0 || num_gpu_blocks == 0 {
        return None;
    }
    let state_pages = observed_state_pages(
        block_size,
        num_gpu_blocks,
        observed_concurrency,
        current_max_len,
    )?;
    let attn_pages_target = attn_pages(target_max_len, block_size)?;
    let denom = attn_pages_target.saturating_add(state_pages);
    if denom == 0 {
        return None;
    }
    let result = f64::from(num_gpu_blocks) / denom as f64;
    result.is_finite().then_some(result)
}

/// Deduced non-attention state cost in pages from observed allocator geometry.
///
/// `state_pages = round(num_gpu_blocks / observed_concurrency)
///              − ceil(current_max_len / block_size)` (≥ 0).
pub fn observed_state_pages(
    block_size: u32,
    num_gpu_blocks: u32,
    observed_concurrency: f64,
    current_max_len: u32,
) -> Option<u64> {
    if block_size == 0
        || num_gpu_blocks == 0
        || current_max_len == 0
        || !observed_concurrency.is_finite()
        || observed_concurrency <= 0.0
    {
        return None;
    }
    let pages_per_seq = (f64::from(num_gpu_blocks) / observed_concurrency).round() as i64;
    let attn_pages_current = i64::try_from(attn_pages(current_max_len, block_size)?).ok()?;
    Some((pages_per_seq - attn_pages_current).max(0) as u64)
}

/// Transcript page cost: `ceil(max_len / block_size)`. `None` when `block_size == 0`.
pub fn attn_pages(max_len: u32, block_size: u32) -> Option<u64> {
    (block_size > 0).then(|| u64::from(max_len.div_ceil(block_size)))
}

/// Two-currency plausibility: fixed state must be smaller than the transcript
/// it rides beside. `state_pages >= attn_pages_current` means the deduction is
/// absorbing a structural mismatch (e.g. sliding-window attention), not
/// measuring state. `state_pages` derives from `vllm:cache_config_info` labels;
/// `attn_pages` joins the label block size with `max_model_len` from server
/// config. No external inputs. Known conservative edge: hybrid models at short
/// `max_model_len` (transcript <= state pages, e.g. `mamba_block_size` 784 with
/// `max_model_len` <= ~2400) trip the gate and stay directional; designed, not
/// a bug.
pub fn page_model_fits(state_pages: u64, attn_pages_current: u64) -> bool {
    state_pages < attn_pages_current
}

/// Fixed per-sequence hybrid (GDN / mamba-class) state bytes from catalog facts.
///
/// ```text
/// recurrent = layers × Kh × Kd × Vd × dtype_bytes
/// conv      = layers × (Kh×Kd×2 + Vh×Vd) × conv_kernel × dtype_bytes
/// total     = recurrent + conv
/// ```
///
/// Recurrent is sized per **key** head, not value head. Grouped value heads
/// share state in vLLM's GDN implementation. Source: H100 ladder 2026-07-17
/// (Qwen3.6, none-mode): `Kh×Kd×Vd` + conv ≈ 56 MB → `ceil(/ page)` = 3,
/// matching observed `state_pages`. Sizing by `Vh` overstated ~3× (~151 MB → 7
/// pages) and falsely flagged the flagship catalog entry as stale.
///
/// Returns `None` if any input is degenerate (`0` dims / dtype).
pub fn catalog_hybrid_state_bytes(
    linear_num_layers: u32,
    linear_key_heads: u32,
    linear_value_heads: u32,
    linear_key_head_dim: u32,
    linear_value_head_dim: u32,
    linear_conv_kernel_dim: u32,
    state_dtype_bytes: u8,
) -> Option<u64> {
    if linear_num_layers == 0
        || linear_key_heads == 0
        || linear_value_heads == 0
        || linear_key_head_dim == 0
        || linear_value_head_dim == 0
        || linear_conv_kernel_dim == 0
        || state_dtype_bytes == 0
    {
        return None;
    }
    let layers = u64::from(linear_num_layers);
    let kh = u64::from(linear_key_heads);
    let vh = u64::from(linear_value_heads);
    let kd = u64::from(linear_key_head_dim);
    let vd = u64::from(linear_value_head_dim);
    let conv_k = u64::from(linear_conv_kernel_dim);
    let dtype_b = u64::from(state_dtype_bytes);

    // Per key head: grouped value heads share the recurrent state (ladder-proven).
    let recurrent = kh.checked_mul(kd)?.checked_mul(vd)?;
    let key_part = kh.checked_mul(kd)?.checked_mul(2)?;
    let value_part = vh.checked_mul(vd)?;
    let conv_dim = key_part.checked_add(value_part)?;
    let conv = conv_dim.checked_mul(conv_k)?;
    let per_layer = recurrent.checked_add(conv)?;
    layers.checked_mul(per_layer)?.checked_mul(dtype_b)
}

/// `ceil(catalog_state_bytes / page_bytes)`. `None` if `page_bytes == 0`.
pub fn catalog_state_pages(catalog_state_bytes: u64, page_bytes: u64) -> Option<u64> {
    if page_bytes == 0 {
        return None;
    }
    Some(catalog_state_bytes.div_ceil(page_bytes))
}

/// Bytes per element for hybrid `state_dtype` catalog strings.
pub fn state_dtype_bytes(dtype: Option<&str>) -> Option<u8> {
    match dtype.map(|s| s.to_ascii_lowercase()).as_deref() {
        Some("fp32" | "float32") => Some(4),
        Some("fp16" | "float16" | "bf16" | "bfloat16") => Some(2),
        Some("fp8" | "fp8_e4m3" | "fp8_e5m2") => Some(1),
        _ => None,
    }
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
        // Catalog H100 SXM BF16 TC: 989 TFLOPS, 3350 GB/s (gpu_catalog.rs).
        // (989e12 × 16) / (3350e9 × 16) = 295.223…
        let r = ridge_batch_size(989.0, 3350.0, 16);
        assert!((r - 295.223).abs() < 0.05);
    }

    #[test]
    fn ridge_batch_size_a100_80gb_bf16() {
        // Catalog A100 80GB BF16 TC: 312 TFLOPS, 2039 GB/s (gpu_catalog.rs).
        // (312e12 × 16) / (2039e9 × 16) = 153.016…
        let r = ridge_batch_size(312.0, 2039.0, 16);
        assert!((r - 153.016).abs() < 0.05);
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
        assert_eq!(kv_bytes_per_element(Some("e4m3fnuz"), 2), 1);
        assert_eq!(kv_bytes_per_element(Some("e5m2"), 2), 1);
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
    fn bytes_per_seq_dense_matches_existing_arithmetic() {
        let arch = ModelArch {
            num_layers: Some(32),
            num_kv_heads: Some(8),
            head_dim: Some(128),
            ..Default::default()
        };
        let got = bytes_per_seq(&arch, 4096, 2);
        let expected = 4096u64 * 2 * 32 * 8 * 128 * 2;
        assert_eq!(got, Some(expected));
    }

    #[test]
    fn bytes_per_seq_qwen36_adds_fixed_whiteboard() {
        let arch = ModelArch {
            num_layers: Some(64),
            num_kv_layers: Some(16),
            num_kv_heads: Some(4),
            head_dim: Some(256),
            linear_num_layers: Some(48),
            linear_key_heads: Some(16),
            linear_value_heads: Some(48),
            linear_key_head_dim: Some(128),
            linear_value_head_dim: Some(128),
            linear_conv_kernel_dim: Some(4),
            state_dtype: Some("fp32".to_string()),
            ..Default::default()
        };
        let transcript = 16u64 * 2 * 4 * 256 * 2 * 4096;
        let whiteboard = 58_195_968;
        assert_eq!(bytes_per_seq(&arch, 4096, 2), Some(transcript + whiteboard));
    }

    #[test]
    fn bytes_per_seq_gemma3_caps_windowed_layers() {
        let arch = ModelArch {
            num_layers: Some(62),
            num_kv_heads: Some(16),
            head_dim: Some(128),
            swa_window: Some(1024),
            num_swa_layers: Some(52),
            ..Default::default()
        };
        let per_token = 2u64 * 16 * 128 * 2;
        let expected = 10 * per_token * 8192 + 52 * per_token * 1024;
        assert_eq!(bytes_per_seq(&arch, 8192, 2), Some(expected));

        let short_expected = 62 * per_token * 512;
        assert_eq!(bytes_per_seq(&arch, 512, 2), Some(short_expected));
    }

    #[test]
    fn bytes_per_seq_declines_unpriced_currency() {
        let linear_without_dtype = ModelArch {
            num_layers: Some(16),
            num_kv_heads: Some(4),
            head_dim: Some(128),
            linear_num_layers: Some(1),
            ..Default::default()
        };
        assert!(bytes_per_seq(&linear_without_dtype, 4096, 2).is_none());

        let half_window = ModelArch {
            num_layers: Some(16),
            num_kv_heads: Some(4),
            head_dim: Some(128),
            swa_window: Some(1024),
            ..Default::default()
        };
        assert!(bytes_per_seq(&half_window, 4096, 2).is_none());
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

    // Source: H100 ladder 2026-07-17 — hybrid (Qwen3-next-style), 390 blocks,
    // block_size 784, observed concurrency 8.667 at max_model_len 32768.
    // Five configs, zero residual vs vLLM-reported kv_cache_max_concurrency.
    #[test]
    fn counterfactual_h100_ladder_geometry() {
        let blocks = 390;
        let block_size = 784;
        let obs = 8.667;
        let current = 32768;
        let at_16384 = counterfactual_concurrency(16384, block_size, blocks, obs, current).unwrap();
        assert!(
            (at_16384 - 16.25).abs() < 1e-9,
            "expected 16.25, got {at_16384}"
        );
        let at_8192 = counterfactual_concurrency(8192, block_size, blocks, obs, current).unwrap();
        assert!(
            (at_8192 - 390.0 / 14.0).abs() < 1e-9,
            "expected 27.857…, got {at_8192}"
        );
        let at_4096 = counterfactual_concurrency(4096, block_size, blocks, obs, current).unwrap();
        assert!(
            (at_4096 - 390.0 / 9.0).abs() < 1e-9,
            "expected 43.333…, got {at_4096}"
        );
    }

    #[test]
    fn counterfactual_align_mode_state_pages_six() {
        // Same ladder geometry; align-mode row: obs 8.125 @ 32768 → state_pages = 6.
        // Identity check at current max_len recovers observed concurrency.
        let c = counterfactual_concurrency(32768, 784, 390, 8.125, 32768).unwrap();
        assert!((c - 8.125).abs() < 1e-9, "expected 8.125, got {c}");
        // Explicit state_pages: round(390/8.125)=48, ceil(32768/784)=42 → 6.
        // At target with 42 attn pages: 390/(42+6)=8.125.
        let pages_per_seq = (390.0_f64 / 8.125).round() as i64;
        let attn = i64::from(32768u32.div_ceil(784));
        assert_eq!(pages_per_seq - attn, 6);
    }

    #[test]
    fn counterfactual_dense_state_pages_zero() {
        // blocks / concurrency == attn pages → state_pages = 0.
        let block_size: u32 = 16;
        let current: u32 = 4096;
        let attn_pages = current.div_ceil(block_size); // 256
        let concurrency = 10.0_f64;
        let blocks = (f64::from(attn_pages) * concurrency) as u32; // 2560
        let at_current =
            counterfactual_concurrency(current, block_size, blocks, concurrency, current).unwrap();
        assert!((at_current - concurrency).abs() < 1e-9);
        let at_half =
            counterfactual_concurrency(2048, block_size, blocks, concurrency, current).unwrap();
        // ceil(2048/16)=128, state=0 → 2560/128 = 20
        assert!((at_half - 20.0).abs() < 1e-9);
    }

    #[test]
    fn counterfactual_none_on_degenerate_inputs() {
        assert!(counterfactual_concurrency(16384, 0, 390, 8.667, 32768).is_none());
        assert!(counterfactual_concurrency(16384, 784, 0, 8.667, 32768).is_none());
        assert!(counterfactual_concurrency(0, 784, 390, 8.667, 32768).is_none());
        assert!(counterfactual_concurrency(16384, 784, 390, 8.667, 0).is_none());
        assert!(counterfactual_concurrency(16384, 784, 390, 0.0, 32768).is_none());
        assert!(counterfactual_concurrency(16384, 784, 390, f64::NAN, 32768).is_none());
        assert!(counterfactual_concurrency(16384, 784, 390, -1.0, 32768).is_none());
    }

    #[test]
    fn observed_state_pages_ladder_none_and_align() {
        assert_eq!(observed_state_pages(784, 390, 8.667, 32768), Some(3));
        assert_eq!(observed_state_pages(784, 390, 8.125, 32768), Some(6));
    }

    #[test]
    fn attn_pages_ceil_divides_and_rejects_zero_block_size() {
        assert_eq!(attn_pages(4096, 16), Some(256));
        assert_eq!(attn_pages(1, 784), Some(1));
        assert_eq!(attn_pages(4096, 0), None);
    }

    #[test]
    fn page_model_gate_rejects_state_as_large_as_transcript() {
        assert!(!page_model_fits(42, 42));
        assert!(!page_model_fits(1862, 512));
    }

    #[test]
    fn page_model_gate_rejects_short_context_hybrid_edge() {
        // Short-context hybrid: three transcript pages cannot distinguish three state pages.
        assert!(!page_model_fits(3, 3));
    }

    #[test]
    fn page_model_gate_accepts_ladder_and_dense_geometry() {
        assert!(page_model_fits(3, 42));
        assert!(page_model_fits(0, 256));
    }

    #[test]
    fn catalog_hybrid_state_bytes_qwen36_fp32() {
        // Qwen3.6-27B catalog facts; GDN recurrent (per key head) + conv.
        // Source: H100 ladder 2026-07-17 — agrees with observed state_pages=3.
        let bytes = catalog_hybrid_state_bytes(48, 16, 48, 128, 128, 4, 4).unwrap();
        assert_eq!(bytes, 58_195_968);
        assert_eq!(catalog_state_pages(bytes, 25_690_112), Some(3));
    }

    #[test]
    fn catalog_state_pages_none_on_zero_page_bytes() {
        assert!(catalog_state_pages(1000, 0).is_none());
    }

    #[test]
    fn state_dtype_bytes_parses_common_names() {
        assert_eq!(state_dtype_bytes(Some("fp32")), Some(4));
        assert_eq!(state_dtype_bytes(Some("BF16")), Some(2));
        assert_eq!(state_dtype_bytes(Some("fp8_e4m3")), Some(1));
        assert_eq!(state_dtype_bytes(None), None);
        assert_eq!(state_dtype_bytes(Some("auto")), None);
    }
}
