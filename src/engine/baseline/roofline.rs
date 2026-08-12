use crate::collectors::config::DEFAULT_GPU_MEMORY_UTILIZATION;
use crate::collectors::effective_tensor_parallel;
use crate::collectors::{window_is_evaluable, window_is_idle};
use crate::context::{AnalysisInput, gpu_prices};

use super::math::{self, KvCacheDtypeSource};

/// Body of the speculation-guard message (no `Note:` prefix).
const SPEC_GUARD_BODY: &str = "Throughput above the decode ceiling (speculative decoding likely). Efficiency % does not apply.";

/// Scoreboard line: `Note:` prefix (side note above metrics).
pub const SPEC_GUARD_WARNING_LINE: &str = "Note: Throughput above the decode ceiling (speculative decoding likely). Efficiency % does not apply.";

/// Limiter / healthy-exit decline: same fact, no `Note:` (stands alone under Rules clear).
pub const SPEC_GUARD_LIMITER_LINE: &str = SPEC_GUARD_BODY;
/// Which detector proved decode beat the one-token-per-read ceiling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpecDetector {
    /// Mean TPOT faster than `1000 / decode.upper`.
    Tpot,
    /// Generation tok/s per concurrent request above `decode.upper`.
    /// Denom: intra-window mean running, else peak; never last-scrape landing.
    PerStream,
    /// Total generation tok/s above `decode.upper * ridge`.
    Absolute,
}

impl SpecDetector {
    fn preference_rank(self) -> u8 {
        match self {
            Self::Tpot => 0,
            Self::PerStream => 1,
            Self::Absolute => 2,
        }
    }
}

/// Evidence that measured decode beat the one-token-per-read ceiling.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpecEvidence {
    pub detector: SpecDetector,
    /// ms for [`SpecDetector::Tpot`], tok/s otherwise.
    pub measured: f64,
    /// Same unit as `measured`.
    pub bound: f64,
}

/// Prefer D1 (TPOT) over D2 (per-stream) over D3 (absolute).
pub(crate) fn stronger_spec_evidence(a: SpecEvidence, b: SpecEvidence) -> SpecEvidence {
    if a.detector.preference_rank() <= b.detector.preference_rank() {
        a
    } else {
        b
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CostEstimate {
    /// Tokens generated per watt of aligned power draw. None if aligned energy unavailable.
    pub tok_per_watt: Option<f64>,
    /// Energy per generated token (J/tok) from the energy-pair window set. None if unavailable.
    pub joules_per_token: Option<f64>,
    /// Estimated cost per 1M output tokens (USD). None if turnover gate fails or tps missing.
    pub cost_per_million_tokens: Option<f64>,
    /// Source of the cost estimate.
    pub cost_source: CostSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CostSource {
    /// Operator-supplied via --cost-per-hour flag.
    UserProvided,
    /// From gpu_prices.json catalog. Always labeled (est).
    Catalog,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CeilingEstimate {
    pub lower: f64,
    pub expected: f64,
    pub upper: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WeightDtypeSource {
    VllmInfoQuantization,
    EnvVarQuantization,
    VllmConfig,
    VllmInfoEndpoint,
    EnvVar,
    Catalog,
    Fallback,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhysicsBaseline {
    pub decode: CeilingEstimate,
    /// Prefill ceiling in **prompts/s** (full forward passes at `seq_len`), not tok/s.
    /// See [`math::prefill_ops_per_sec`].
    pub prefill: Option<CeilingEstimate>,
    pub efficiency_pct: Option<f64>,
    pub headroom_pct: Option<f64>,
    pub weight_dtype_source: WeightDtypeSource,
    /// Model weight memory footprint in GB. Derived from bits_per_param (quantization chain or dtype chain).
    pub weight_gb: f64,
    /// Bytes per weight parameter (`bits_per_param / 8`). Used for weight footprint only.
    pub weight_bytes_per_param: u8,
    /// KV element width from kv_cache_dtype resolution (never weight width).
    pub kv_bytes_per_element: u8,
    pub kv_cache_dtype_source: KvCacheDtypeSource,
    /// VRAM remaining after weights, if total VRAM is known. Negative means weights alone exceed VRAM.
    pub kv_headroom_gb: Option<f64>,
    /// Theoretical minimum time-per-output-token at decode ceiling (ms).
    pub tpot_floor_ms: f64,
    /// Theoretical minimum prefill latency (ms per full prompt) at prefill ceiling.
    /// None when seq_len unknown.
    pub prefill_latency_floor_ms: Option<f64>,
    /// Concurrent batch size at which decode crosses from BW-bound to compute-bound (roofline ridge).
    pub ridge_batch_size: f64,
    /// Efficiency relative to max_num_seqs config ceiling (not ridge). None when GPU unknown or inputs missing.
    pub config_relative_efficiency_pct: Option<f64>,
    pub cost: Option<CostEstimate>,
    /// Evidence that measured decode beat the one-token-per-read ceiling.
    /// None: guard did not fire. Some: which detector, measured value, bound it beat.
    pub spec_suspected: Option<SpecEvidence>,
    /// When run-level speculation OR is set: `(flagged, active_total)` windows
    /// for the verbose coverage line. Per-window compute leaves this None.
    pub spec_window_counts: Option<(usize, usize)>,
}

#[derive(Debug, Clone, Copy)]
pub struct StaticBaselineSubset {
    bits_per_param: u8,
    weight_dtype_source: WeightDtypeSource,
    ridge_batch_size: f64,
    weight_gb: f64,
    weight_bytes_per_param: u8,
}

/// Lower/upper band around roofline ceiling `expected` (conservative / optimistic).
const CEILING_LOWER_BAND: f64 = 0.85;
const CEILING_UPPER_BAND: f64 = 1.05;

/// Why `compute` returned None, read from the same inputs it guards on.
/// Order matches compute's early returns: GPU first, then model.
pub fn baseline_missing_reason(ctx: &crate::context::StaticContext) -> &'static str {
    if ctx.gpu.peak_flops_tc_tflops.is_none() || ctx.gpu.peak_bw_gbps.is_none() {
        return "GPU not in catalog";
    }
    if ctx.model.active_param_count.is_none() && ctx.model.param_count.is_none() {
        return "model not in catalog";
    }
    "hardware ceiling inputs incomplete"
}

pub fn compute(input: &AnalysisInput<'_>) -> Option<PhysicsBaseline> {
    let ctx = input.ctx;
    let peak_flops = ctx.gpu.peak_flops_tc_tflops?;
    let peak_bw = ctx.gpu.peak_bw_gbps?;
    let collected = input.window.snapshot.collected_gpu_count();
    let tp = effective_tensor_parallel(ctx.config.tensor_parallel_size, collected)? as f64;

    let roofline_params = ctx.model.active_param_count.or(ctx.model.param_count)?;
    let weight_params = ctx.model.param_count.or(ctx.model.active_param_count)?;
    let subset = static_baseline_subset(ctx, peak_flops, peak_bw, weight_params)?;
    compute_with_subset(input, subset, peak_flops, peak_bw, tp, roofline_params)
}

pub fn static_baseline_subset(
    ctx: &crate::context::StaticContext,
    peak_flops: f64,
    peak_bw: f64,
    weight_params: u64,
) -> Option<StaticBaselineSubset> {
    let catalog_default_dtype = ctx.model.default_weight_dtype.as_deref();
    let (bits_per_param, weight_dtype_source) = resolve_bits_per_param(
        ctx.config.vllm_reported_dtype.as_deref(),
        ctx.config.vllm_reported_dtype_resolved,
        ctx.config.vllm_reported_quantization.as_deref(),
        ctx.config.dtype.as_deref(),
        ctx.config.quantization.as_deref(),
        catalog_default_dtype,
    );
    let ridge_batch_size = math::ridge_batch_size(peak_flops, peak_bw, bits_per_param);
    if !ridge_batch_size.is_finite() {
        return None;
    }
    let weight_gb = math::weight_gb(weight_params, bits_per_param);
    let weight_bytes_per_param = (bits_per_param / 8).max(1);
    Some(StaticBaselineSubset {
        bits_per_param,
        weight_dtype_source,
        ridge_batch_size,
        weight_gb,
        weight_bytes_per_param,
    })
}

pub fn compute_with_subset(
    input: &AnalysisInput<'_>,
    subset: StaticBaselineSubset,
    peak_flops: f64,
    peak_bw: f64,
    tp: f64,
    roofline_params: u64,
) -> Option<PhysicsBaseline> {
    let ctx = input.ctx;
    let bits_per_param = subset.bits_per_param;
    let weight_dtype_source = subset.weight_dtype_source;
    let ridge_batch_size = subset.ridge_batch_size;
    let weight_gb = subset.weight_gb;
    let weight_bytes_per_param = subset.weight_bytes_per_param;

    let decode_expected = math::decode_ceiling_tps(peak_bw * tp, roofline_params, bits_per_param);
    let decode = make_estimate(decode_expected)?;

    let seq_len = resolve_seq_len(
        ctx.config.max_model_len,
        input.window.snapshot.vllm.prompt_tokens_mean,
    );
    // Note: Prefill ceiling assumes standard BF16 GEMM throughput. Fused dequantization
    // kernels (e.g., Marlin/AWQ) introduce instruction overhead that typically lowers
    // achievable compute utilization.
    let attn_coeff = ctx
        .model
        .attn_flops_coeff
        .unwrap_or_else(|| ctx.model.hidden_dim.map(|h| 2 * h as u64).unwrap_or(0));
    let num_layers = ctx.model.num_layers.unwrap_or(0);

    let prefill = seq_len.and_then(|len| {
        let expected = math::prefill_ops_per_sec(
            peak_flops * tp,
            roofline_params,
            len,
            num_layers,
            attn_coeff,
        );
        make_estimate(expected)
    });

    let ceiling = decode.expected;
    let snap = &input.window.snapshot;

    // Speculation guard: measured decode faster than one token per weight-read
    // allows. Any detector clears efficiency claims (Humble). Compare to the
    // UPPER ceiling band only; no extra margin (margining a measurement fabricates).
    let spec_suspected = detect_speculation(snap, &decode, ridge_batch_size);

    // Fraction of the absolute hardware ceiling in use. Denominator is ceiling × ridge_batch_size
    // (the compute ceiling), independent of current traffic. An idle server reads low, correctly.
    let (efficiency_pct, config_relative_efficiency_pct, headroom_pct) = if spec_suspected.is_some()
    {
        (None, None, None)
    } else {
        let efficiency_pct = snap
            .vllm
            .generation_tokens_per_sec
            .filter(|v| v.is_finite() && *v > 0.0)
            .map(|actual| {
                let absolute_ceiling = ceiling * ridge_batch_size;
                let pct = math::efficiency_pct(actual, absolute_ceiling);
                // Above expected×ridge but at or below upper×ridge: inside the
                // estimate band. Clamp to 100. Above upper×ridge, D3 already
                // flagged and this branch is not taken.
                if pct.is_finite() && pct > 100.0 {
                    100.0
                } else {
                    pct
                }
            })
            .filter(|pct| pct.is_finite());

        let config_relative_efficiency_pct = snap
            .vllm
            .generation_tokens_per_sec
            .filter(|v| v.is_finite() && *v > 0.0)
            .and_then(|actual| {
                let max_seqs = ctx.config.max_num_seqs?;
                Some(
                    math::config_relative_efficiency_pct(
                        actual,
                        ceiling,
                        max_seqs,
                        ridge_batch_size,
                    )
                    .min(100.0),
                )
            })
            .filter(|pct| pct.is_finite());

        let headroom_pct = efficiency_pct.map(|raw| 100.0 - raw.min(100.0));
        (efficiency_pct, config_relative_efficiency_pct, headroom_pct)
    };

    let kv_dtype = math::effective_kv_cache_dtype(
        snap.vllm.cache_config.cache_dtype.as_deref(),
        ctx.config.kv_cache_dtype.as_deref(),
    );
    let (kv_bytes_per_element, kv_cache_dtype_source) = math::resolve_kv_cache_element(kv_dtype);
    let kv_headroom_gb = ctx.gpu.vram_gb.map(|vram| {
        let gpu_util = ctx
            .config
            .gpu_memory_utilization
            .unwrap_or(DEFAULT_GPU_MEMORY_UTILIZATION);
        (vram * gpu_util) - math::ACTIVATION_KV_BUFFER_GB - (weight_gb / tp)
    });
    let tpot_floor_ms = math::latency_floor_ms(decode.expected);
    let prefill_latency_floor_ms = prefill.map(|p| math::latency_floor_ms(p.expected));

    let tps = snap
        .vllm
        .generation_tokens_per_sec
        .filter(|v| v.is_finite() && *v > 0.0);
    // Energy: energy-pair set only (aligned_power ÷ aligned_generation tok/s).
    // Never join unaligned power with all-active tok/s.
    // $/1M output tok joins cost/hr with generation tok/s only; no GPU clock.
    // Turnover gate: completed requests must cover mean running concurrency,
    // or the rate is noise under load (field is request count, not tokens).
    let tps_for_dollar = dollar_cost_tps(snap, tps);
    let (tok_per_watt, joules_per_token) = energy_metrics(snap);

    let cost = if let Some(hr) = ctx
        .config
        .cost_per_hour
        .filter(|v| v.is_finite() && *v > 0.0)
    {
        Some(build_cost_estimate(
            tok_per_watt,
            joules_per_token,
            hr,
            tps_for_dollar,
            CostSource::UserProvided,
        ))
    } else if let Some(gpu_name) = ctx.gpu.name.as_deref() {
        gpu_prices::lookup_gpu_price(gpu_name).map(|p| {
            build_cost_estimate(
                tok_per_watt,
                joules_per_token,
                p.on_demand_per_hr * tp,
                tps_for_dollar,
                CostSource::Catalog,
            )
        })
    } else {
        None
    };

    Some(PhysicsBaseline {
        decode,
        prefill,
        efficiency_pct,
        headroom_pct,
        weight_dtype_source,
        weight_gb,
        weight_bytes_per_param,
        kv_bytes_per_element,
        kv_cache_dtype_source,
        kv_headroom_gb,
        tpot_floor_ms,
        prefill_latency_floor_ms,
        ridge_batch_size,
        config_relative_efficiency_pct,
        cost,
        spec_suspected,
        spec_window_counts: None,
    })
}

/// Detectors D1–D3 against `decode.upper`. Idle / non-evaluable: no flag.
/// Missing input skips that detector only.
fn detect_speculation(
    snap: &crate::collectors::RawSnapshot,
    decode: &CeilingEstimate,
    ridge_batch_size: f64,
) -> Option<SpecEvidence> {
    if !window_is_evaluable(snap) || window_is_idle(snap) {
        return None;
    }
    let upper = decode.upper;
    if !(upper.is_finite() && upper > 0.0) {
        return None;
    }

    let mut best: Option<SpecEvidence> = None;

    if let Some(tpot) = snap.vllm.tpot_ms.filter(|v| v.is_finite() && *v > 0.0) {
        let floor_ms = 1000.0 / upper;
        if floor_ms.is_finite() && floor_ms > 0.0 && tpot < floor_ms {
            let ev = SpecEvidence {
                detector: SpecDetector::Tpot,
                measured: tpot,
                bound: floor_ms,
            };
            best = Some(best.map_or(ev, |b| stronger_spec_evidence(b, ev)));
        }
    }

    let tps = snap
        .vllm
        .generation_tokens_per_sec
        .filter(|v| v.is_finite() && *v > 0.0);
    // D2: window-rate tok/s over window-average concurrency (averaged ÷ averaged).
    // Landing (last scrape) is the wrong pair: a mid-window drain collapses the
    // denom and false-fires ("speculation likely" with no drafter = a lie).
    // Mean first; peak fallback when mean unread. Never landing.
    if let (Some(tps), Some(running)) = (
        tps,
        snap.vllm
            .num_requests_running_mean
            .filter(|v| v.is_finite() && *v >= 1.0)
            .or_else(|| {
                snap.vllm
                    .num_requests_running_peak
                    .filter(|v| v.is_finite() && *v >= 1.0)
            }),
    ) {
        let per = tps / running;
        if per.is_finite() && per > upper {
            let ev = SpecEvidence {
                detector: SpecDetector::PerStream,
                measured: per,
                bound: upper,
            };
            best = Some(best.map_or(ev, |b| stronger_spec_evidence(b, ev)));
        }
    }

    if let Some(tps) = tps {
        let abs = upper * ridge_batch_size;
        if abs.is_finite() && abs > 0.0 && tps > abs {
            let ev = SpecEvidence {
                detector: SpecDetector::Absolute,
                measured: tps,
                bound: abs,
            };
            best = Some(best.map_or(ev, |b| stronger_spec_evidence(b, ev)));
        }
    }

    best
}

/// Apply run-level speculation OR onto the summary baseline.
///
/// One proven over-ceiling window poisons the run average, so OR is correct here.
/// Opposite of the R5 seat-wall rule (summary does not OR that flag): that flag
/// asserts a bottleneck; this one withdraws a claim. Withdrawing on any proof is
/// Humble; asserting on any spike is not.
pub(crate) fn apply_spec_run_or(
    baseline: &mut Option<PhysicsBaseline>,
    flagged: usize,
    active_total: usize,
    strongest: Option<SpecEvidence>,
) {
    if flagged == 0 {
        return;
    }
    let Some(b) = baseline.as_mut() else {
        return;
    };
    b.efficiency_pct = None;
    b.config_relative_efficiency_pct = None;
    b.headroom_pct = None;
    b.spec_suspected = strongest.or(b.spec_suspected);
    b.spec_window_counts = Some((flagged, active_total));
}

/// J/tok and tok/W from the energy-pair window set only.
///
/// `sum(aligned_power_watts) / aligned_generation_tokens_per_sec`. Empty pair set → both None.
/// Never falls back to unaligned `power_watts` or all-active `generation_tokens_per_sec`.
fn energy_metrics(snap: &crate::collectors::RawSnapshot) -> (Option<f64>, Option<f64>) {
    let aligned_tps = snap
        .vllm
        .aligned_generation_tokens_per_sec
        .filter(|t| t.is_finite() && *t > 0.0);
    let total_power: f64 = snap
        .gpus
        .iter()
        .filter_map(|g| g.aligned_power_watts.filter(|p| p.is_finite()))
        .sum();
    let power = (total_power.is_finite() && total_power > 0.0).then_some(total_power);
    match (aligned_tps, power) {
        (Some(t), Some(p)) => {
            let jtok = p / t;
            let tpw = t / p;
            (
                tpw.is_finite().then_some(tpw),
                // Some implies finite; delta/print consumers rely on it.
                jtok.is_finite().then_some(jtok),
            )
        }
        _ => (None, None),
    }
}

/// True when this iteration completed enough requests to cover mean running.
/// Missing completed or running → not sufficient (show `-`, do not invent).
/// `generation_tokens_completed` is a request count (+Inf bucket), not tokens.
///
/// Uses mean running, not peak: turnover asks whether completions covered the
/// steady concurrent seat count that produced the tok/s in the cost formula.
/// Peak would over-require completions for a brief spike. (KV usable-cap uses
/// peak running for a different job: one burst above the full-context guarantee
/// already falsifies that wall.)
fn dollar_cost_turnover_ok(snap: &crate::collectors::RawSnapshot) -> bool {
    let Some(completed) = snap
        .vllm
        .generation_tokens_completed
        .filter(|c| c.is_finite())
    else {
        return false;
    };
    let Some(running) = snap.vllm.num_requests_running.filter(|r| r.is_finite()) else {
        return false;
    };
    running > 0.0 && completed >= running
}

fn dollar_cost_tps(snap: &crate::collectors::RawSnapshot, tps: Option<f64>) -> Option<f64> {
    tps.filter(|t| *t > 0.0)
        .filter(|_| dollar_cost_turnover_ok(snap))
}

fn build_cost_estimate(
    tok_per_watt: Option<f64>,
    joules_per_token: Option<f64>,
    cost_per_hr: f64,
    tps: Option<f64>,
    cost_source: CostSource,
) -> CostEstimate {
    let cost_per_million_tokens = tps.filter(|t| *t > 0.0).and_then(|t| {
        let cpm = cost_per_hr * 1_000_000.0 / (t * 3600.0);
        // Some implies finite; delta/print consumers rely on it.
        cpm.is_finite().then_some(cpm)
    });
    CostEstimate {
        tok_per_watt,
        joules_per_token,
        cost_per_million_tokens,
        cost_source,
    }
}

fn make_estimate(expected: f64) -> Option<CeilingEstimate> {
    if !expected.is_finite() {
        return None;
    }
    let lower = expected * CEILING_LOWER_BAND;
    let upper = expected * CEILING_UPPER_BAND;
    if !(lower.is_finite() && upper.is_finite()) {
        return None;
    }
    Some(CeilingEstimate {
        lower,
        expected,
        upper,
    })
}

fn resolve_seq_len(max_model_len: Option<u32>, prompt_tokens_mean: Option<f64>) -> Option<u32> {
    if let Some(v) = prompt_tokens_mean
        .filter(|v| v.is_finite())
        .map(|v| v.round())
        .filter(|v| *v > 0.0 && *v <= u32::MAX as f64)
        .map(|v| v as u32)
    {
        return Some(v);
    }
    max_model_len.filter(|v| *v > 0)
}

fn resolve_bits_per_param(
    vllm_reported_dtype: Option<&str>,
    vllm_reported_dtype_resolved: bool,
    vllm_reported_quantization: Option<&str>,
    dtype_env: Option<&str>,
    quant_env: Option<&str>,
    catalog_default_dtype: Option<&str>,
) -> (u8, WeightDtypeSource) {
    if let Some(bits) = vllm_reported_quantization.and_then(quantization_to_bits) {
        return (bits, WeightDtypeSource::VllmInfoQuantization);
    }
    if let Some(bits) = quant_env.and_then(quantization_to_bits) {
        return (bits, WeightDtypeSource::EnvVarQuantization);
    }
    if let Some(bits) = vllm_reported_dtype.and_then(dtype_to_bits) {
        let source = if vllm_reported_dtype_resolved {
            WeightDtypeSource::VllmInfoEndpoint
        } else {
            WeightDtypeSource::VllmConfig
        };
        return (bits, source);
    }
    if let Some(bits) = dtype_env.and_then(dtype_to_bits) {
        return (bits, WeightDtypeSource::EnvVar);
    }
    if let Some(bits) = catalog_default_dtype.and_then(dtype_to_bits) {
        return (bits, WeightDtypeSource::Catalog);
    }
    (16, WeightDtypeSource::Fallback)
}

fn dtype_to_bits(dtype: &str) -> Option<u8> {
    let d = dtype.trim().to_ascii_lowercase();
    if d.is_empty() {
        return None;
    }

    if d.contains("fp8") || d.contains("e4m3") || d.contains("e5m2") {
        return Some(8);
    }
    if d.contains("bf16") || d.contains("fp16") || d.contains("float16") || d == "half" {
        return Some(16);
    }
    if d.contains("fp32") || d.contains("float32") || d == "f32" {
        return Some(32);
    }
    None
}

fn quantization_to_bits(scheme: &str) -> Option<u8> {
    match scheme.trim().to_ascii_lowercase().as_str() {
        "awq" | "awq_marlin" | "gptq" | "gptq_marlin" | "marlin" => Some(4),
        // ModelOpt NVFP4 / FP4 (Muse Glimmer on Blackwell); /info may say modelopt.
        "modelopt" | "modelopt_fp4" | "nvfp4" | "fp4" => Some(4),
        "int8" | "w8a8" | "fp8" => Some(8),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use std::time::SystemTime;

    use crate::{
        collectors::{GpuRawMetrics, RawSnapshot, VllmConfig, VllmRawMetrics},
        context::{AnalysisInput, RuntimeWindow, StaticContext},
    };

    use super::*;

    fn baseline_input(
        model_params: Option<u64>,
        active_params: Option<u64>,
        default_dtype: Option<&str>,
        peak_flops: Option<f64>,
        peak_bw: Option<f64>,
        cfg: VllmConfig,
        snapshot: VllmRawMetrics,
    ) -> (StaticContext, RuntimeWindow) {
        let n_gpus = cfg.tensor_parallel_size.unwrap_or(1).max(1) as usize;
        let ctx = StaticContext {
            model: crate::context::ModelArch {
                param_count: model_params,
                active_param_count: active_params,
                num_layers: Some(1),
                hidden_dim: Some(1),
                default_weight_dtype: default_dtype.map(str::to_string),
                num_kv_heads: None,
                head_dim: None,
                num_kv_layers: None,
                attn_flops_coeff: None,
                linear_num_layers: None,
                linear_key_heads: None,
                linear_value_heads: None,
                linear_key_head_dim: None,
                linear_value_head_dim: None,
                linear_conv_kernel_dim: None,
                state_dtype: None,
                swa_window: None,
                num_swa_layers: None,
            },
            gpu: crate::context::GPUModel {
                name: Some("gpu".to_string()),
                vram_gb: Some(80.0),
                peak_flops_tc_tflops: peak_flops,
                peak_bw_gbps: peak_bw,
            },
            config: cfg,
            fp8_compiler_available: false,
        };
        let win = RuntimeWindow::from_snapshot(RawSnapshot {
            gpu_observed_at: SystemTime::UNIX_EPOCH,
            vllm_observed_at: SystemTime::UNIX_EPOCH,
            timestamp: SystemTime::UNIX_EPOCH,
            vllm: snapshot,
            gpus: vec![GpuRawMetrics::default(); n_gpus],

            host_memory: None,
        });
        (ctx, win)
    }

    #[test]
    fn compute_none_when_catalog_fields_missing() {
        let (ctx, win) = baseline_input(
            None,
            None,
            None,
            Some(67.0),
            Some(3350.0),
            VllmConfig::default(),
            VllmRawMetrics::default(),
        );
        let input = AnalysisInput::new(&ctx, &win);
        assert!(compute(&input).is_none());
        assert_eq!(baseline_missing_reason(&ctx), "model not in catalog");
    }

    #[test]
    fn baseline_missing_reason_gpu_absent() {
        let (ctx, _) = baseline_input(
            Some(8_000_000_000),
            None,
            None,
            None,
            None,
            VllmConfig::default(),
            VllmRawMetrics::default(),
        );
        assert_eq!(baseline_missing_reason(&ctx), "GPU not in catalog");
    }

    #[test]
    fn baseline_missing_reason_model_absent() {
        let (ctx, _) = baseline_input(
            None,
            None,
            None,
            Some(67.0),
            Some(3350.0),
            VllmConfig::default(),
            VllmRawMetrics::default(),
        );
        assert_eq!(baseline_missing_reason(&ctx), "model not in catalog");
    }

    #[test]
    fn baseline_missing_reason_catalog_complete() {
        let (ctx, win) = baseline_input(
            Some(8_000_000_000),
            None,
            None,
            Some(67.0),
            Some(3350.0),
            VllmConfig::default(),
            VllmRawMetrics::default(),
        );
        assert_eq!(
            baseline_missing_reason(&ctx),
            "hardware ceiling inputs incomplete"
        );
        assert!(compute(&AnalysisInput::new(&ctx, &win)).is_some());
    }

    #[test]
    fn compute_prefers_active_params_and_dtype_priority() {
        let cfg = VllmConfig {
            dtype: Some("fp32".to_string()),
            kv_cache_dtype: Some("fp8".to_string()),
            max_model_len: Some(2048),
            ..Default::default()
        };
        let snap = VllmRawMetrics {
            generation_tokens_per_sec: Some(50.0),
            prompt_tokens_mean: Some(100.0),
            ..Default::default()
        };
        let (ctx, win) = baseline_input(
            Some(100_000_000_000),
            Some(10_000_000_000),
            Some("bf16"),
            Some(67.0),
            Some(3350.0),
            cfg,
            snap,
        );
        let input = AnalysisInput::new(&ctx, &win);
        let out = compute(&input);
        assert!(out.is_some());
        let b = out.expect("expected baseline");
        assert_eq!(b.weight_dtype_source, WeightDtypeSource::EnvVar);
        // No runtime cache_dtype → config fp8 fallback.
        assert_eq!(b.kv_bytes_per_element, 1);
        assert_eq!(b.kv_cache_dtype_source, KvCacheDtypeSource::ExplicitFp8);
        let expected_decode = math::decode_ceiling_tps(3350.0, 10_000_000_000, 32);
        assert!((b.decode.expected - expected_decode).abs() < 1e-9);
        assert!(
            (b.weight_gb - 400.0).abs() < 1e-3,
            "weight_gb uses total param_count (100B × 32 bits / 8), got {}",
            b.weight_gb
        );
    }

    #[test]
    fn baseline_kv_dtype_prefers_runtime_label_over_config() {
        let mk = |runtime: Option<&str>, config: Option<&str>| {
            let cfg = VllmConfig {
                dtype: Some("bf16".to_string()),
                kv_cache_dtype: config.map(str::to_string),
                max_model_len: Some(2048),
                ..Default::default()
            };
            let snap = VllmRawMetrics {
                generation_tokens_per_sec: Some(50.0),
                cache_config: crate::collectors::CacheConfigLabels {
                    cache_dtype: runtime.map(str::to_string),
                    ..Default::default()
                },
                ..Default::default()
            };
            let (ctx, win) = baseline_input(
                Some(7_000_000_000),
                None,
                Some("bf16"),
                Some(312.0),
                Some(2039.0),
                cfg,
                snap,
            );
            let b = compute(&AnalysisInput::new(&ctx, &win)).expect("baseline");
            (b.kv_bytes_per_element, b.kv_cache_dtype_source)
        };
        assert_eq!(mk(Some("fp8"), None), (1, KvCacheDtypeSource::ExplicitFp8));
        assert_eq!(mk(None, Some("fp8")), (1, KvCacheDtypeSource::ExplicitFp8));
        assert_eq!(
            mk(Some("bf16"), Some("fp8")),
            (2, KvCacheDtypeSource::ExplicitActivation)
        );
        assert_eq!(mk(None, None), (2, KvCacheDtypeSource::Auto));
    }

    #[test]
    fn vllm_reported_dtype_beats_catalog() {
        let cfg = VllmConfig {
            vllm_reported_dtype: Some("fp8".to_string()),
            max_model_len: Some(2048),
            ..Default::default()
        };
        let snap = VllmRawMetrics {
            generation_tokens_per_sec: Some(50.0),
            num_requests_running: Some(1.0),
            ..Default::default()
        };
        let (ctx, win) = baseline_input(
            Some(8_000_000_000),
            None,
            Some("bf16"),
            Some(67.0),
            Some(3350.0),
            cfg,
            snap,
        );
        let b = compute(&AnalysisInput::new(&ctx, &win)).expect("baseline");
        assert_eq!(b.weight_dtype_source, WeightDtypeSource::VllmConfig);
        let expected_decode = math::decode_ceiling_tps(3350.0, 8_000_000_000, 8);
        assert!((b.decode.expected - expected_decode).abs() < 1e-9);
    }

    #[test]
    fn vllm_info_endpoint_used_when_resolved_from_info() {
        let cfg = VllmConfig {
            vllm_reported_dtype: Some("bfloat16".to_string()),
            vllm_reported_dtype_resolved: true,
            max_model_len: Some(2048),
            ..Default::default()
        };
        let snap = VllmRawMetrics {
            generation_tokens_per_sec: Some(50.0),
            num_requests_running: Some(1.0),
            ..Default::default()
        };
        let (ctx, win) = baseline_input(
            Some(8_000_000_000),
            None,
            Some("fp32"),
            Some(67.0),
            Some(3350.0),
            cfg,
            snap,
        );
        let b = compute(&AnalysisInput::new(&ctx, &win)).expect("baseline");
        assert_eq!(b.weight_dtype_source, WeightDtypeSource::VllmInfoEndpoint);
    }

    #[test]
    fn vllm_reported_dtype_beats_env_and_catalog() {
        let cfg = VllmConfig {
            vllm_reported_dtype: Some("fp8".to_string()),
            dtype: Some("bf16".to_string()),
            max_model_len: Some(2048),
            ..Default::default()
        };
        let snap = VllmRawMetrics {
            generation_tokens_per_sec: Some(50.0),
            num_requests_running: Some(1.0),
            ..Default::default()
        };
        let (ctx, win) = baseline_input(
            Some(8_000_000_000),
            None,
            Some("bf16"),
            Some(67.0),
            Some(3350.0),
            cfg,
            snap,
        );
        let b = compute(&AnalysisInput::new(&ctx, &win)).expect("baseline");
        assert_eq!(b.weight_dtype_source, WeightDtypeSource::VllmConfig);
        let expected_decode = math::decode_ceiling_tps(3350.0, 8_000_000_000, 8);
        assert!((b.decode.expected - expected_decode).abs() < 1e-9);
    }

    #[test]
    fn moe_weight_uses_total_params_roofline_uses_active() {
        let cfg = VllmConfig {
            dtype: Some("bf16".to_string()),
            max_model_len: Some(2048),
            ..Default::default()
        };
        let snap = VllmRawMetrics {
            generation_tokens_per_sec: Some(50.0),
            prompt_tokens_mean: Some(512.0),
            ..Default::default()
        };
        let (ctx, win) = baseline_input(
            Some(671_000_000_000),
            Some(37_000_000_000),
            Some("bf16"),
            Some(67.0),
            Some(3350.0),
            cfg,
            snap,
        );
        let input = AnalysisInput::new(&ctx, &win);
        let b = compute(&input).expect("baseline");
        let expected_weight = math::weight_gb(671_000_000_000, 16);
        assert!((b.weight_gb - expected_weight).abs() < 1e-3);
        let expected_decode = math::decode_ceiling_tps(3350.0, 37_000_000_000, 16);
        assert!((b.decode.expected - expected_decode).abs() < 1e-6);
    }

    #[test]
    fn compute_prefill_none_when_no_seq_len() {
        let cfg = VllmConfig::default();
        let snap = VllmRawMetrics {
            generation_tokens_per_sec: Some(50.0),
            prompt_tokens_mean: None,
            ..Default::default()
        };
        let (ctx, win) = baseline_input(
            Some(8_000_000_000),
            None,
            Some("bf16"),
            Some(67.0),
            Some(3350.0),
            cfg,
            snap,
        );
        let input = AnalysisInput::new(&ctx, &win);
        let out = compute(&input);
        assert!(out.is_some());
        let b = out.expect("expected baseline");
        assert!(b.prefill.is_none());
    }

    #[test]
    fn efficiency_clamped_at_100_when_above_hardware_ceiling() {
        let cfg = VllmConfig {
            kv_cache_dtype: Some("bf16".to_string()),
            max_model_len: Some(2048),
            ..Default::default()
        };
        let snap = VllmRawMetrics {
            generation_tokens_per_sec: Some(10_000.0),
            num_requests_running: Some(1.0),
            ..Default::default()
        };
        let (ctx, win) = baseline_input(
            Some(8_000_000_000),
            None,
            Some("bf16"),
            Some(67.0),
            Some(3350.0),
            cfg,
            snap,
        );
        let input = AnalysisInput::new(&ctx, &win);
        let b = compute(&input).expect("baseline");
        let absolute_ceiling = b.decode.expected * b.ridge_batch_size;
        assert!(
            10_000.0 > absolute_ceiling,
            "test setup: actual must exceed hardware ceiling (ceiling={absolute_ceiling})"
        );
        let eff = b.efficiency_pct.expect("efficiency");
        assert!((eff - 100.0).abs() < 1e-9);
        assert!((b.headroom_pct.expect("headroom") - 0.0).abs() < 1e-9);
    }

    #[test]
    fn efficiency_some_when_actual_below_decode_ceiling() {
        let cfg = VllmConfig {
            kv_cache_dtype: Some("bf16".to_string()),
            max_model_len: Some(2048),
            ..Default::default()
        };
        let snap = VllmRawMetrics {
            generation_tokens_per_sec: Some(50.0),
            num_requests_running: Some(1.0),
            ..Default::default()
        };
        let (ctx, win) = baseline_input(
            Some(8_000_000_000),
            None,
            Some("bf16"),
            Some(67.0),
            Some(3350.0),
            cfg,
            snap,
        );
        let input = AnalysisInput::new(&ctx, &win);
        let b = compute(&input).expect("baseline");
        let eff = b.efficiency_pct.expect("efficiency");
        assert!((0.0..=100.0).contains(&eff));
        assert!(b.headroom_pct.is_some());
    }

    #[test]
    fn compute_filters_non_finite_estimates() {
        let cfg = VllmConfig {
            kv_cache_dtype: Some("bf16".to_string()),
            max_model_len: Some(0),
            ..Default::default()
        };
        let snap = VllmRawMetrics::default();
        let (ctx, win) = baseline_input(
            Some(0),
            None,
            Some("bf16"),
            Some(67.0),
            Some(3350.0),
            cfg,
            snap,
        );
        let input = AnalysisInput::new(&ctx, &win);
        assert!(compute(&input).is_none());
    }

    #[test]
    fn dtype_source_fallback_when_unrecognized_everywhere() {
        let cfg = VllmConfig {
            vllm_reported_dtype: Some("unknown".to_string()),
            dtype: Some("mystery".to_string()),
            max_model_len: Some(1024),
            ..Default::default()
        };
        let snap = VllmRawMetrics::default();
        let (ctx, win) = baseline_input(
            Some(8_000_000_000),
            None,
            Some("??"),
            Some(67.0),
            Some(3350.0),
            cfg,
            snap,
        );
        let input = AnalysisInput::new(&ctx, &win);
        let out = compute(&input);
        assert!(out.is_some());
        let b = out.expect("expected baseline");
        assert_eq!(b.weight_dtype_source, WeightDtypeSource::Fallback);
    }

    #[test]
    fn resolve_seq_len_prefers_prompt_tokens_mean_when_both_present() {
        let cfg = VllmConfig {
            dtype: Some("bf16".to_string()),
            max_model_len: Some(8192),
            ..Default::default()
        };
        let snap_prompt = VllmRawMetrics {
            prompt_tokens_mean: Some(512.0),
            generation_tokens_per_sec: Some(10.0),
            ..Default::default()
        };
        let snap_no_prompt = VllmRawMetrics {
            prompt_tokens_mean: None,
            generation_tokens_per_sec: Some(10.0),
            ..Default::default()
        };
        let (ctx, win_short) = baseline_input(
            Some(8_000_000_000),
            None,
            Some("bf16"),
            Some(67.0),
            Some(3350.0),
            cfg.clone(),
            snap_prompt,
        );
        let (_, win_long) = baseline_input(
            Some(8_000_000_000),
            None,
            Some("bf16"),
            Some(67.0),
            Some(3350.0),
            cfg,
            snap_no_prompt,
        );
        let input_short = AnalysisInput::new(&ctx, &win_short);
        let input_long = AnalysisInput::new(&ctx, &win_long);
        let b_short = compute(&input_short).expect("baseline");
        let b_long = compute(&input_long).expect("baseline");
        let ps = b_short.prefill.expect("prefill");
        let pl = b_long.prefill.expect("prefill");
        assert!(
            ps.expected > pl.expected,
            "shorter seq_len from prompt mean should raise prefill ceiling"
        );
    }

    #[test]
    fn resolve_seq_len_falls_back_to_max_model_len_when_prompt_missing() {
        let cfg = VllmConfig {
            dtype: Some("bf16".to_string()),
            max_model_len: Some(1024),
            ..Default::default()
        };
        let snap = VllmRawMetrics {
            prompt_tokens_mean: None,
            generation_tokens_per_sec: Some(10.0),
            ..Default::default()
        };
        let (ctx, win) = baseline_input(
            Some(8_000_000_000),
            None,
            Some("bf16"),
            Some(67.0),
            Some(3350.0),
            cfg,
            snap,
        );
        let input = AnalysisInput::new(&ctx, &win);
        let b = compute(&input);
        assert!(b.is_some());
        let expected = math::prefill_ops_per_sec(67.0, 8_000_000_000, 1024, 1, 2);
        let got = b.and_then(|x| x.prefill).expect("prefill").expected;
        assert!((got - expected).abs() < 1e-6);
    }

    #[test]
    fn fallback_when_catalog_dtype_none_and_no_env_vars() {
        // StaticContext built with param_count but no default_weight_dtype (e.g. model not in
        // catalog but GPU is). Previously this caused compute() to return None; now hits Fallback.
        let cfg = VllmConfig {
            max_model_len: Some(1024),
            ..Default::default()
        };
        let snap = VllmRawMetrics::default();
        let (ctx, win) = baseline_input(
            Some(8_000_000_000),
            None,
            None, // no catalog dtype
            Some(67.0),
            Some(3350.0),
            cfg,
            snap,
        );
        let input = AnalysisInput::new(&ctx, &win);
        let out = compute(&input);
        assert!(out.is_some());
        let b = out.expect("expected baseline");
        assert_eq!(b.weight_dtype_source, WeightDtypeSource::Fallback);
    }

    #[test]
    fn hw_efficiency_uses_absolute_hardware_ceiling() {
        let cfg = VllmConfig {
            kv_cache_dtype: Some("bf16".to_string()),
            max_model_len: Some(2048),
            ..Default::default()
        };
        let snap = VllmRawMetrics {
            generation_tokens_per_sec: Some(4000.0),
            num_requests_running: Some(64.0),
            ..Default::default()
        };
        let (ctx, win) = baseline_input(
            Some(8_000_000_000),
            None,
            Some("bf16"),
            Some(67.0),
            Some(4800.0),
            cfg,
            snap,
        );
        let input = AnalysisInput::new(&ctx, &win);
        let b = compute(&input).expect("baseline");
        let absolute_ceiling = b.decode.expected * b.ridge_batch_size;
        let expected = math::efficiency_pct(4000.0, absolute_ceiling);
        let eff = b.efficiency_pct.expect("efficiency");
        assert!(
            (eff - expected).abs() < 0.05,
            "expected ~{expected:.1}%, got {eff:.1}%"
        );
    }

    #[test]
    fn hw_efficiency_is_traffic_independent() {
        let cfg = VllmConfig {
            kv_cache_dtype: Some("bf16".to_string()),
            max_model_len: Some(2048),
            ..Default::default()
        };
        let actual_tps = 4000.0;
        let mut efficiencies = Vec::new();
        for running in [1.0, 4.0, 8.0, 16.0, 32.0] {
            let snap = VllmRawMetrics {
                generation_tokens_per_sec: Some(actual_tps),
                num_requests_running: Some(running),
                ..Default::default()
            };
            let (ctx, win) = baseline_input(
                Some(8_000_000_000),
                None,
                Some("bf16"),
                Some(67.0),
                Some(4800.0),
                cfg.clone(),
                snap,
            );
            let b = compute(&AnalysisInput::new(&ctx, &win)).expect("baseline");
            efficiencies.push(b.efficiency_pct.expect("efficiency"));
        }
        let first = efficiencies[0];
        for (i, eff) in efficiencies.iter().enumerate().skip(1) {
            assert!(
                (eff - first).abs() < 1e-9,
                "running index {i}: expected {first}, got {eff}"
            );
        }
    }

    #[test]
    fn efficiency_some_when_num_running_zero() {
        let cfg = VllmConfig {
            kv_cache_dtype: Some("bf16".to_string()),
            max_model_len: Some(2048),
            ..Default::default()
        };
        let snap = VllmRawMetrics {
            generation_tokens_per_sec: Some(6043.0),
            num_requests_running: Some(0.0),
            ..Default::default()
        };
        let (ctx, win) = baseline_input(
            Some(8_000_000_000),
            None,
            Some("bf16"),
            Some(67.0),
            Some(4800.0),
            cfg,
            snap,
        );
        let input = AnalysisInput::new(&ctx, &win);
        let b = compute(&input).expect("baseline");
        assert!(b.efficiency_pct.is_some());
        assert!(b.headroom_pct.is_some());
    }

    #[test]
    fn efficiency_some_when_num_running_missing() {
        let cfg = VllmConfig {
            kv_cache_dtype: Some("bf16".to_string()),
            max_model_len: Some(2048),
            ..Default::default()
        };
        let snap = VllmRawMetrics {
            generation_tokens_per_sec: Some(6043.0),
            num_requests_running: None,
            ..Default::default()
        };
        let (ctx, win) = baseline_input(
            Some(8_000_000_000),
            None,
            Some("bf16"),
            Some(67.0),
            Some(4800.0),
            cfg,
            snap,
        );
        let input = AnalysisInput::new(&ctx, &win);
        let b = compute(&input).expect("baseline");
        assert!(b.efficiency_pct.is_some());
        assert!(b.headroom_pct.is_some());
    }

    #[test]
    fn tok_per_watt_from_power_and_generation_tps() {
        let cfg = VllmConfig {
            kv_cache_dtype: Some("bf16".to_string()),
            max_model_len: Some(2048),
            ..Default::default()
        };
        let snap = VllmRawMetrics {
            generation_tokens_per_sec: Some(100.0),
            aligned_generation_tokens_per_sec: Some(100.0),
            num_requests_running: Some(1.0),
            generation_tokens_completed: Some(1.0),
            ..Default::default()
        };
        let (mut ctx, mut win) = baseline_input(
            Some(8_000_000_000),
            None,
            Some("bf16"),
            Some(67.0),
            Some(3350.0),
            cfg,
            snap,
        );
        ctx.gpu.name = Some("NVIDIA H100 80GB HBM3".to_string());
        win.snapshot.gpus.first_mut().expect("gpu").power_watts = Some(50.0);
        win.snapshot
            .gpus
            .first_mut()
            .expect("gpu")
            .aligned_power_watts = Some(50.0);
        let input = AnalysisInput::new(&ctx, &win);
        let b = compute(&input).expect("baseline");
        let cost = b.cost.expect("cost block");
        let tpw = cost.tok_per_watt.expect("tok/W");
        assert!((tpw - 2.0).abs() < 1e-9);
        assert_eq!(cost.joules_per_token, Some(0.5));
    }

    #[test]
    fn joules_per_token_none_when_power_missing() {
        // No power telemetry: energy fields absent; $/1M still fires from cost_per_hour.
        let cfg = VllmConfig {
            kv_cache_dtype: Some("bf16".to_string()),
            max_model_len: Some(2048),
            cost_per_hour: Some(2.0),
            ..Default::default()
        };
        let snap = VllmRawMetrics {
            generation_tokens_per_sec: Some(100.0),
            num_requests_running: Some(1.0),
            generation_tokens_completed: Some(1.0),
            ..Default::default()
        };
        let (ctx, mut win) = baseline_input(
            Some(8_000_000_000),
            None,
            Some("bf16"),
            Some(67.0),
            Some(3350.0),
            cfg,
            snap,
        );
        win.snapshot.gpus.first_mut().expect("gpu").power_watts = None;
        win.snapshot
            .gpus
            .first_mut()
            .expect("gpu")
            .aligned_power_watts = None;
        let cost = compute(&AnalysisInput::new(&ctx, &win))
            .expect("baseline")
            .cost
            .expect("cost block");
        assert!(cost.tok_per_watt.is_none());
        assert!(cost.joules_per_token.is_none());
        assert!(cost.cost_per_million_tokens.is_some());
        assert_eq!(cost.cost_source, CostSource::UserProvided);
    }

    #[test]
    fn dollar_cost_survives_when_aligned_power_missing_but_raw_present() {
        // All windows skewed: raw power exists, aligned is None. Energy refuses the
        // bad join; $/1M output tok still uses cost_per_hour × tok/s.
        let cfg = VllmConfig {
            kv_cache_dtype: Some("bf16".to_string()),
            max_model_len: Some(2048),
            cost_per_hour: Some(3.6),
            ..Default::default()
        };
        let snap = VllmRawMetrics {
            generation_tokens_per_sec: Some(100.0),
            num_requests_running: Some(1.0),
            generation_tokens_completed: Some(1.0),
            ..Default::default()
        };
        let (ctx, mut win) = baseline_input(
            Some(8_000_000_000),
            None,
            Some("bf16"),
            Some(67.0),
            Some(3350.0),
            cfg,
            snap,
        );
        win.snapshot.gpus.first_mut().expect("gpu").power_watts = Some(400.0);
        win.snapshot
            .gpus
            .first_mut()
            .expect("gpu")
            .aligned_power_watts = None;
        let cost = compute(&AnalysisInput::new(&ctx, &win))
            .expect("baseline")
            .cost
            .expect("cost block");
        assert!(cost.tok_per_watt.is_none());
        assert!(cost.joules_per_token.is_none());
        let cpm = cost.cost_per_million_tokens.expect("$/1M");
        // 3.6 $/hr × 1e6 / (100 tok/s × 3600) = 10.0
        assert!((cpm - 10.0).abs() < 1e-9);
    }

    #[test]
    fn joules_per_token_none_when_tps_zero() {
        let cfg = VllmConfig {
            kv_cache_dtype: Some("bf16".to_string()),
            max_model_len: Some(2048),
            ..Default::default()
        };
        let snap = VllmRawMetrics {
            generation_tokens_per_sec: Some(0.0),
            aligned_generation_tokens_per_sec: None,
            num_requests_running: Some(1.0),
            ..Default::default()
        };
        let (mut ctx, mut win) = baseline_input(
            Some(8_000_000_000),
            None,
            Some("bf16"),
            Some(67.0),
            Some(3350.0),
            cfg,
            snap,
        );
        ctx.gpu.name = Some("NVIDIA H100 80GB HBM3".to_string());
        win.snapshot.gpus.first_mut().expect("gpu").power_watts = Some(50.0);
        win.snapshot
            .gpus
            .first_mut()
            .expect("gpu")
            .aligned_power_watts = Some(50.0);
        let b = compute(&AnalysisInput::new(&ctx, &win)).expect("baseline");
        let cost = b.cost.expect("catalog price");
        assert!(cost.joules_per_token.is_none());
        assert!(cost.tok_per_watt.is_none());
        assert!(cost.cost_per_million_tokens.is_none());
    }

    #[test]
    fn joules_per_token_is_inverse_of_tok_per_watt() {
        let cfg = VllmConfig {
            kv_cache_dtype: Some("bf16".to_string()),
            max_model_len: Some(2048),
            ..Default::default()
        };
        let snap = VllmRawMetrics {
            generation_tokens_per_sec: Some(200.0),
            aligned_generation_tokens_per_sec: Some(200.0),
            num_requests_running: Some(1.0),
            generation_tokens_completed: Some(1.0),
            ..Default::default()
        };
        let (mut ctx, mut win) = baseline_input(
            Some(8_000_000_000),
            None,
            Some("bf16"),
            Some(67.0),
            Some(3350.0),
            cfg,
            snap,
        );
        ctx.gpu.name = Some("NVIDIA H100 80GB HBM3".to_string());
        win.snapshot.gpus.first_mut().expect("gpu").power_watts = Some(100.0);
        win.snapshot
            .gpus
            .first_mut()
            .expect("gpu")
            .aligned_power_watts = Some(100.0);
        let cost = compute(&AnalysisInput::new(&ctx, &win))
            .expect("baseline")
            .cost
            .expect("cost");
        let tpw = cost.tok_per_watt.expect("tok/W");
        let jpt = cost.joules_per_token.expect("J/tok");
        assert!((tpw * jpt - 1.0).abs() < 1e-9);
    }

    #[test]
    fn cost_per_million_tokens_catalog_h100() {
        let cfg = VllmConfig {
            kv_cache_dtype: Some("bf16".to_string()),
            max_model_len: Some(2048),
            ..Default::default()
        };
        let snap = VllmRawMetrics {
            generation_tokens_per_sec: Some(100.0),
            aligned_generation_tokens_per_sec: Some(100.0),
            num_requests_running: Some(1.0),
            generation_tokens_completed: Some(1.0),
            ..Default::default()
        };
        let (mut ctx, win) = baseline_input(
            Some(8_000_000_000),
            None,
            Some("bf16"),
            Some(67.0),
            Some(3350.0),
            cfg,
            snap,
        );
        ctx.gpu.name = Some("NVIDIA H100 80GB HBM3".to_string());
        // Energy needs aligned power; dollar cost does not.
        let mut win = win;
        win.snapshot.gpus.first_mut().expect("gpu").power_watts = Some(400.0);
        win.snapshot
            .gpus
            .first_mut()
            .expect("gpu")
            .aligned_power_watts = Some(400.0);
        let input = AnalysisInput::new(&ctx, &win);
        let b = compute(&input).expect("baseline");
        let cost = b.cost.expect("cost");
        assert_eq!(cost.cost_source, CostSource::Catalog);
        let cpm = cost.cost_per_million_tokens.expect("cpm");
        let expected = 2.99 * 1_000_000.0 / (100.0 * 3600.0);
        assert!((cpm - expected).abs() < 1e-6);
    }

    #[test]
    fn cost_source_user_provided_overrides_catalog() {
        let cfg = VllmConfig {
            cost_per_hour: Some(5.0),
            kv_cache_dtype: Some("bf16".to_string()),
            max_model_len: Some(2048),
            ..Default::default()
        };
        let snap = VllmRawMetrics {
            generation_tokens_per_sec: Some(200.0),
            aligned_generation_tokens_per_sec: Some(200.0),
            num_requests_running: Some(1.0),
            generation_tokens_completed: Some(1.0),
            ..Default::default()
        };
        let (mut ctx, mut win) = baseline_input(
            Some(8_000_000_000),
            None,
            Some("bf16"),
            Some(67.0),
            Some(3350.0),
            cfg,
            snap,
        );
        ctx.gpu.name = Some("NVIDIA H100 80GB HBM3".to_string());
        win.snapshot.gpus.first_mut().expect("gpu").power_watts = Some(400.0);
        win.snapshot
            .gpus
            .first_mut()
            .expect("gpu")
            .aligned_power_watts = Some(400.0);
        let input = AnalysisInput::new(&ctx, &win);
        let cost = compute(&input).expect("baseline").cost.expect("cost");
        assert_eq!(cost.cost_source, CostSource::UserProvided);
        let expected = 5.0 * 1_000_000.0 / (200.0 * 3600.0);
        assert!((cost.cost_per_million_tokens.unwrap() - expected).abs() < 1e-6);
    }

    #[test]
    fn tp2_doubles_decode_ceiling() {
        let cfg_tp1 = VllmConfig {
            dtype: Some("bf16".to_string()),
            max_model_len: Some(2048),
            tensor_parallel_size: Some(1),
            ..Default::default()
        };
        let cfg_tp2 = VllmConfig {
            tensor_parallel_size: Some(2),
            ..cfg_tp1.clone()
        };
        let snap = VllmRawMetrics {
            generation_tokens_per_sec: Some(50.0),
            num_requests_running: Some(1.0),
            ..Default::default()
        };
        let (ctx1, win1) = baseline_input(
            Some(8_000_000_000),
            None,
            Some("bf16"),
            Some(67.0),
            Some(3350.0),
            cfg_tp1,
            snap.clone(),
        );
        let (ctx2, win2) = baseline_input(
            Some(8_000_000_000),
            None,
            Some("bf16"),
            Some(67.0),
            Some(3350.0),
            cfg_tp2,
            snap,
        );
        let b1 = compute(&AnalysisInput::new(&ctx1, &win1)).expect("tp1 baseline");
        let b2 = compute(&AnalysisInput::new(&ctx2, &win2)).expect("tp2 baseline");
        assert!(
            (b2.decode.expected - b1.decode.expected * 2.0).abs() < 1e-6,
            "tp2 decode {} expected ~2× tp1 {}",
            b2.decode.expected,
            b1.decode.expected
        );
    }

    #[test]
    fn kv_headroom_accounts_for_tp() {
        let cfg = VllmConfig {
            dtype: Some("bf16".to_string()),
            max_model_len: Some(2048),
            tensor_parallel_size: Some(2),
            ..Default::default()
        };
        let snap = VllmRawMetrics {
            generation_tokens_per_sec: Some(50.0),
            num_requests_running: Some(1.0),
            ..Default::default()
        };
        let (ctx, win) = baseline_input(
            Some(70_000_000_000),
            None,
            Some("bf16"),
            Some(67.0),
            Some(3350.0),
            cfg,
            snap,
        );
        let b = compute(&AnalysisInput::new(&ctx, &win)).expect("baseline");
        assert!((b.weight_gb - 140.0).abs() < 1e-3);
        let headroom = b.kv_headroom_gb.expect("kv headroom");
        let expected =
            (80.0 * DEFAULT_GPU_MEMORY_UTILIZATION) - math::ACTIVATION_KV_BUFFER_GB - (140.0 / 2.0);
        assert!(
            (headroom - expected).abs() < 1e-3,
            "expected {expected}GB kv headroom per GPU, got {headroom}"
        );
    }

    #[test]
    fn dollar_cost_absent_when_tps_missing() {
        // Price source present → CostEstimate always built; cpm None without tps.
        let cfg = VllmConfig {
            kv_cache_dtype: Some("bf16".to_string()),
            max_model_len: Some(2048),
            ..Default::default()
        };
        let (mut ctx, win) = baseline_input(
            Some(8_000_000_000),
            None,
            Some("bf16"),
            Some(67.0),
            Some(3350.0),
            cfg,
            VllmRawMetrics {
                generation_tokens_per_sec: None,
                num_requests_running: Some(1.0),
                generation_tokens_completed: Some(1.0),
                ..Default::default()
            },
        );
        ctx.gpu.name = Some("NVIDIA H100 80GB HBM3".to_string());
        let cost = compute(&AnalysisInput::new(&ctx, &win))
            .expect("baseline")
            .cost
            .expect("catalog price");
        assert!(cost.cost_per_million_tokens.is_none());
        assert!(cost.joules_per_token.is_none());
    }

    #[test]
    fn dollar_cost_none_when_turnover_below_running() {
        let cfg = VllmConfig {
            kv_cache_dtype: Some("bf16".to_string()),
            max_model_len: Some(2048),
            cost_per_hour: Some(3.6),
            ..Default::default()
        };
        let snap = VllmRawMetrics {
            generation_tokens_per_sec: Some(100.0),
            num_requests_running: Some(10.0),
            generation_tokens_completed: Some(3.0),
            ..Default::default()
        };
        let (ctx, win) = baseline_input(
            Some(8_000_000_000),
            None,
            Some("bf16"),
            Some(67.0),
            Some(3350.0),
            cfg,
            snap,
        );
        let cost = compute(&AnalysisInput::new(&ctx, &win))
            .expect("baseline")
            .cost
            .expect("cost block with price source");
        assert!(cost.cost_per_million_tokens.is_none());
        assert_eq!(cost.cost_source, CostSource::UserProvided);
    }

    #[test]
    fn energy_uses_aligned_tps_not_active_mean() {
        let cfg = VllmConfig {
            kv_cache_dtype: Some("bf16".to_string()),
            max_model_len: Some(2048),
            cost_per_hour: Some(1.0),
            ..Default::default()
        };
        let snap = VllmRawMetrics {
            generation_tokens_per_sec: Some(250.0),
            aligned_generation_tokens_per_sec: Some(100.0),
            num_requests_running: Some(1.0),
            generation_tokens_completed: Some(1.0),
            ..Default::default()
        };
        let (ctx, mut win) = baseline_input(
            Some(8_000_000_000),
            None,
            Some("bf16"),
            Some(67.0),
            Some(3350.0),
            cfg,
            snap,
        );
        win.snapshot
            .gpus
            .first_mut()
            .expect("gpu")
            .aligned_power_watts = Some(200.0);
        let cost = compute(&AnalysisInput::new(&ctx, &win))
            .expect("baseline")
            .cost
            .expect("cost");
        assert!((cost.joules_per_token.expect("J/tok") - 2.0).abs() < 1e-9);
        assert!((cost.tok_per_watt.expect("tok/W") - 0.5).abs() < 1e-9);
        let cpm = cost.cost_per_million_tokens.expect("cpm");
        let expected = 1.0 * 1_000_000.0 / (250.0 * 3600.0);
        assert!((cpm - expected).abs() < 1e-9);
    }

    #[test]
    fn catalog_dollar_cost_without_aligned_power() {
        // Catalog price + tok/s yields $/1M even when power telemetry is absent.
        let cfg = VllmConfig {
            kv_cache_dtype: Some("bf16".to_string()),
            max_model_len: Some(2048),
            ..Default::default()
        };
        let (mut ctx2, mut win2) = baseline_input(
            Some(8_000_000_000),
            None,
            Some("bf16"),
            Some(67.0),
            Some(3350.0),
            cfg,
            VllmRawMetrics {
                generation_tokens_per_sec: Some(100.0),
                num_requests_running: Some(1.0),
                generation_tokens_completed: Some(1.0),
                ..Default::default()
            },
        );
        ctx2.gpu.name = Some("NVIDIA H100 80GB HBM3".to_string());
        win2.snapshot.gpus.first_mut().expect("gpu").power_watts = None;
        win2.snapshot
            .gpus
            .first_mut()
            .expect("gpu")
            .aligned_power_watts = None;
        let cost = compute(&AnalysisInput::new(&ctx2, &win2))
            .expect("baseline")
            .cost
            .expect("catalog $/1M");
        assert!(cost.tok_per_watt.is_none());
        assert!(cost.joules_per_token.is_none());
        assert!(cost.cost_per_million_tokens.is_some());
        assert_eq!(cost.cost_source, CostSource::Catalog);
    }

    #[test]
    fn consumer_amd_no_price_omits_dollar_per_million() {
        // gpu_prices.json has no RX 7900 XTX row → no $/1M tok.
        assert!(crate::context::gpu_prices::lookup_gpu_price("AMD Radeon RX 7900 XTX").is_none());
        let cfg = VllmConfig {
            kv_cache_dtype: Some("bf16".to_string()),
            max_model_len: Some(2048),
            ..Default::default()
        };
        let snap = VllmRawMetrics {
            generation_tokens_per_sec: Some(100.0),
            num_requests_running: Some(1.0),
            ..Default::default()
        };
        let (mut ctx, mut win) = baseline_input(
            Some(8_000_000_000),
            None,
            Some("bf16"),
            Some(61.4),
            Some(960.0),
            cfg,
            snap,
        );
        ctx.gpu.name = Some("AMD Radeon RX 7900 XTX".to_string());
        win.snapshot.gpus.first_mut().expect("gpu").power_watts = Some(300.0);
        win.snapshot
            .gpus
            .first_mut()
            .expect("gpu")
            .aligned_power_watts = Some(300.0);
        let b = compute(&AnalysisInput::new(&ctx, &win)).expect("baseline");
        // No price source → cost absent (energy alone does not create CostEstimate).
        assert!(b.cost.is_none());
    }

    #[test]
    fn awq_quantization_source_used_when_info_provides_scheme() {
        let cfg = VllmConfig {
            vllm_reported_quantization: Some("awq".to_string()),
            max_model_len: Some(2048),
            ..Default::default()
        };
        let snap = VllmRawMetrics {
            generation_tokens_per_sec: Some(50.0),
            num_requests_running: Some(1.0),
            ..Default::default()
        };
        let (ctx, win) = baseline_input(
            Some(70_000_000_000),
            None,
            Some("bf16"),
            Some(67.0),
            Some(3350.0),
            cfg,
            snap,
        );
        let b = compute(&AnalysisInput::new(&ctx, &win)).expect("baseline");
        assert_eq!(
            b.weight_dtype_source,
            WeightDtypeSource::VllmInfoQuantization
        );
        assert!((b.weight_gb - 35.0).abs() < 1e-3);
        let expected_decode = math::decode_ceiling_tps(3350.0, 70_000_000_000, 4);
        assert!(
            (b.decode.expected - expected_decode).abs() < 1e-6,
            "AWQ 4-bit decode ceiling should be 4× higher than bf16; got {}",
            b.decode.expected
        );
    }

    #[test]
    fn env_var_quantization_used_when_info_missing() {
        let cfg = VllmConfig {
            quantization: Some("gptq".to_string()),
            max_model_len: Some(2048),
            ..Default::default()
        };
        let snap = VllmRawMetrics {
            generation_tokens_per_sec: Some(50.0),
            num_requests_running: Some(1.0),
            ..Default::default()
        };
        let (ctx, win) = baseline_input(
            Some(70_000_000_000),
            None,
            Some("bf16"),
            Some(67.0),
            Some(3350.0),
            cfg,
            snap,
        );
        let b = compute(&AnalysisInput::new(&ctx, &win)).expect("baseline");
        assert_eq!(b.weight_dtype_source, WeightDtypeSource::EnvVarQuantization);
        assert!((b.weight_gb - 35.0).abs() < 1e-3);
    }

    #[test]
    fn modelopt_nvfp4_env_quantization_prices_four_bit() {
        let cfg = VllmConfig {
            quantization: Some("modelopt".to_string()),
            max_model_len: Some(2048),
            ..Default::default()
        };
        let snap = VllmRawMetrics {
            generation_tokens_per_sec: Some(50.0),
            num_requests_running: Some(1.0),
            ..Default::default()
        };
        let (ctx, win) = baseline_input(
            Some(29_600_000_000),
            None,
            Some("bf16"),
            Some(209.5),
            Some(1792.0),
            cfg,
            snap,
        );
        let b = compute(&AnalysisInput::new(&ctx, &win)).expect("baseline");
        assert_eq!(b.weight_dtype_source, WeightDtypeSource::EnvVarQuantization);
        assert!((b.weight_gb - 14.8).abs() < 1e-3);
        let expected_decode = math::decode_ceiling_tps(1792.0, 29_600_000_000, 4);
        assert!((b.decode.expected - expected_decode).abs() < 1e-6);
    }

    #[test]
    fn unrecognized_quantization_falls_through_to_dtype_chain() {
        let cfg = VllmConfig {
            vllm_reported_quantization: Some("gguf".to_string()),
            dtype: Some("bf16".to_string()),
            max_model_len: Some(2048),
            ..Default::default()
        };
        let snap = VllmRawMetrics {
            generation_tokens_per_sec: Some(50.0),
            num_requests_running: Some(1.0),
            ..Default::default()
        };
        let (ctx, win) = baseline_input(
            Some(8_000_000_000),
            None,
            Some("bf16"),
            Some(67.0),
            Some(3350.0),
            cfg,
            snap,
        );
        let b = compute(&AnalysisInput::new(&ctx, &win)).expect("baseline");
        assert_eq!(b.weight_dtype_source, WeightDtypeSource::EnvVar);
    }

    #[test]
    fn quantization_beats_prometheus_dtype_for_awq() {
        let cfg = VllmConfig {
            vllm_reported_dtype: Some("bfloat16".to_string()),
            vllm_reported_quantization: Some("awq".to_string()),
            max_model_len: Some(2048),
            ..Default::default()
        };
        let snap = VllmRawMetrics {
            generation_tokens_per_sec: Some(50.0),
            num_requests_running: Some(1.0),
            ..Default::default()
        };
        let (ctx, win) = baseline_input(
            Some(8_000_000_000),
            None,
            Some("bf16"),
            Some(67.0),
            Some(3350.0),
            cfg,
            snap,
        );
        let b = compute(&AnalysisInput::new(&ctx, &win)).expect("baseline");
        assert_eq!(
            b.weight_dtype_source,
            WeightDtypeSource::VllmInfoQuantization
        );
        let expected_decode = math::decode_ceiling_tps(3350.0, 8_000_000_000, 4);
        assert!((b.decode.expected - expected_decode).abs() < 1e-9);
    }

    fn baseline_input_llama8b(
        snapshot: VllmRawMetrics,
        cfg: VllmConfig,
    ) -> (StaticContext, RuntimeWindow) {
        let ctx = StaticContext {
            model: crate::context::ModelArch {
                param_count: Some(8_000_000_000),
                active_param_count: None,
                num_layers: Some(32),
                hidden_dim: Some(4096),
                default_weight_dtype: Some("bf16".to_string()),
                num_kv_heads: Some(8),
                head_dim: Some(128),
                num_kv_layers: None,
                attn_flops_coeff: None,
                linear_num_layers: None,
                linear_key_heads: None,
                linear_value_heads: None,
                linear_key_head_dim: None,
                linear_value_head_dim: None,
                linear_conv_kernel_dim: None,
                state_dtype: None,
                swa_window: None,
                num_swa_layers: None,
            },
            gpu: crate::context::GPUModel {
                name: Some("NVIDIA H100 80GB HBM3".to_string()),
                vram_gb: Some(80.0),
                peak_flops_tc_tflops: Some(67.0),
                peak_bw_gbps: Some(3350.0),
            },
            config: cfg,
            fp8_compiler_available: false,
        };
        let win = RuntimeWindow::from_snapshot(RawSnapshot {
            gpu_observed_at: SystemTime::UNIX_EPOCH,
            vllm_observed_at: SystemTime::UNIX_EPOCH,
            timestamp: SystemTime::UNIX_EPOCH,
            vllm: snapshot,
            gpus: vec![GpuRawMetrics::default()],

            host_memory: None,
        });
        (ctx, win)
    }

    #[test]
    fn prefill_ceiling_uses_attention_correction_when_hidden_dim_known() {
        let cfg = VllmConfig {
            dtype: Some("bf16".to_string()),
            max_model_len: Some(8192),
            ..Default::default()
        };
        let snap = VllmRawMetrics {
            prompt_tokens_mean: Some(2048.0),
            generation_tokens_per_sec: Some(10.0),
            ..Default::default()
        };
        let (ctx, win) = baseline_input_llama8b(snap, cfg);
        let b = compute(&AnalysisInput::new(&ctx, &win)).expect("baseline");
        let p = b.prefill.expect("prefill");
        let linear_only = math::prefill_ops_per_sec(67.0, 8_000_000_000, 2048, 0, 0);
        assert!(p.expected < linear_only);
    }

    #[test]
    fn prefill_ceiling_uses_explicit_attn_coeff_for_mla() {
        let cfg = VllmConfig {
            dtype: Some("bf16".to_string()),
            max_model_len: Some(4096),
            ..Default::default()
        };
        let mut ctx = baseline_input_llama8b(VllmRawMetrics::default(), cfg.clone()).0;
        ctx.model.attn_flops_coeff = Some(139_264);
        ctx.model.param_count = Some(37_000_000_000);
        ctx.model.active_param_count = Some(37_000_000_000);
        ctx.model.num_layers = Some(61);
        let win = RuntimeWindow::from_snapshot(RawSnapshot {
            gpu_observed_at: SystemTime::UNIX_EPOCH,
            vllm_observed_at: SystemTime::UNIX_EPOCH,
            timestamp: SystemTime::UNIX_EPOCH,
            vllm: VllmRawMetrics {
                prompt_tokens_mean: Some(4096.0),
                generation_tokens_per_sec: Some(10.0),
                ..Default::default()
            },
            gpus: vec![GpuRawMetrics::default()],

            host_memory: None,
        });
        let b = compute(&AnalysisInput::new(&ctx, &win)).expect("baseline");
        let expected = math::prefill_ops_per_sec(67.0, 37_000_000_000, 4096, 61, 139_264);
        assert!((b.prefill.expect("prefill").expected - expected).abs() < 1e-6);
    }

    #[test]
    fn prefill_ceiling_falls_back_to_linear_when_hidden_dim_and_coeff_both_none() {
        let cfg = VllmConfig {
            max_model_len: Some(1024),
            ..Default::default()
        };
        let snap = VllmRawMetrics {
            generation_tokens_per_sec: Some(10.0),
            ..Default::default()
        };
        let (ctx, win) = baseline_input(
            Some(8_000_000_000),
            None,
            None,
            Some(67.0),
            Some(3350.0),
            cfg,
            snap,
        );
        let b = compute(&AnalysisInput::new(&ctx, &win)).expect("baseline");
        let expected = math::prefill_ops_per_sec(67.0, 8_000_000_000, 1024, 0, 0);
        assert!((b.prefill.expect("prefill").expected - expected).abs() < 1e-6);
    }

    fn active_snap(tps: Option<f64>, running: Option<f64>, tpot_ms: Option<f64>) -> VllmRawMetrics {
        let run = running.or(Some(4.0));
        VllmRawMetrics {
            window_duration_secs: Some(2.0),
            num_requests_running: run,
            // Steady fixtures: mean = peak = landing. Drain cases set fields apart.
            num_requests_running_mean: run,
            num_requests_running_peak: run,
            generation_tokens_per_sec: tps.or(Some(100.0)),
            tpot_ms,
            ..Default::default()
        }
    }

    #[test]
    fn spec_guard_below_ceiling_no_flag_efficiency_present() {
        let (ctx, win) = baseline_input(
            Some(8_000_000_000),
            None,
            Some("bf16"),
            Some(312.0),
            Some(2039.0),
            VllmConfig {
                max_num_seqs: Some(32),
                ..Default::default()
            },
            active_snap(Some(100.0), Some(4.0), Some(50.0)),
        );
        let b = compute(&AnalysisInput::new(&ctx, &win)).expect("baseline");
        assert!(b.spec_suspected.is_none());
        assert!(b.efficiency_pct.is_some());
        assert!(b.headroom_pct.is_some());
    }

    #[test]
    fn spec_guard_d1_tpot_clears_efficiency() {
        let (ctx, win) = baseline_input(
            Some(8_000_000_000),
            None,
            Some("bf16"),
            Some(312.0),
            Some(2039.0),
            VllmConfig::default(),
            active_snap(Some(100.0), Some(4.0), Some(0.01)),
        );
        let b = compute(&AnalysisInput::new(&ctx, &win)).expect("baseline");
        let ev = b.spec_suspected.expect("flag");
        assert_eq!(ev.detector, SpecDetector::Tpot);
        assert!(b.efficiency_pct.is_none());
        assert!(b.config_relative_efficiency_pct.is_none());
        assert!(b.headroom_pct.is_none());
    }

    #[test]
    fn spec_guard_d2_per_stream_clears_efficiency() {
        let (ctx, win) = baseline_input(
            Some(8_000_000_000),
            None,
            Some("bf16"),
            Some(312.0),
            Some(2039.0),
            VllmConfig::default(),
            // Slow TPOT so D1 does not fire; per-stream rate above decode.upper.
            active_snap(Some(50_000.0), Some(1.0), Some(100.0)),
        );
        let b = compute(&AnalysisInput::new(&ctx, &win)).expect("baseline");
        let ev = b.spec_suspected.expect("flag");
        assert_eq!(ev.detector, SpecDetector::PerStream);
        assert!(b.efficiency_pct.is_none());
    }

    #[test]
    fn spec_guard_d3_absolute_clears_efficiency() {
        let (ctx, win) = baseline_input(
            Some(8_000_000_000),
            None,
            Some("bf16"),
            Some(312.0),
            Some(2039.0),
            VllmConfig::default(),
            active_snap(None, Some(4.0), Some(50.0)),
        );
        let probe = compute(&AnalysisInput::new(&ctx, &win)).expect("probe");
        let abs = probe.decode.upper * probe.ridge_batch_size * 1.1;
        // Keep per-stream under upper so D2 does not win preference over D3.
        let running = (abs / probe.decode.upper).ceil() + 10.0;
        let (ctx, win) = baseline_input(
            Some(8_000_000_000),
            None,
            Some("bf16"),
            Some(312.0),
            Some(2039.0),
            VllmConfig::default(),
            active_snap(Some(abs), Some(running), Some(100.0)),
        );
        let b = compute(&AnalysisInput::new(&ctx, &win)).expect("baseline");
        let ev = b.spec_suspected.expect("flag");
        assert_eq!(ev.detector, SpecDetector::Absolute);
        assert!(b.efficiency_pct.is_none());
    }

    #[test]
    fn spec_guard_inside_estimate_band_clamps_no_flag() {
        let (ctx, win) = baseline_input(
            Some(8_000_000_000),
            None,
            Some("bf16"),
            Some(312.0),
            Some(2039.0),
            VllmConfig {
                max_num_seqs: Some(256),
                ..Default::default()
            },
            active_snap(None, Some(4.0), Some(50.0)),
        );
        let probe = compute(&AnalysisInput::new(&ctx, &win)).expect("probe");
        // Just below upper×ridge so D3 does not fire; above expected×ridge → raw > 100.
        let target = probe.decode.expected * probe.ridge_batch_size * 1.02;
        assert!(target <= probe.decode.upper * probe.ridge_batch_size);
        // High concurrency: per-stream stays under upper (D2 silent). Slow TPOT (D1 silent).
        let running = (target / probe.decode.upper).ceil() + 10.0;
        let (ctx, win) = baseline_input(
            Some(8_000_000_000),
            None,
            Some("bf16"),
            Some(312.0),
            Some(2039.0),
            VllmConfig {
                max_num_seqs: Some(256),
                ..Default::default()
            },
            active_snap(Some(target), Some(running), Some(50.0)),
        );
        let b = compute(&AnalysisInput::new(&ctx, &win)).expect("baseline");
        assert!(b.spec_suspected.is_none(), "inside band must not flag");
        assert_eq!(b.efficiency_pct, Some(100.0));
    }

    #[test]
    fn spec_guard_unknown_gpu_no_baseline() {
        let (ctx, win) = baseline_input(
            Some(8_000_000_000),
            None,
            Some("bf16"),
            None,
            None,
            VllmConfig::default(),
            active_snap(Some(1_000_000.0), Some(1.0), Some(0.01)),
        );
        assert!(compute(&AnalysisInput::new(&ctx, &win)).is_none());
    }

    #[test]
    fn spec_guard_idle_window_no_flag() {
        let (ctx, win) = baseline_input(
            Some(8_000_000_000),
            None,
            Some("bf16"),
            Some(312.0),
            Some(2039.0),
            VllmConfig::default(),
            VllmRawMetrics {
                window_duration_secs: Some(2.0),
                num_requests_running: Some(0.0),
                generation_tokens_per_sec: Some(0.0),
                tpot_ms: Some(0.01),
                ..Default::default()
            },
        );
        let b = compute(&AnalysisInput::new(&ctx, &win)).expect("baseline");
        assert!(b.spec_suspected.is_none());
    }

    #[test]
    fn spec_guard_missing_tpot_still_evaluates_d2() {
        let (ctx, win) = baseline_input(
            Some(8_000_000_000),
            None,
            Some("bf16"),
            Some(312.0),
            Some(2039.0),
            VllmConfig::default(),
            active_snap(Some(50_000.0), Some(1.0), None),
        );
        let b = compute(&AnalysisInput::new(&ctx, &win)).expect("baseline");
        let ev = b.spec_suspected.expect("flag");
        assert_eq!(ev.detector, SpecDetector::PerStream);
    }

    #[test]
    fn spec_guard_d2_uses_mean_running_not_drained_landing() {
        // Drain: landing=1, mean stays honest for the window. tok/s ÷ landing would
        // false-fire; ÷ mean must not.
        let (ctx, win) = baseline_input(
            Some(8_000_000_000),
            None,
            Some("bf16"),
            Some(312.0),
            Some(2039.0),
            VllmConfig::default(),
            active_snap(None, Some(1.0), Some(100.0)),
        );
        let probe = compute(&AnalysisInput::new(&ctx, &win)).expect("probe");
        let upper = probe.decode.upper;
        let mean = 40.0;
        let tps = upper * 10.0;
        assert!(tps / 1.0 > upper);
        assert!(tps / mean <= upper);

        let mut snap = active_snap(Some(tps), Some(1.0), Some(100.0));
        snap.num_requests_running = Some(1.0);
        snap.num_requests_running_mean = Some(mean);
        snap.num_requests_running_peak = Some(80.0);
        let (ctx, win) = baseline_input(
            Some(8_000_000_000),
            None,
            Some("bf16"),
            Some(312.0),
            Some(2039.0),
            VllmConfig {
                max_num_seqs: Some(256),
                ..Default::default()
            },
            snap,
        );
        let b = compute(&AnalysisInput::new(&ctx, &win)).expect("baseline");
        assert!(
            b.spec_suspected.is_none(),
            "mean denom must block drain false-fire"
        );
        assert!(b.efficiency_pct.is_some());
    }

    #[test]
    fn spec_guard_d2_falls_back_to_peak_when_mean_unread() {
        let mut snap = active_snap(Some(50_000.0), Some(1.0), Some(100.0));
        snap.num_requests_running_mean = None;
        snap.num_requests_running_peak = Some(1.0);
        snap.num_requests_running = Some(1.0);
        let (ctx, win) = baseline_input(
            Some(8_000_000_000),
            None,
            Some("bf16"),
            Some(312.0),
            Some(2039.0),
            VllmConfig::default(),
            snap,
        );
        let b = compute(&AnalysisInput::new(&ctx, &win)).expect("baseline");
        let ev = b.spec_suspected.expect("peak fallback");
        assert_eq!(ev.detector, SpecDetector::PerStream);
    }

    #[test]
    fn spec_guard_d2_never_uses_landing_alone() {
        // Landing would fire; mean and peak unread → D2 silent (D3 may still fire).
        let (ctx, win) = baseline_input(
            Some(8_000_000_000),
            None,
            Some("bf16"),
            Some(312.0),
            Some(2039.0),
            VllmConfig::default(),
            active_snap(None, Some(1.0), Some(100.0)),
        );
        let probe = compute(&AnalysisInput::new(&ctx, &win)).expect("probe");
        let upper = probe.decode.upper;
        // Keep absolute under the ridge product so only D2 could have fired via landing.
        let tps = upper * 2.0;
        assert!(tps / 1.0 > upper);
        assert!(tps < upper * probe.ridge_batch_size);

        let mut snap = active_snap(Some(tps), Some(1.0), Some(100.0));
        snap.num_requests_running = Some(1.0);
        snap.num_requests_running_mean = None;
        snap.num_requests_running_peak = None;
        let (ctx, win) = baseline_input(
            Some(8_000_000_000),
            None,
            Some("bf16"),
            Some(312.0),
            Some(2039.0),
            VllmConfig {
                max_num_seqs: Some(256),
                ..Default::default()
            },
            snap,
        );
        let b = compute(&AnalysisInput::new(&ctx, &win)).expect("baseline");
        assert!(
            b.spec_suspected
                .as_ref()
                .is_none_or(|e| e.detector != SpecDetector::PerStream),
            "landing-only must not trip D2: {:?}",
            b.spec_suspected
        );
        assert!(
            b.spec_suspected.is_none(),
            "fixture must stay under D3 as well: {:?}",
            b.spec_suspected
        );
        assert!(b.efficiency_pct.is_some());
    }

    #[test]
    fn apply_spec_run_or_poisons_summary_efficiency() {
        let (ctx, win) = baseline_input(
            Some(8_000_000_000),
            None,
            Some("bf16"),
            Some(312.0),
            Some(2039.0),
            VllmConfig {
                max_num_seqs: Some(32),
                ..Default::default()
            },
            active_snap(Some(100.0), Some(4.0), Some(50.0)),
        );
        let mut baseline = compute(&AnalysisInput::new(&ctx, &win));
        assert!(baseline.as_ref().unwrap().efficiency_pct.is_some());
        let ev = SpecEvidence {
            detector: SpecDetector::Tpot,
            measured: 1.0,
            bound: 10.0,
        };
        apply_spec_run_or(&mut baseline, 3, 12, Some(ev));
        let b = baseline.expect("baseline");
        assert!(b.efficiency_pct.is_none());
        assert_eq!(b.spec_suspected, Some(ev));
        assert_eq!(b.spec_window_counts, Some((3, 12)));
    }
}
