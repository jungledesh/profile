use std::fmt::Write;
use std::time::Duration;
use std::time::SystemTime;

use chrono::{DateTime, Utc};

use crate::collectors::{
    GpuRawMetrics, RawSnapshot, VllmConfig, VllmRawMetrics, snapshot_uses_display_names,
    snapshot_uses_index_only,
};
use crate::context::{AnalysisInput, RuntimeWindow};
use crate::engine;
use crate::fmt::{fmt_seconds_from_ms, fmt_seconds_from_ms_maybe_floor};
use crate::profiler::DiagnoseResult;

const VLLM_LABEL_W: usize = 20;
/// Width for GPU section labels: wide enough for `GPU [xxxxxxxx]` (14 chars) plus breathing room.
const GPU_LABEL_W: usize = 20;
const VLLM_LABEL_METRICS_GAP: &str = " ";
/// Show the VRAM peak parenthetical only when it crosses 90% of total VRAM - suppresses noise
/// from tiny fluctuations that are irrelevant to the operator.
const VRAM_PEAK_SHOW_THRESHOLD_FRAC: f64 = 0.90;

/// Global GPU temp parenthetical until per-arch throttle thresholds exist (Hopper ~83°C).
const GPU_TEMP_PEAK_SHOW_THRESHOLD_C: f64 = 80.0;

#[inline]
fn show_kv_cache_peak_parenthetical(avg_pct: f64, peak_pct: f64) -> bool {
    peak_pct > avg_pct + 10.0 || peak_pct >= 95.0
}

#[inline]
fn show_vram_peak_parenthetical(used_mb: u64, peak_mb: u64, total_mb: u64) -> bool {
    peak_mb > used_mb && (peak_mb as f64 / total_mb as f64) >= VRAM_PEAK_SHOW_THRESHOLD_FRAC
}

#[inline]
fn show_gpu_temp_peak_parenthetical(current_c: f64, peak_c: f64) -> bool {
    peak_c > current_c && peak_c >= GPU_TEMP_PEAK_SHOW_THRESHOLD_C
}

/// Print using a pre-computed report. Use when the caller already has the report.
pub fn print_diagnose_table_with_report(
    result: &DiagnoseResult,
    report: &engine::Report,
    aggregate_win: &RuntimeWindow,
    verbose_rules: bool,
) {
    let lines = build_diagnose_lines(result, aggregate_win, report, verbose_rules);
    print_boxed(&lines);
    if let Some(j) = journey_line(report) {
        println!();
        println!("{j}");
    }
}

fn build_diagnose_lines(
    result: &DiagnoseResult,
    aggregate_win: &RuntimeWindow,
    report: &engine::Report,
    verbose_rules: bool,
) -> Vec<String> {
    let snapshot = &result.snapshot;
    let v = &snapshot.vllm;
    let n_gpus = snapshot.gpus.len();
    let agg = snapshot.aggregate_gpu();
    let cluster_gpu = aggregate_to_display_gpu(&agg);
    let duration = result.duration;
    let started_at = result.started_at;

    let model = v.model_name.as_deref().unwrap_or("(unknown model)");
    let gpu_label = if n_gpus > 1 {
        agg.gpu_name
            .as_deref()
            .map(|n| format!("{n} x{n_gpus}"))
            .unwrap_or_else(|| format!("{n_gpus} GPUs"))
    } else {
        snapshot
            .gpus
            .first()
            .and_then(|g| g.gpu_name.as_deref())
            .or(agg.gpu_name.as_deref())
            .unwrap_or("(no GPU)")
            .to_string()
    };
    let ts = format_profile_timestamp(started_at);
    let mut lines = vec![profile_header_line(
        env!("CARGO_PKG_VERSION"),
        model,
        &gpu_label,
        &ts,
        duration,
    )];

    push_gpu_advisories(&mut lines, snapshot);

    let aggregated_prefix_hit_rate =
        engine::aggregate_prefix_hit_rate_for_diagnose(&result.windows);
    if verbose_rules {
        lines.push(String::new());
        lines.extend(baseline_lines(
            report.baseline,
            aggregate_win.snapshot.vllm.num_requests_running,
            aggregate_win.snapshot.vllm.cache_config.num_gpu_blocks,
        ));
        lines.push(String::new());
    }

    if !result.any_evaluable || result.all_idle {
        lines.push(String::new());
        lines.push(vllm_label_row("Target:", &result.metrics_input));
        lines.push(String::new());
        let hint = engine::LoadHintParams {
            model_name: result.snapshot.vllm.model_name.as_deref(),
            metrics_url: &result.metrics_input,
            max_num_seqs: result.static_ctx.config.max_num_seqs,
            duration_secs: result.duration.as_secs(),
        };
        lines.extend(engine::empty_run_diagnose_lines(
            verbose_rules,
            &result.windows,
            result.any_evaluable,
            &hint,
            &result.metrics_input,
        ));
        return lines;
    }

    if !verbose_rules {
        lines.extend(quiet_efficiency_fallback_lines(report.baseline.as_ref()));
        lines.push(String::new());
    }
    lines.push(format!(
        "{:<width$}{}{}",
        "GPU =>",
        VLLM_LABEL_METRICS_GAP,
        gpu_gauges_line(
            &cluster_gpu,
            report.baseline.as_ref(),
            v.generation_tokens_per_sec,
            verbose_rules,
        ),
        width = GPU_LABEL_W
    ));
    if verbose_rules && report.energy_skew_skipped > 0 {
        lines.push(format!(
            "energy: skipped {} windows (observation skew).",
            report.energy_skew_skipped
        ));
    }
    if n_gpus <= 1 {
        let g = snapshot.gpus.first().unwrap_or(&cluster_gpu);
        lines.push(format!(
            "{:<width$}{}{}",
            "",
            VLLM_LABEL_METRICS_GAP,
            gpu_detail_line(g, verbose_rules, false),
            width = GPU_LABEL_W
        ));
    } else {
        lines.push(String::new());
        for gpu in &snapshot.gpus {
            lines.push(format!(
                "{:<width$}{}{}",
                format!("GPU [{}]", gpu.display_id()),
                VLLM_LABEL_METRICS_GAP,
                gpu_detail_line(gpu, verbose_rules, true),
                width = GPU_LABEL_W
            ));
        }
    }
    lines.push(String::new());
    lines.push(vllm_label_row("vLLM =>", ""));
    lines.push(vllm_label_row(
        "REQUESTS",
        &vllm_requests_value(v, result.static_ctx.config.max_num_seqs),
    ));
    lines.push(vllm_label_row(
        "LATENCY",
        &vllm_latency_value(v, verbose_rules),
    ));
    lines.push(vllm_label_row(
        "CACHE",
        &vllm_prompt_value(v, verbose_rules, aggregated_prefix_hit_rate),
    ));
    lines.push(vllm_label_row("THROUGHPUT", &vllm_throughput_value(v)));
    lines.push(vllm_label_row("TRAFFIC", &vllm_traffic_value(v)));
    if verbose_rules {
        lines.push(vllm_label_row("MEMORY", &vllm_memory_value(v)));
        lines.push(vllm_label_row("CACHE CFG", &vllm_cache_cfg_value(v)));
        lines.push(String::new());
        lines.push(vllm_label_row("Config:", ""));
        let cfg = &result.static_ctx.config;
        lines.push(vllm_label_row("PARALLEL", &config_parallel_value(cfg)));
        lines.push(vllm_label_row("MODEL", &config_model_value(cfg)));
        lines.push(vllm_label_row("KV", &config_kv_value(cfg)));
    }

    let summary_input = AnalysisInput::new(&result.static_ctx, aggregate_win);
    let rule_lines = engine::format_diagnose_rules_for_windows(
        &result.windows,
        summary_input,
        report,
        verbose_rules,
        &result.metrics_input,
        result.duration.as_secs(),
    );
    if !rule_lines.is_empty() {
        lines.push(String::new());
        lines.push("ISSUES:".to_string());
        lines.push(String::new());
        lines.extend(rule_lines);
    }

    lines
}

/// Action line: printed below the box, not inside it.
/// Returns None when no issues fired (clean run or not evaluable).
fn journey_line(report: &engine::Report) -> Option<&'static str> {
    // Same sparse gate as the formatter ISSUES branch: never prompt "apply the fix"
    // when recommendations are hidden behind Insufficient Sustained Load.
    if report.n_eval < engine::ENGINE_MIN_PERSISTENT_WINDOWS {
        return None;
    }
    if report.recommendations.is_empty() {
        return None;
    }
    Some("▶  Apply the fix above. Profile re-measures after your change.")
}

fn push_gpu_advisories(lines: &mut Vec<String>, snapshot: &RawSnapshot) {
    if snapshot_uses_display_names(&snapshot.gpus) {
        lines.push(
            "[i] GPU identity unavailable. Assigning display names. Keep GPU device ordering stable across runs."
                .to_string(),
        );
    } else if snapshot_uses_index_only(&snapshot.gpus) {
        lines.push(
            "[i] GPU identity: device index only. Keep GPU device ordering stable across runs."
                .to_string(),
        );
    }
}

fn aggregate_to_display_gpu(agg: &crate::collectors::AggregateGpuMetrics) -> GpuRawMetrics {
    GpuRawMetrics {
        gpu_name: agg.gpu_name.clone(),
        gpu_util_pct: agg.gpu_util_pct,
        mem_util_pct: agg.mem_util_pct,
        power_watts: agg.power_watts,
        vram_used_mb: agg.vram_used_mb,
        vram_peak_mb: agg.vram_peak_mb,
        vram_total_mb: agg.sum_vram_total_mb,
        temperature_peak_c: agg.temperature_peak_c,
        ..Default::default()
    }
}

fn format_profile_timestamp(st: SystemTime) -> String {
    let utc: DateTime<Utc> = st.into();
    utc.format("%Y-%m-%d %H:%M:%S UTC").to_string()
}

fn profile_header_line(
    version: &str,
    model: &str,
    gpu: &str,
    ts: &str,
    duration: Duration,
) -> String {
    let suffix = if duration > Duration::from_secs(2) {
        format!("({} from {ts})", duration_short(duration))
    } else {
        format!("[{ts}]")
    };
    [
        format!("PROFILE v{version}"),
        format!("[{model}]"),
        format!("[{gpu}]"),
        suffix,
    ]
    .join(" ")
}

fn duration_short(duration: Duration) -> String {
    let secs = duration.as_secs();
    if secs.is_multiple_of(60) {
        format!("{}m", secs / 60)
    } else {
        format!("{secs}s")
    }
}

fn quiet_efficiency_fallback_lines(baseline: Option<&engine::PhysicsBaseline>) -> Vec<String> {
    let Some(b) = baseline else {
        return Vec::new();
    };
    if b.weight_dtype_source != engine::WeightDtypeSource::Fallback {
        return Vec::new();
    }
    let Some(e) = b.efficiency_pct.filter(|e| e.is_finite()) else {
        return Vec::new();
    };
    vec![
        vllm_label_row("Efficiency", &format!("~{e:.1}% of ceiling (est)")),
        format!(
            "{:<width$}{}{}",
            "",
            VLLM_LABEL_METRICS_GAP,
            "Note: weight dtype not reported; ceiling assumes bf16.",
            width = GPU_LABEL_W
        ),
        String::new(),
    ]
}

fn weight_dtype_display(source: engine::WeightDtypeSource, weight_gb: f64) -> String {
    let suffix = match source {
        engine::WeightDtypeSource::VllmInfoQuantization => "vLLM /info (quant)",
        engine::WeightDtypeSource::EnvVarQuantization => "env (quant)",
        engine::WeightDtypeSource::VllmConfig => "vLLM",
        engine::WeightDtypeSource::VllmInfoEndpoint => "vLLM /info",
        engine::WeightDtypeSource::EnvVar => "env",
        engine::WeightDtypeSource::Catalog => "catalog",
        engine::WeightDtypeSource::Fallback => "assumed bf16",
    };
    format!("weight {:.0}GB ({})", weight_gb, suffix)
}

fn baseline_lines(
    baseline: Option<engine::PhysicsBaseline>,
    num_requests_running: Option<f64>,
    num_gpu_blocks: Option<u32>,
) -> Vec<String> {
    let Some(b) = baseline else {
        return vec![format!(
            "{:<width$}{}{}",
            "HW LIMITS",
            VLLM_LABEL_METRICS_GAP,
            "unavailable (model not recognized). Add model to catalog",
            width = GPU_LABEL_W
        )];
    };

    // ~ marks values derived from estimated ceilings. Measured values carry no tilde.
    // Keep this distinction strict.
    // Line 1: efficiency + throughput ceilings
    let mut seg1 = Vec::new();
    let eff = match b.efficiency_pct {
        Some(e) => format!("~{e:.1}%"),
        None => "-".to_string(),
    };
    seg1.push(format!("decode_eff {eff}"));
    if b.decode.expected >= 0.5 {
        seg1.push(format!("decode ~{:.0} tok/s (est)", b.decode.expected));
    }
    if let Some(prefill) = b.prefill
        && prefill.expected >= 10.0
    {
        seg1.push(format!("prefill ~{:.0} prompts/s (est)", prefill.expected));
    }

    // Line 2: memory budget + latency floors
    let mut seg2 = Vec::new();
    seg2.push(weight_dtype_display(b.weight_dtype_source, b.weight_gb));
    if let Some(blocks) = num_gpu_blocks {
        seg2.push(format!("kv_blocks {blocks}"));
    } else if let Some(headroom) = b.kv_headroom_gb
        && headroom < 0.0
    {
        seg2.push(format!("kv_headroom {:.0}GB (needs TP)", headroom));
    }
    seg2.push(format!("tpot_floor ~{:.1}ms", b.tpot_floor_ms));
    if let Some(pf) = b.prefill_latency_floor_ms {
        let compute_bound = num_requests_running
            .filter(|x| x.is_finite())
            .is_some_and(|n| n >= b.ridge_batch_size);
        let prefill_ceiling_meaningful = b.prefill.is_some_and(|p| p.expected >= 10.0);
        if pf >= 0.5 && !compute_bound && prefill_ceiling_meaningful {
            seg2.push(format!("prefill_floor ~{:.0}ms", pf));
        }
    }

    let mut out = vec![
        format!(
            "{:<width$}{}{}",
            "HW LIMITS",
            VLLM_LABEL_METRICS_GAP,
            seg1.join(" | "),
            width = GPU_LABEL_W
        ),
        format!(
            "{:<width$}{}{}",
            "",
            VLLM_LABEL_METRICS_GAP,
            seg2.join(" | "),
            width = GPU_LABEL_W
        ),
    ];
    if b.weight_dtype_source == engine::WeightDtypeSource::Fallback {
        out.push(format!(
            "{:<width$}{}{}",
            "",
            VLLM_LABEL_METRICS_GAP,
            "weight dtype assumed bf16. Confirm via vLLM metrics or DTYPE env var",
            width = GPU_LABEL_W
        ));
    }
    if b.kv_cache_dtype_source == engine::KvCacheDtypeSource::Unknown {
        out.push(format!(
            "{:<width$}{}{}",
            "",
            VLLM_LABEL_METRICS_GAP,
            "kv_cache_dtype unrecognized; priced as bf16 activation (2 bytes/element)",
            width = GPU_LABEL_W
        ));
    }
    out
}

fn vllm_label_row(label: &str, value: &str) -> String {
    format!(
        "{:<width$}{}{}",
        label,
        VLLM_LABEL_METRICS_GAP,
        value,
        width = VLLM_LABEL_W
    )
}

fn print_boxed(lines: &[String]) {
    let inner = lines.iter().map(|l| l.chars().count()).max().unwrap_or(0);
    let border = format!("+{}+", "-".repeat(inner));
    println!("{}", border);
    for line in lines {
        let w = line.chars().count();
        let padded = if w < inner {
            format!("{}{}", line, " ".repeat(inner - w))
        } else {
            line.clone()
        };
        println!("|{}|", padded);
    }
    println!("{}", border);
}

// gpu_util_pct and mem_util_pct are intentionally absent here and from all top-level display.
// GPU SM util reports "active" regardless of useful work (spin-locks, graph capture, async polling
// make 99% util compatible with near-zero MFU). Efficiency % is the honest saturation signal.
fn gpu_gauges_line(
    g: &GpuRawMetrics,
    baseline: Option<&crate::engine::PhysicsBaseline>,
    actual_tps: Option<f64>,
    verbose: bool,
) -> String {
    let efficiency = format_efficiency_label(
        baseline.and_then(|b| b.efficiency_pct),
        actual_tps,
        baseline.map(|b| b.decode.expected),
    );

    let power = g
        .power_watts
        .map(|draw| format!("power {:.0}W", draw))
        .unwrap_or_else(|| "power -".to_string());

    let mut segments = vec![efficiency, power];

    if let Some(cost) = baseline.and_then(|b| b.cost.as_ref()) {
        if let Some(jtok) = cost.joules_per_token.filter(|v| v.is_finite() && *v > 0.0) {
            segments.push(format!("{:.2} J/tok", jtok));
        }
        if verbose && let Some(tpw) = cost.tok_per_watt.filter(|v| v.is_finite() && *v > 0.0) {
            segments.push(format!("{:.1} tok/W", tpw));
        }
        if let Some(cpm) = cost
            .cost_per_million_tokens
            .filter(|v| v.is_finite() && *v > 0.0)
        {
            let label = match cost.cost_source {
                engine::CostSource::Catalog => format!("${:.2}/1M tok (est)", cpm),
                engine::CostSource::UserProvided => format!("${:.2}/1M tok", cpm),
            };
            if !label.is_empty() {
                segments.push(label);
            }
        }
    }

    let mem = format_vram(g);

    segments.push(mem);
    segments.join(" | ")
}

fn format_vram(g: &GpuRawMetrics) -> String {
    match (g.vram_used_mb, g.vram_total_mb) {
        (Some(used), Some(total)) if total > 0 => {
            // Binary GB (MiB/1024) for operator display so an 80 GiB card reads
            // 80GB. Physics headroom uses mib_to_decimal_gb; do not feed this
            // display path into weight/headroom math.
            let u_gb = used as f64 / 1024.0;
            let t_gb = total as f64 / 1024.0;
            let mut s = format!("vRAM {:.0}/{:.0}GB", u_gb, t_gb);
            if let Some(pk) = g.vram_peak_mb
                && show_vram_peak_parenthetical(used, pk, total)
            {
                let pk_gb = pk as f64 / 1024.0;
                let _ = write!(s, " (peak {:.0}GB)", pk_gb);
            }
            s
        }
        _ => "vRAM -".to_string(),
    }
}

fn format_efficiency_label(
    efficiency_pct: Option<f64>,
    actual_tps: Option<f64>,
    decode_ceiling: Option<f64>,
) -> String {
    // ~ marks values derived from estimated ceilings. Measured values carry no tilde.
    // Keep this distinction strict.
    if let Some(e) = efficiency_pct.filter(|e| e.is_finite()) {
        return format!("decode_eff ~{:.1}%", e);
    }
    let actual = actual_tps.filter(|t| t.is_finite() && *t > 0.0);
    let ceiling = decode_ceiling.filter(|c| c.is_finite() && *c > 0.0);
    if actual.is_some() && ceiling.is_some() {
        "decode_eff ?".to_string()
    } else {
        "decode_eff -".to_string()
    }
}

fn vllm_requests_value(v: &VllmRawMetrics, config_max_num_seqs: Option<u32>) -> String {
    let max_n = v.max_num_seqs.or(config_max_num_seqs).filter(|&m| m > 0);
    let run = match v.num_requests_running.filter(|x| x.is_finite()) {
        Some(avg) => {
            let rounded = avg.round();
            if let Some(max_n) = max_n {
                let pct = (avg / f64::from(max_n)) * 100.0;
                format!("run {:.0} ({:.1}%)", rounded, pct)
            } else {
                format!("run {:.0}", rounded)
            }
        }
        None => "run -".to_string(),
    };
    let wait = match v.num_requests_waiting.filter(|x| x.is_finite()) {
        Some(w) => format!("wait {:.0}", w.round()),
        None => "wait -".to_string(),
    };
    let max_seq = max_n
        .map(|n| format!("max {n}"))
        .unwrap_or_else(|| "max -".to_string());

    format!("{run} | {wait} | {max_seq}")
}

fn fmt_gauge(x: f64) -> String {
    if (x - x.round()).abs() < 1e-6 {
        format!("{:.0}", x)
    } else {
        format!("{:.1}", x)
    }
}

fn vllm_latency_value(v: &VllmRawMetrics, verbose: bool) -> String {
    let ttft = v
        .ttft_ms
        .map(fmt_seconds_from_ms)
        .unwrap_or_else(|| "-".to_string());
    let tpot = v
        .tpot_ms
        .map(fmt_seconds_from_ms)
        .unwrap_or_else(|| "-".to_string());
    let ttft_p95 = v
        .ttft_p95_ms
        .map(|p| {
            format!(
                " (p95 {})",
                fmt_seconds_from_ms_maybe_floor(p, v.ttft_p95_clamped)
            )
        })
        .unwrap_or_default();
    let tpot_p95 = v
        .tpot_p95_ms
        .map(|p| {
            format!(
                " (p95 {})",
                fmt_seconds_from_ms_maybe_floor(p, v.tpot_p95_clamped)
            )
        })
        .unwrap_or_default();

    if !verbose {
        return format!("ttft {ttft}{ttft_p95} | tpot {tpot}{tpot_p95}");
    }

    let prefill = v
        .prefill_latency_ms
        .map(fmt_seconds_from_ms)
        .unwrap_or_else(|| "-".to_string());
    let queue = v
        .queue_delay_ms
        .map(fmt_seconds_from_ms)
        .unwrap_or_else(|| "-".to_string());

    format!("ttft {ttft}{ttft_p95} | tpot {tpot}{tpot_p95} | prefill {prefill} | queue {queue}")
}

fn vllm_prompt_kv_fragment(v: &VllmRawMetrics) -> String {
    match v.kv_cache_usage_perc.filter(|x| x.is_finite()) {
        Some(avg) => {
            let mut s = format!("kv_cache {:.1}% avg", avg);
            if let Some(pk) = v.kv_cache_peak_perc.filter(|x| x.is_finite())
                && show_kv_cache_peak_parenthetical(avg, pk)
            {
                let _ = write!(s, " ({:.1}% peak)", pk);
            }
            s
        }
        None => "kv_cache -".to_string(),
    }
}

fn vllm_prompt_value(v: &VllmRawMetrics, verbose: bool, prefix_hit_rate: Option<f64>) -> String {
    let kv = vllm_prompt_kv_fragment(v);
    let cache = cache_use_fragment(prefix_hit_rate);
    if !verbose {
        return format!("{kv} | {cache}");
    }
    let n = v
        .prompt_tokens_mean
        .map(fmt_tok)
        .unwrap_or_else(|| "-".to_string());
    format!("{n} tok | {kv} | {cache}")
}

fn fmt_tok(t: f64) -> String {
    if (t - t.round()).abs() < 1e-6 {
        format!("{:.0}", t)
    } else {
        format!("{:.1}", t)
    }
}

fn vllm_throughput_value(v: &VllmRawMetrics) -> String {
    v.generation_tokens_per_sec
        .map(|t| format!("{:.0} tok/s", t))
        .unwrap_or_else(|| "- tok/s".to_string())
}

fn cache_use_fragment(prefix_hit_rate: Option<f64>) -> String {
    match prefix_hit_rate {
        Some(0.0) => "pfix_cache 0%".to_string(),
        Some(r) => format!("pfix_cache {:.1}%", r * 100.0),
        None => "pfix_cache -".to_string(),
    }
}

fn gpu_detail_line(g: &GpuRawMetrics, verbose: bool, is_multi_gpu: bool) -> String {
    let mut base = String::new();

    if is_multi_gpu {
        let vram = format_vram(g);
        let power = g
            .power_watts
            .map(|w| format!("power {:.0}W", w))
            .unwrap_or_else(|| "power -".to_string());
        let _ = write!(base, "{vram} | {power} | ");
    }

    let mem_util = g
        .mem_util_pct
        .map(|u| format!("mem_util {:.0}%", u))
        .unwrap_or_else(|| "mem_util -".to_string());

    base.push_str(&mem_util);

    if !verbose {
        return base;
    }

    let temp = match g.temperature_c.filter(|t| t.is_finite()) {
        Some(cur) => {
            let mut s = format!("temp {:.0}°C", cur);
            if let Some(pk) = g.temperature_peak_c.filter(|t| t.is_finite())
                && show_gpu_temp_peak_parenthetical(cur, pk)
            {
                let _ = write!(s, " (peak {:.0}°C)", pk);
            }
            s
        }
        None => "temp -".to_string(),
    };
    let sm = g
        .sm_clock_mhz
        .map(|c| format!("sm {}MHz", c))
        .unwrap_or_else(|| "sm -".to_string());
    let limit = g
        .power_limit_watts
        .map(|l| format!("limit {:.0}W", l))
        .unwrap_or_else(|| "limit -".to_string());

    format!("{base} | {temp} | {sm} | {limit}")
}

fn vllm_memory_value(v: &VllmRawMetrics) -> String {
    let swapped = v
        .num_requests_swapped
        .map(fmt_gauge)
        .map(|s| format!("swapped {s}"))
        .unwrap_or_else(|| "swapped -".to_string());
    let cpu_cache = match v.cpu_cache_usage_perc.filter(|x| x.is_finite()) {
        Some(p) => format!("cpu_cache {:.1}%", p),
        None => "cpu_cache -".to_string(),
    };
    format!("{swapped} | {cpu_cache}")
}

fn vllm_traffic_value(v: &VllmRawMetrics) -> String {
    let qps = v
        .request_success_per_sec
        .map(|q| format!("qps {:.1}", q))
        .unwrap_or_else(|| "qps -".to_string());
    let req_total = v
        .request_success_total
        .map(|t| format!("req_total {:.0}", t))
        .unwrap_or_else(|| "req_total -".to_string());
    let gen_total = v
        .generation_tokens_total
        .map(|t| format!("gen_total {:.0}", t))
        .unwrap_or_else(|| "gen_total -".to_string());
    let preempt_rate = v
        .num_preemptions_per_sec
        .map(|p| format!("preempt/s {:.2}", p))
        .unwrap_or_else(|| "preempt/s -".to_string());
    let preempt_total = v
        .num_preemptions_total
        .map(|t| format!("preempt_total {:.0}", t))
        .unwrap_or_else(|| "preempt_total -".to_string());
    format!("{qps} | {req_total} | {gen_total} | {preempt_rate} | {preempt_total}")
}

fn vllm_cache_cfg_value(v: &VllmRawMetrics) -> String {
    let block = v
        .cache_config
        .block_size
        .map(|b| format!("block {b}"))
        .unwrap_or_else(|| "block -".to_string());
    let dtype = v.cache_config.cache_dtype.as_deref().unwrap_or("-");
    let prefix = v
        .cache_config
        .enable_prefix_caching
        .map(|b| {
            if b {
                "prefix_cache on"
            } else {
                "prefix_cache off"
            }
        })
        .unwrap_or("prefix_cache unknown");
    let chunked = v
        .cache_config
        .enable_chunked_prefill
        .map(|b| {
            if b {
                "chunked_prefill on"
            } else {
                "chunked_prefill off"
            }
        })
        .unwrap_or("chunked_prefill unknown");
    format!("{block} | dtype {dtype} | {prefix} | {chunked}")
}

fn config_parallel_value(cfg: &VllmConfig) -> String {
    let tp = cfg
        .tensor_parallel_size
        .map(|v| format!("tp {v}"))
        .unwrap_or_else(|| "tp -".to_string());
    let pp = cfg
        .pipeline_parallel_size
        .map(|v| format!("pp {v}"))
        .unwrap_or_else(|| "pp -".to_string());
    format!("{tp} | {pp}")
}

fn config_model_value(cfg: &VllmConfig) -> String {
    let max_len = cfg
        .max_model_len
        .map(|v| format!("max_len {v}"))
        .unwrap_or_else(|| "max_len -".to_string());
    let dtype = cfg.dtype.as_deref().unwrap_or("-");
    let quant = cfg.quantization.as_deref().unwrap_or("-");
    let gpu_mem = cfg
        .gpu_memory_utilization
        .map(|v| format!("gpu_mem_util {:.2}", v))
        .unwrap_or_else(|| "gpu_mem_util -".to_string());
    format!("{max_len} | dtype {dtype} | quant {quant} | {gpu_mem}")
}

fn config_kv_value(cfg: &VllmConfig) -> String {
    let kv_dtype = cfg.kv_cache_dtype.as_deref().unwrap_or("-");
    let (_, kv_source) = engine::baseline::resolve_kv_cache_element(cfg.kv_cache_dtype.as_deref());
    let kv_suffix = if kv_source == engine::KvCacheDtypeSource::Unknown {
        " (priced as bf16)"
    } else {
        ""
    };
    let block = cfg
        .block_size
        .map(|b| format!("block {b}"))
        .unwrap_or_else(|| "block -".to_string());
    let prefix = cfg
        .enable_prefix_caching
        .map(|b| {
            if b {
                "prefix_cache on"
            } else {
                "prefix_cache off"
            }
        })
        .unwrap_or("prefix_cache unknown");
    let chunked = cfg
        .enable_chunked_prefill
        .map(|b| {
            if b {
                "chunked_prefill on"
            } else {
                "chunked_prefill off"
            }
        })
        .unwrap_or("chunked_prefill unknown");
    format!("dtype {kv_dtype}{kv_suffix} | {block} | {prefix} | {chunked}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collectors::{GpuRawMetrics, RawSnapshot, VllmConfig, VllmRawMetrics};
    use crate::context::{AnalysisInput, RuntimeWindow, StaticContext};
    use crate::profiler::DiagnoseResult;
    use std::time::{Duration, UNIX_EPOCH};

    fn diagnose_lines_for(result: &DiagnoseResult, verbose_rules: bool) -> Vec<String> {
        let aggregate_win = RuntimeWindow::from_snapshot(result.snapshot.clone());
        let summary_input = AnalysisInput::new(&result.static_ctx, &aggregate_win);
        let report = engine::build_report_for_diagnose(&result.windows, summary_input);
        build_diagnose_lines(result, &aggregate_win, &report, verbose_rules)
    }

    #[test]
    fn format_profile_timestamp_unix_epoch_utc() {
        assert_eq!(
            format_profile_timestamp(UNIX_EPOCH),
            "1970-01-01 00:00:00 UTC"
        );
    }

    #[test]
    fn format_profile_timestamp_known_instant() {
        let t = UNIX_EPOCH + Duration::from_secs(1_776_030_794);
        assert_eq!(format_profile_timestamp(t), "2026-04-12 21:53:14 UTC");
    }

    #[test]
    fn profile_header_line_single_space_between_segments() {
        let v = env!("CARGO_PKG_VERSION");
        assert_eq!(
            profile_header_line(
                v,
                "llama3",
                "NVIDIA H100 80GB HBM3",
                "2026-04-12 21:53:14 UTC",
                Duration::from_secs(2),
            ),
            format!("PROFILE v{v} [llama3] [NVIDIA H100 80GB HBM3] [2026-04-12 21:53:14 UTC]")
        );
    }

    #[test]
    fn profile_header_line_duration_view_includes_utc_timestamp() {
        let v = env!("CARGO_PKG_VERSION");
        assert_eq!(
            profile_header_line(
                v,
                "llama3",
                "NVIDIA H100 80GB HBM3",
                "2026-04-13 10:42:31 UTC",
                Duration::from_secs(5 * 60),
            ),
            format!(
                "PROFILE v{v} [llama3] [NVIDIA H100 80GB HBM3] (5m from 2026-04-13 10:42:31 UTC)"
            )
        );
    }

    #[test]
    fn baseline_lines_efficiency_none_renders_dash() {
        let b = engine::PhysicsBaseline {
            decode: engine::CeilingEstimate {
                lower: 85.0,
                expected: 100.0,
                upper: 105.0,
            },
            prefill: Some(engine::CeilingEstimate {
                lower: 90.0,
                expected: 50.0,
                upper: 55.0,
            }),
            efficiency_pct: None,
            headroom_pct: None,
            weight_dtype_source: engine::WeightDtypeSource::EnvVar,
            weight_gb: 16.0,
            weight_bytes_per_param: 2,
            kv_bytes_per_element: 2,
            kv_cache_dtype_source: engine::KvCacheDtypeSource::Auto,
            kv_headroom_gb: Some(8.0),
            tpot_floor_ms: 10.0,
            prefill_latency_floor_ms: Some(20.0),
            ridge_batch_size: 40.0,
            config_relative_efficiency_pct: None,
            cost: None,
        };
        let lines = baseline_lines(Some(b), None, None);
        assert_eq!(
            lines[0],
            format!(
                "{:<width$}{}{}",
                "HW LIMITS",
                VLLM_LABEL_METRICS_GAP,
                "decode_eff - | decode ~100 tok/s (est) | prefill ~50 prompts/s (est)",
                width = GPU_LABEL_W
            )
        );
        assert!(
            lines[1].contains("prefill_floor ~20ms"),
            "prefill_floor shown when efficiency invalid: {}",
            lines[1]
        );
    }

    #[test]
    fn baseline_lines_prefill_floor_hidden_when_at_or_above_ridge_running() {
        let base = || engine::PhysicsBaseline {
            decode: engine::CeilingEstimate {
                lower: 85.0,
                expected: 100.0,
                upper: 105.0,
            },
            prefill: Some(engine::CeilingEstimate {
                lower: 90.0,
                expected: 100.0,
                upper: 110.0,
            }),
            efficiency_pct: Some(50.0),
            headroom_pct: Some(50.0),
            weight_dtype_source: engine::WeightDtypeSource::EnvVar,
            weight_gb: 16.0,
            weight_bytes_per_param: 2,
            kv_bytes_per_element: 2,
            kv_cache_dtype_source: engine::KvCacheDtypeSource::Auto,
            kv_headroom_gb: Some(8.0),
            tpot_floor_ms: 10.0,
            prefill_latency_floor_ms: Some(42.0),
            ridge_batch_size: 40.0,
            config_relative_efficiency_pct: None,
            cost: None,
        };
        let above = baseline_lines(Some(base()), Some(40.0), None);
        assert!(
            above[0].contains("decode_eff ~50.0%"),
            "derived efficiency should carry estimate marker: {}",
            above[0]
        );
        assert!(
            !above[1].contains("prefill_floor"),
            "expected no prefill_floor at ridge: {}",
            above[1]
        );
        let below = baseline_lines(Some(base()), Some(39.0), None);
        assert!(
            below[1].contains("prefill_floor ~42ms"),
            "expected prefill_floor below ridge: {}",
            below[1]
        );
    }

    #[test]
    fn baseline_lines_prefill_ceiling_and_floor_suppressed_when_prefill_below_10_tok_s() {
        let b = engine::PhysicsBaseline {
            decode: engine::CeilingEstimate {
                lower: 85.0,
                expected: 100.0,
                upper: 105.0,
            },
            prefill: Some(engine::CeilingEstimate {
                lower: 4.0,
                expected: 5.0,
                upper: 6.0,
            }),
            efficiency_pct: Some(50.0),
            headroom_pct: Some(50.0),
            weight_dtype_source: engine::WeightDtypeSource::EnvVar,
            weight_gb: 16.0,
            weight_bytes_per_param: 2,
            kv_bytes_per_element: 2,
            kv_cache_dtype_source: engine::KvCacheDtypeSource::Auto,
            kv_headroom_gb: Some(8.0),
            tpot_floor_ms: 10.0,
            prefill_latency_floor_ms: Some(200.0),
            ridge_batch_size: 40.0,
            config_relative_efficiency_pct: None,
            cost: None,
        };
        let lines = baseline_lines(Some(b), Some(5.0), None);
        assert!(
            !lines[0].contains("prefill ~"),
            "line1 should omit low prefill ceiling: {}",
            lines[0]
        );
        assert!(
            !lines[1].contains("prefill_floor"),
            "line2 should omit prefill_floor when ceiling suppressed: {}",
            lines[1]
        );
    }

    #[test]
    fn baseline_lines_prefill_ceiling_and_floor_shown_when_meaningful_and_below_ridge() {
        let b = engine::PhysicsBaseline {
            decode: engine::CeilingEstimate {
                lower: 85.0,
                expected: 100.0,
                upper: 105.0,
            },
            prefill: Some(engine::CeilingEstimate {
                lower: 90.0,
                expected: 50.0,
                upper: 55.0,
            }),
            efficiency_pct: Some(50.0),
            headroom_pct: Some(50.0),
            weight_dtype_source: engine::WeightDtypeSource::EnvVar,
            weight_gb: 16.0,
            weight_bytes_per_param: 2,
            kv_bytes_per_element: 2,
            kv_cache_dtype_source: engine::KvCacheDtypeSource::Auto,
            kv_headroom_gb: Some(8.0),
            tpot_floor_ms: 10.0,
            prefill_latency_floor_ms: Some(20.0),
            ridge_batch_size: 40.0,
            config_relative_efficiency_pct: None,
            cost: None,
        };
        let lines = baseline_lines(Some(b), Some(10.0), None);
        assert!(
            lines[0].contains("prefill ~50 prompts/s (est)"),
            "line1 should include prefill ceiling: {}",
            lines[0]
        );
        assert!(
            lines[1].contains("prefill_floor ~20ms"),
            "line2 should include prefill_floor below ridge: {}",
            lines[1]
        );
    }

    #[test]
    fn cache_use_fragment_formats_hit_rate_only() {
        assert_eq!(cache_use_fragment(None), "pfix_cache -");
        assert_eq!(cache_use_fragment(Some(0.0)), "pfix_cache 0%");
        assert_eq!(cache_use_fragment(Some(0.728)), "pfix_cache 72.8%");
    }

    fn baseline_efficiency(eff: f64) -> crate::engine::PhysicsBaseline {
        use crate::engine::baseline::{CeilingEstimate, PhysicsBaseline, WeightDtypeSource};
        PhysicsBaseline {
            decode: CeilingEstimate {
                lower: 100.0,
                expected: 100.0,
                upper: 100.0,
            },
            prefill: None,
            efficiency_pct: Some(eff),
            headroom_pct: None,
            weight_dtype_source: WeightDtypeSource::Fallback,
            weight_gb: 1.0,
            weight_bytes_per_param: 2,
            kv_bytes_per_element: 2,
            kv_cache_dtype_source: engine::KvCacheDtypeSource::Auto,
            kv_headroom_gb: None,
            tpot_floor_ms: 10.0,
            prefill_latency_floor_ms: None,
            ridge_batch_size: 1.0,
            config_relative_efficiency_pct: None,
            cost: None,
        }
    }

    fn baseline_with_cost(
        eff: f64,
        source: engine::CostSource,
        cpm: f64,
        tok_per_watt: Option<f64>,
        joules_per_token: Option<f64>,
    ) -> crate::engine::PhysicsBaseline {
        use crate::engine::baseline::{
            CeilingEstimate, CostEstimate, PhysicsBaseline, WeightDtypeSource,
        };
        PhysicsBaseline {
            decode: CeilingEstimate {
                lower: 100.0,
                expected: 100.0,
                upper: 100.0,
            },
            prefill: None,
            efficiency_pct: Some(eff),
            headroom_pct: None,
            weight_dtype_source: WeightDtypeSource::Fallback,
            weight_gb: 1.0,
            weight_bytes_per_param: 2,
            kv_bytes_per_element: 2,
            kv_cache_dtype_source: engine::KvCacheDtypeSource::Auto,
            kv_headroom_gb: None,
            tpot_floor_ms: 10.0,
            prefill_latency_floor_ms: None,
            ridge_batch_size: 1.0,
            config_relative_efficiency_pct: None,
            cost: Some(CostEstimate {
                tok_per_watt,
                joules_per_token,
                cost_per_million_tokens: Some(cpm),
                cost_source: source,
            }),
        }
    }

    #[test]
    fn gpu_gauges_line_includes_jtok_and_catalog_cost_est() {
        let g = GpuRawMetrics {
            power_watts: Some(421.0),
            ..Default::default()
        };
        let b = baseline_with_cost(
            31.7,
            engine::CostSource::Catalog,
            1.84,
            Some(14.2),
            Some(0.31),
        );
        let s = gpu_gauges_line(&g, Some(&b), Some(5978.2), false);
        assert!(s.contains("decode_eff ~31.7%"));
        assert!(s.contains("power 421W"));
        assert!(s.contains("0.31 J/tok"));
        assert!(!s.contains("tok/W"));
        assert!(s.contains("$1.84/1M tok (est)"));
    }

    #[test]
    fn gpu_gauges_line_verbose_shows_both_jtok_and_tok_per_watt() {
        let g = GpuRawMetrics {
            power_watts: Some(421.0),
            ..Default::default()
        };
        let b = baseline_with_cost(
            31.7,
            engine::CostSource::Catalog,
            1.84,
            Some(14.2),
            Some(0.31),
        );
        let s = gpu_gauges_line(&g, Some(&b), Some(5978.2), true);
        assert!(s.contains("0.31 J/tok"));
        assert!(s.contains("14.2 tok/W"));
    }

    #[test]
    fn gpu_gauges_line_user_cost_without_est_suffix() {
        let g = GpuRawMetrics {
            power_watts: Some(100.0),
            ..Default::default()
        };
        let b = baseline_with_cost(50.0, engine::CostSource::UserProvided, 2.50, None, None);
        let s = gpu_gauges_line(&g, Some(&b), None, false);
        assert!(s.contains("$2.50/1M tok"));
        assert!(!s.contains("(est)"));
        assert!(!s.contains("tok/W"));
    }

    #[test]
    fn quiet_efficiency_fallback_disclaimer_when_dtype_unknown() {
        let lines = quiet_efficiency_fallback_lines(Some(&baseline_efficiency(6.7)));
        assert_eq!(lines.len(), 3);
        assert!(lines[0].contains("Efficiency"));
        assert!(lines[0].contains("~6.7% of ceiling (est)"));
        assert!(lines[1].contains("Note: weight dtype not reported; ceiling assumes bf16."));
        assert!(lines[2].is_empty());
    }

    #[test]
    fn quiet_efficiency_fallback_absent_when_dtype_known() {
        use crate::engine::baseline::{CeilingEstimate, PhysicsBaseline, WeightDtypeSource};
        let b = PhysicsBaseline {
            decode: CeilingEstimate {
                lower: 100.0,
                expected: 100.0,
                upper: 100.0,
            },
            prefill: None,
            efficiency_pct: Some(50.0),
            headroom_pct: None,
            weight_dtype_source: WeightDtypeSource::EnvVar,
            weight_gb: 1.0,
            weight_bytes_per_param: 2,
            kv_bytes_per_element: 2,
            kv_cache_dtype_source: engine::KvCacheDtypeSource::Auto,
            kv_headroom_gb: None,
            tpot_floor_ms: 10.0,
            prefill_latency_floor_ms: None,
            ridge_batch_size: 1.0,
            config_relative_efficiency_pct: None,
            cost: None,
        };
        assert!(quiet_efficiency_fallback_lines(Some(&b)).is_empty());
    }

    #[test]
    fn gpu_gauges_line_formats_mem_gb() {
        let g = GpuRawMetrics {
            gpu_util_pct: Some(28.0),
            power_watts: Some(310.0),
            power_limit_watts: Some(400.0),
            vram_used_mb: Some(72 * 1024),
            vram_total_mb: Some(80 * 1024),
            ..Default::default()
        };
        let s = gpu_gauges_line(&g, Some(&baseline_efficiency(28.0)), None, false);
        assert!(s.contains("decode_eff ~28.0%"));
        assert!(s.contains("power 310W"));
        assert!(s.contains("vRAM 72/80GB"));
        let g_peak = GpuRawMetrics {
            gpu_util_pct: Some(28.0),
            power_watts: Some(310.0),
            vram_used_mb: Some(60 * 1024),
            vram_peak_mb: Some(78 * 1024),
            vram_total_mb: Some(80 * 1024),
            ..Default::default()
        };
        let s_peak = gpu_gauges_line(&g_peak, None, None, false);
        assert!(s_peak.contains("vRAM 60/80GB (peak 78GB)"));
        // 70/80 = 87.5% < 90% threshold - peak should be suppressed (noise, not signal)
        let g_peak_below_frac = GpuRawMetrics {
            gpu_util_pct: Some(28.0),
            power_watts: Some(310.0),
            vram_used_mb: Some(60 * 1024),
            vram_peak_mb: Some(70 * 1024),
            vram_total_mb: Some(80 * 1024),
            ..Default::default()
        };
        assert!(!gpu_gauges_line(&g_peak_below_frac, None, None, false).contains("peak 70GB"));
        let g_no_recovery = GpuRawMetrics {
            gpu_util_pct: Some(28.0),
            power_watts: Some(310.0),
            vram_used_mb: Some(78 * 1024),
            vram_peak_mb: Some(78 * 1024),
            vram_total_mb: Some(80 * 1024),
            ..Default::default()
        };
        assert!(!gpu_gauges_line(&g_no_recovery, None, None, false).contains("peak"));
    }

    #[test]
    fn gpu_gauges_line_efficiency_unknown_when_throughput_exceeds_ceiling_model() {
        let g = GpuRawMetrics::default();
        let mut b = baseline_efficiency(50.0);
        b.efficiency_pct = None;
        let s = gpu_gauges_line(&g, Some(&b), Some(200.0), false);
        assert!(s.contains("decode_eff ?"));
    }

    #[test]
    fn gpu_detail_line_shows_temp_peak_when_spike_hot_enough() {
        let g = GpuRawMetrics {
            mem_util_pct: Some(88.3),
            temperature_c: Some(72.0),
            temperature_peak_c: Some(86.0),
            sm_clock_mhz: Some(1980),
            power_limit_watts: Some(700.0),
            ..Default::default()
        };
        let line = gpu_detail_line(&g, true, false);
        assert!(line.contains("temp 72°C (peak 86°C)"));
    }

    #[test]
    fn gpu_detail_line_hides_temp_peak_below_threshold_or_not_above_current() {
        let below_80 = GpuRawMetrics {
            mem_util_pct: Some(50.0),
            temperature_c: Some(60.0),
            temperature_peak_c: Some(70.0),
            ..Default::default()
        };
        assert!(!gpu_detail_line(&below_80, true, false).contains("peak"));
        let not_recovered = GpuRawMetrics {
            mem_util_pct: Some(50.0),
            temperature_c: Some(86.0),
            temperature_peak_c: Some(86.0),
            ..Default::default()
        };
        assert!(!gpu_detail_line(&not_recovered, true, false).contains("peak"));
    }

    #[test]
    fn gpu_detail_line_multi_gpu_includes_vram_and_power() {
        let g = GpuRawMetrics {
            mem_util_pct: Some(13.0),
            power_watts: Some(171.0),
            vram_used_mb: Some(49 * 1024),
            vram_total_mb: Some(80 * 1024),
            ..Default::default()
        };
        let default = gpu_detail_line(&g, false, true);
        assert!(default.contains("vRAM 49/80GB"));
        assert!(default.contains("power 171W"));
        assert!(default.contains("mem_util 13%"));

        let verbose = gpu_detail_line(&g, true, true);
        assert!(verbose.contains("vRAM 49/80GB"));
        assert!(verbose.contains("power 171W"));
        assert!(verbose.contains("mem_util 13%"));
        assert!(verbose.contains("temp -"));
        assert!(verbose.contains("sm -"));
        assert!(verbose.contains("limit -"));
    }

    #[test]
    fn gpu_detail_line_single_gpu_omits_vram_and_power() {
        let g = GpuRawMetrics {
            mem_util_pct: Some(55.0),
            power_watts: Some(244.0),
            vram_used_mb: Some(40 * 1024),
            vram_total_mb: Some(80 * 1024),
            ..Default::default()
        };
        let default = gpu_detail_line(&g, false, false);
        assert!(!default.contains("vRAM"));
        assert!(!default.contains("power"));
        assert!(default.contains("mem_util 55%"));

        let verbose = gpu_detail_line(&g, true, false);
        assert!(!verbose.contains("vRAM"));
        assert!(!verbose.contains("power"));
        assert!(verbose.contains("mem_util 55%"));
    }

    #[test]
    fn vllm_requests_value_run_wait_max() {
        let v = VllmRawMetrics {
            num_requests_running: Some(2.0),
            num_requests_waiting: Some(1.0),
            max_num_seqs: Some(256),
            ..Default::default()
        };
        assert_eq!(
            vllm_requests_value(&v, Some(256)),
            "run 2 (0.8%) | wait 1 | max 256"
        );
    }

    #[test]
    fn vllm_requests_value_omits_pct_when_max_unknown() {
        let v = VllmRawMetrics {
            num_requests_running: Some(4.0),
            num_requests_waiting: Some(0.0),
            max_num_seqs: None,
            ..Default::default()
        };
        assert_eq!(vllm_requests_value(&v, None), "run 4 | wait 0 | max -");
        assert_eq!(
            vllm_requests_value(&v, Some(64)),
            "run 4 (6.2%) | wait 0 | max 64"
        );
    }

    #[test]
    fn vllm_throughput_value_tok_s_only() {
        let v = VllmRawMetrics {
            generation_tokens_per_sec: Some(59.0),
            prefix_cache_hit_rate: Some(0.5),
            ..Default::default()
        };
        assert_eq!(vllm_throughput_value(&v), "59 tok/s");
    }

    #[test]
    fn vllm_prompt_value_default_includes_kv_and_pfix_cache() {
        let v = VllmRawMetrics {
            prompt_tokens_mean: Some(18.0),
            kv_cache_usage_perc: Some(45.25),
            prefix_cache_hit_rate: Some(0.5),
            ..Default::default()
        };
        assert_eq!(
            vllm_prompt_value(&v, false, v.prefix_cache_hit_rate),
            "kv_cache 45.2% avg | pfix_cache 50.0%"
        );
    }

    #[test]
    fn vllm_prompt_value_verbose_includes_prompt_tok_kv_and_pfix_cache() {
        let v = VllmRawMetrics {
            prompt_tokens_mean: Some(18.0),
            kv_cache_usage_perc: Some(45.25),
            prefix_cache_hit_rate: Some(0.5),
            ..Default::default()
        };
        assert_eq!(
            vllm_prompt_value(&v, true, v.prefix_cache_hit_rate),
            "18 tok | kv_cache 45.2% avg | pfix_cache 50.0%"
        );
        let v_peak = VllmRawMetrics {
            prompt_tokens_mean: Some(18.0),
            kv_cache_usage_perc: Some(40.0),
            kv_cache_peak_perc: Some(92.0),
            prefix_cache_hit_rate: Some(0.5),
            ..Default::default()
        };
        assert_eq!(
            vllm_prompt_value(&v_peak, true, v_peak.prefix_cache_hit_rate),
            "18 tok | kv_cache 40.0% avg (92.0% peak) | pfix_cache 50.0%"
        );
        let peak_ceiling = VllmRawMetrics {
            kv_cache_usage_perc: Some(92.0),
            kv_cache_peak_perc: Some(100.0),
            ..Default::default()
        };
        assert_eq!(
            vllm_prompt_kv_fragment(&peak_ceiling),
            "kv_cache 92.0% avg (100.0% peak)"
        );
        let peak_no_spike = VllmRawMetrics {
            kv_cache_usage_perc: Some(67.0),
            kv_cache_peak_perc: Some(68.0),
            ..Default::default()
        };
        assert_eq!(
            vllm_prompt_kv_fragment(&peak_no_spike),
            "kv_cache 67.0% avg"
        );
        let peak_not_above_avg = VllmRawMetrics {
            prompt_tokens_mean: Some(18.0),
            kv_cache_usage_perc: Some(92.0),
            kv_cache_peak_perc: Some(92.0),
            ..Default::default()
        };
        assert_eq!(
            vllm_prompt_value(
                &peak_not_above_avg,
                true,
                peak_not_above_avg.prefix_cache_hit_rate
            ),
            "18 tok | kv_cache 92.0% avg | pfix_cache -"
        );
        let no_kv = VllmRawMetrics {
            prompt_tokens_mean: Some(512.0),
            ..Default::default()
        };
        assert_eq!(
            vllm_prompt_value(&no_kv, true, no_kv.prefix_cache_hit_rate),
            "512 tok | kv_cache - | pfix_cache -"
        );
    }

    #[test]
    fn vllm_latency_value_default_ttft_tpot_only() {
        let v = VllmRawMetrics {
            ttft_ms: Some(120.0),
            tpot_ms: Some(50.0),
            prefill_latency_ms: Some(200.0),
            queue_delay_ms: Some(10.0),
            ttft_p95_ms: Some(150.0),
            tpot_p95_ms: Some(60.0),
            ..Default::default()
        };
        assert_eq!(
            vllm_latency_value(&v, false),
            "ttft 120ms (p95 150ms) | tpot 50ms (p95 60ms)"
        );
        assert_eq!(
            vllm_latency_value(&v, true),
            "ttft 120ms (p95 150ms) | tpot 50ms (p95 60ms) | prefill 200ms | queue 10ms"
        );
    }

    #[test]
    fn vllm_latency_value_shows_p95_when_available() {
        let v = VllmRawMetrics {
            ttft_ms: Some(120.0),
            tpot_ms: Some(50.0),
            ttft_p95_ms: Some(892.0),
            tpot_p95_ms: Some(180.0),
            ..Default::default()
        };
        assert_eq!(
            vllm_latency_value(&v, false),
            "ttft 120ms (p95 892ms) | tpot 50ms (p95 180ms)"
        );
    }

    #[test]
    fn vllm_latency_value_shows_floor_when_p95_clamped() {
        let v = VllmRawMetrics {
            ttft_ms: Some(120.0),
            tpot_ms: Some(50.0),
            ttft_p95_ms: Some(40_000.0),
            ttft_p95_clamped: true,
            tpot_p95_ms: Some(40_000.0),
            tpot_p95_clamped: true,
            ..Default::default()
        };
        assert_eq!(
            vllm_latency_value(&v, false),
            "ttft 120ms (p95 >= 40.0s) | tpot 50ms (p95 >= 40.0s)"
        );
        let normal = VllmRawMetrics {
            ttft_ms: Some(120.0),
            tpot_ms: Some(50.0),
            ttft_p95_ms: Some(40_000.0),
            tpot_p95_ms: Some(180.0),
            ..Default::default()
        };
        assert_eq!(
            vllm_latency_value(&normal, false),
            "ttft 120ms (p95 40.0s) | tpot 50ms (p95 180ms)"
        );
        assert!(!vllm_latency_value(&normal, false).contains(">="));
    }

    #[test]
    fn vllm_label_row_aligns_labels_and_gap_before_metrics() {
        let line = vllm_label_row("REQUESTS", "run 2 (0.8%) | wait 1 | max 256");
        assert!(line.starts_with("REQUESTS"));
        assert!(line.contains(" run 2"));
        let t = vllm_label_row("THROUGHPUT", "59 tok/s");
        assert!(t.starts_with("THROUGHPUT"));
        assert!(t.contains(" 59 tok/s"));
        let c = vllm_label_row("CACHE", "kv_cache 67.0% avg | pfix_cache 72.8%");
        assert!(c.starts_with("CACHE"));
        assert!(c.contains("pfix_cache 72.8%"));
    }

    #[test]
    fn header_and_r3_body_use_same_aggregated_prefix_hit_rate() {
        let t = UNIX_EPOCH;
        let mk_snapshot = || RawSnapshot {
            gpu_observed_at: t,
            vllm_observed_at: t,
            timestamp: t,
            vllm: VllmRawMetrics {
                model_name: Some("meta-llama/Llama-3.1-8B-Instruct".to_string()),
                num_requests_running: Some(70.0),
                num_requests_waiting: Some(2.0),
                max_num_seqs: Some(256),
                prompt_tokens_mean: Some(64.0),
                request_success_per_sec: Some(20.0),
                prefix_cache_hit_rate: Some(0.276),
                generation_tokens_per_sec: Some(200.0),
                window_duration_secs: Some(2.0),
                ..Default::default()
            },
            gpus: vec![GpuRawMetrics {
                gpu_name: Some("NVIDIA H100 80GB HBM3".to_string()),
                gpu_util_pct: Some(70.0),
                ..Default::default()
            }],
        };
        let w1 = RuntimeWindow::from_snapshot(mk_snapshot());
        let w2 = RuntimeWindow::from_snapshot(mk_snapshot());
        let w3 = RuntimeWindow::from_snapshot(mk_snapshot());
        let snapshot = w3.snapshot.clone();
        let static_ctx = StaticContext::from_snapshot(&snapshot, VllmConfig::default());
        let result = DiagnoseResult {
            snapshot,
            windows: vec![w1, w2, w3],
            static_ctx,
            duration: Duration::from_secs(6),
            started_at: UNIX_EPOCH,
            any_evaluable: true,
            all_idle: false,
            metrics_input: "http://127.0.0.1:8000/metrics".to_string(),
        };
        let text = diagnose_lines_for(&result, false).join("\n");
        assert!(text.contains("pfix_cache 27.6%"));
        assert!(text.contains("Prefix hit rate 27.6%"));
    }

    #[test]
    fn diagnose_lines_when_no_evaluable_skip_metric_table() {
        let idle_snap = RawSnapshot {
            gpu_observed_at: UNIX_EPOCH,
            vllm_observed_at: UNIX_EPOCH,
            timestamp: UNIX_EPOCH,
            vllm: VllmRawMetrics {
                num_requests_running: Some(0.0),
                generation_tokens_per_sec: Some(0.0),
                model_name: Some("test-model".into()),
                ..Default::default()
            },
            gpus: vec![GpuRawMetrics {
                gpu_name: Some("Test GPU".into()),
                ..Default::default()
            }],
        };
        let result = DiagnoseResult {
            snapshot: idle_snap.clone(),
            windows: vec![RuntimeWindow::from_snapshot(idle_snap)],
            static_ctx: StaticContext::default(),
            duration: Duration::from_secs(2),
            started_at: UNIX_EPOCH,
            any_evaluable: false,
            all_idle: false,
            metrics_input: "http://127.0.0.1:8000/metrics".into(),
        };
        let lines = diagnose_lines_for(&result, false);
        let text = lines.join("\n");
        let target_idx = lines
            .iter()
            .position(|l| l.contains("Target:"))
            .expect("Target row");
        assert!(target_idx > 0 && lines[target_idx - 1].is_empty());
        assert!(text.contains("Target:") && text.contains("127.0.0.1"));
        assert!(text.contains("[!] Telemetry Failure"));
        assert!(text.contains("curl -s http://127.0.0.1:8000/metrics"));
        assert!(!text.contains("Server is idle"));
        assert!(!text.contains("benchmark_serving.py"));
        assert!(!text.contains("GPU =>"));
    }

    #[test]
    fn diagnose_lines_idle_server_shows_load_hint() {
        let idle_snap = RawSnapshot {
            gpu_observed_at: UNIX_EPOCH,
            vllm_observed_at: UNIX_EPOCH,
            timestamp: UNIX_EPOCH,
            vllm: VllmRawMetrics {
                num_requests_running: Some(0.0),
                generation_tokens_per_sec: Some(0.0),
                window_duration_secs: Some(2.0),
                model_name: Some("test-model".into()),
                ..Default::default()
            },
            gpus: vec![GpuRawMetrics {
                gpu_name: Some("Test GPU".into()),
                ..Default::default()
            }],
        };
        let result = DiagnoseResult {
            snapshot: idle_snap.clone(),
            windows: vec![RuntimeWindow::from_snapshot(idle_snap)],
            static_ctx: StaticContext::default(),
            duration: Duration::from_secs(30),
            started_at: UNIX_EPOCH,
            any_evaluable: true,
            all_idle: true,
            metrics_input: "http://127.0.0.1:8000/metrics".into(),
        };
        let lines = diagnose_lines_for(&result, false);
        let text = lines.join("\n");
        let target_idx = lines
            .iter()
            .position(|l| l.contains("Target:"))
            .expect("Target row");
        assert!(target_idx > 0 && lines[target_idx - 1].is_empty());
        assert!(text.contains("Server is idle"));
        assert!(text.contains("benchmark_serving.py"));
        assert!(!text.contains("No issues detected"));
    }

    #[test]
    fn advisory_index_only_identity() {
        let mut lines = Vec::new();
        let snap = RawSnapshot {
            gpus: vec![GpuRawMetrics {
                gpu_index: Some(0),
                ..Default::default()
            }],
            ..Default::default()
        };
        push_gpu_advisories(&mut lines, &snap);
        assert_eq!(lines.len(), 1);
        assert!(lines[0].contains("device index only"));
    }

    #[test]
    fn multi_gpu_diagnose_shows_cluster_header_and_per_gpu_rows() {
        let t = UNIX_EPOCH;
        let snapshot = RawSnapshot {
            gpu_observed_at: t,
            vllm_observed_at: t,
            timestamp: t,
            vllm: VllmRawMetrics {
                model_name: Some("test-model".into()),
                num_requests_running: Some(8.0),
                generation_tokens_per_sec: Some(200.0),
                window_duration_secs: Some(2.0),
                ..Default::default()
            },
            gpus: vec![
                GpuRawMetrics {
                    gpu_index: Some(0),
                    gpu_name: Some("NVIDIA H100 80GB HBM3".into()),
                    mem_util_pct: Some(40.0),
                    power_watts: Some(300.0),
                    vram_used_mb: Some(40 * 1024),
                    vram_total_mb: Some(80 * 1024),
                    ..Default::default()
                },
                GpuRawMetrics {
                    gpu_index: Some(1),
                    gpu_name: Some("NVIDIA H100 80GB HBM3".into()),
                    mem_util_pct: Some(42.0),
                    power_watts: Some(310.0),
                    vram_used_mb: Some(41 * 1024),
                    vram_total_mb: Some(80 * 1024),
                    ..Default::default()
                },
            ],
        };
        let static_ctx = StaticContext::from_snapshot(&snapshot, VllmConfig::default());
        let result = DiagnoseResult {
            snapshot: snapshot.clone(),
            windows: vec![RuntimeWindow::from_snapshot(snapshot)],
            static_ctx,
            duration: Duration::from_secs(2),
            started_at: UNIX_EPOCH,
            any_evaluable: true,
            all_idle: false,
            metrics_input: "http://127.0.0.1:8000/metrics".into(),
        };
        let text = diagnose_lines_for(&result, false).join("\n");
        assert!(text.contains(" x2"));
        assert!(text.contains("GPU [0]"));
        assert!(text.contains("GPU [1]"));
        assert!(text.contains("power 610W"));
    }

    fn report_with_n_eval_and_rec(n_eval: usize) -> engine::Report {
        engine::Report {
            baseline: None,
            recommendations: vec![engine::Recommendation {
                rule_name: "oom_risk",
                layer: 1,
                impact: 5,
                confidence: 0.9,
                display_lines: vec!["[!] OOM".to_string()],
            }],
            suppressed_rules: Vec::new(),
            kv_max_seqs: None,
            prescribed_kv_capacity: None,
            catalog_state_mismatch: None,
            memory_budget_self_grade: None,
            n_eval,
            skipped_broken: 0,
            skipped_idle: 0,
            energy_skew_skipped: 0,
            gauge_missing: Default::default(),
            limiter_evidence: None,
        }
    }

    #[test]
    fn journey_line_none_when_sparse_even_with_recommendations() {
        assert_eq!(
            journey_line(&report_with_n_eval_and_rec(2)),
            None,
            "sparse n_eval must suppress apply-fix footer"
        );
    }

    #[test]
    fn journey_line_some_when_sustained_with_recommendations() {
        assert_eq!(
            journey_line(&report_with_n_eval_and_rec(3)),
            Some("▶  Apply the fix above. Profile re-measures after your change.")
        );
    }
}
