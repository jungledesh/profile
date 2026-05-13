use std::time::Duration;
use std::time::SystemTime;

use chrono::{DateTime, Utc};

use crate::collectors::{GpuRawMetrics, VllmConfig, VllmRawMetrics};
use crate::context::AnalysisInput;
use crate::engine;
use crate::profiler::DiagnoseResult;

const VLLM_LABEL_W: usize = 10;
const VLLM_LABEL_METRICS_GAP: &str = " ";

/// KV cache peak parenthetical only if spike reached at least this (rule r2 neighborhood).
const KV_CACHE_PEAK_SHOW_THRESHOLD_PCT: f64 = 85.0;
/// Peak VRAM / total must reach this fraction to show a spike parenthetical.
const VRAM_PEAK_SHOW_THRESHOLD_FRAC: f64 = 0.90;
/// Global GPU temp parenthetical until per-arch throttle thresholds exist (Hopper ~83°C).
const GPU_TEMP_PEAK_SHOW_THRESHOLD_C: f64 = 80.0;

#[inline]
fn show_kv_cache_peak_parenthetical(last_pct: f64, peak_pct: f64) -> bool {
    peak_pct > last_pct && peak_pct >= KV_CACHE_PEAK_SHOW_THRESHOLD_PCT
}

#[inline]
fn show_vram_peak_parenthetical(used_mb: u64, peak_mb: u64, total_mb: u64) -> bool {
    peak_mb > used_mb && (peak_mb as f64 / total_mb as f64) >= VRAM_PEAK_SHOW_THRESHOLD_FRAC
}

#[inline]
fn show_gpu_temp_peak_parenthetical(current_c: f64, peak_c: f64) -> bool {
    peak_c > current_c && peak_c >= GPU_TEMP_PEAK_SHOW_THRESHOLD_C
}

pub fn print_diagnose_table(result: &DiagnoseResult, verbose_rules: bool) {
    let lines = build_diagnose_lines(result, verbose_rules);
    print_boxed(&lines);
}

fn build_diagnose_lines(result: &DiagnoseResult, verbose_rules: bool) -> Vec<String> {
    let snapshot = &result.snapshot;
    let v = &snapshot.vllm;
    let g = &snapshot.gpu;
    let duration = result.duration;
    let started_at = result.started_at;

    let model = v.model_name.as_deref().unwrap_or("(unknown model)");
    let gpu_label = g.gpu_name.as_deref().unwrap_or("(no GPU)");
    let ts = format_profile_timestamp(started_at);
    let mut lines = vec![profile_header_line(
        env!("CARGO_PKG_VERSION"),
        model,
        gpu_label,
        &ts,
        duration,
    )];

    // Build AnalysisInput from the aggregate snapshot for baseline + single-window rule evaluation.
    let aggregate_win = crate::context::RuntimeWindow::from_snapshot(result.snapshot.clone());
    let summary_input = AnalysisInput::new(&result.static_ctx, &aggregate_win);
    let report = engine::build_report(summary_input);
    if verbose_rules {
        lines.push(String::new());
        lines.extend(baseline_lines(
            report.baseline,
            aggregate_win.snapshot.vllm.prefix_cache_hit_rate,
            aggregate_win.snapshot.vllm.num_requests_running,
        ));
        lines.push(String::new());
    }

    if !result.any_evaluable {
        lines.push(vllm_label_row("Target:", &result.metrics_input));
        lines.push(String::new());
        lines.extend(engine::no_evaluable_diagnose_lines(
            verbose_rules,
            &result.windows,
        ));
        return lines;
    }

    lines.push(format!(
        "{:<width$}{}{}",
        "GPU =>",
        VLLM_LABEL_METRICS_GAP,
        gpu_gauges_line(g),
        width = VLLM_LABEL_W
    ));
    if verbose_rules {
        lines.push(format!(
            "{:<width$}{}{}",
            "",
            VLLM_LABEL_METRICS_GAP,
            gpu_detail_line(g),
            width = VLLM_LABEL_W
        ));
    }
    lines.push(String::new());
    lines.push(vllm_label_row("vLLM:", ""));
    lines.push(vllm_label_row("REQUESTS", &vllm_requests_value(v)));
    lines.push(vllm_label_row(
        "LATENCY",
        &vllm_latency_value(v, verbose_rules),
    ));
    lines.push(vllm_label_row(
        "PROMPT",
        &vllm_prompt_value(v, verbose_rules),
    ));
    lines.push(vllm_label_row("THROUGHPUT", &vllm_throughput_value(v)));
    if verbose_rules {
        lines.push(vllm_label_row("MEMORY", &vllm_memory_value(v)));
        lines.push(vllm_label_row("TRAFFIC", &vllm_traffic_value(v)));
        lines.push(vllm_label_row("CACHE CFG", &vllm_cache_cfg_value(v)));
        lines.push(String::new());
        lines.push(vllm_label_row("Config:", ""));
        let cfg = &result.static_ctx.config;
        lines.push(vllm_label_row("PARALLEL", &config_parallel_value(cfg)));
        lines.push(vllm_label_row("MODEL", &config_model_value(cfg)));
        lines.push(vllm_label_row("KV", &config_kv_value(cfg)));
    }

    let rule_lines = if result.windows.len() <= 1 {
        engine::format_diagnose_rules(summary_input, verbose_rules)
    } else {
        engine::format_diagnose_rules_for_windows(&result.windows, summary_input, verbose_rules)
    };
    if !rule_lines.is_empty() {
        lines.push(String::new());
        lines.push("ISSUES:".to_string());
        lines.push(String::new());
        lines.extend(rule_lines);
    }

    lines
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

fn over_ceiling_hint(
    raw_eff: f64,
    decode_expected: f64,
    ridge_batch_size: f64,
    num_requests_running: Option<f64>,
    cache_hit_rate: Option<f64>,
) -> &'static str {
    let actual_tps = (raw_eff / 100.0) * decode_expected;
    if let Some(r) = cache_hit_rate.filter(|x| x.is_finite() && *x >= 0.0 && *x < 1.0) {
        let effective_ceiling = decode_expected / (1.0 - r);
        if actual_tps <= effective_ceiling {
            return "(prefix cache inflating throughput)";
        }
    }
    if let Some(n) = num_requests_running.filter(|x| x.is_finite()) {
        if n >= ridge_batch_size {
            return "(large batch — compute-bound)";
        }
    }
    "(verify weight dtype)"
}

fn baseline_lines(
    baseline: Option<engine::PhysicsBaseline>,
    prefix_cache_hit_rate: Option<f64>,
    num_requests_running: Option<f64>,
) -> Vec<String> {
    let Some(b) = baseline else {
        return vec!["HW LIMITS  unavailable — model not recognized".to_string()];
    };

    // Line 1: efficiency + throughput ceilings
    let mut seg1 = Vec::new();
    if let Some(raw_eff) = b.efficiency_pct {
        if raw_eff > 100.0 {
            let hint = over_ceiling_hint(
                raw_eff,
                b.decode.expected,
                b.ridge_batch_size,
                num_requests_running,
                prefix_cache_hit_rate,
            );
            seg1.push(format!(">100% of decode ceiling {hint}"));
        } else {
            seg1.push(format!("{raw_eff:.1}% of decode ceiling"));
        }
    }
    if b.decode.expected >= 0.5 {
        seg1.push(format!("decode ~{:.0} tok/s (est)", b.decode.expected));
    }
    if let Some(prefill) = b.prefill {
        if prefill.expected >= 10.0 {
            seg1.push(format!("prefill ~{:.0} tok/s (est)", prefill.expected));
        }
    }

    // Line 2: memory budget + latency floors
    let mut seg2 = Vec::new();
    seg2.push(format!("weight {:.0}GB", b.weight_gb));
    if let Some(headroom) = b.kv_headroom_gb {
        if headroom < 0.0 {
            seg2.push(format!("kv_headroom {:.0}GB (needs TP)", headroom));
        } else {
            seg2.push(format!("kv_headroom {:.0}GB", headroom));
        }
    }
    seg2.push(format!("tpot_floor ~{:.0}ms", b.tpot_floor_ms));
    if let Some(pf) = b.prefill_latency_floor_ms {
        let compute_bound = num_requests_running
            .filter(|x| x.is_finite())
            .is_some_and(|n| n >= b.ridge_batch_size);
        let prefill_ceiling_meaningful = b.prefill.is_some_and(|p| p.expected >= 10.0);
        let over_ceiling = b.efficiency_pct.is_some_and(|e| e > 100.0);
        if pf >= 0.5 && !compute_bound && prefill_ceiling_meaningful && !over_ceiling {
            seg2.push(format!("prefill_floor ~{:.0}ms", pf));
        }
    }

    let mut out = vec![
        format!("HW LIMITS  {}", seg1.join(" | ")),
        format!("           {}", seg2.join(" | ")),
    ];
    if b.weight_dtype_source == engine::WeightDtypeSource::Fallback {
        out.push("           weight dtype assumed bf16 — set DTYPE env var to confirm".to_string());
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

fn gpu_gauges_line(g: &GpuRawMetrics) -> String {
    let util = g
        .gpu_util_pct
        .map(|u| format!("UTIL {:.1}%", u))
        .unwrap_or_else(|| "UTIL —".to_string());

    let power = g
        .power_watts
        .map(|draw| format!("POWER {:.0}W", draw))
        .unwrap_or_else(|| "POWER —".to_string());

    let mem = match (g.vram_used_mb, g.vram_total_mb) {
        (Some(used), Some(total)) if total > 0 => {
            let u_gb = used as f64 / 1024.0;
            let t_gb = total as f64 / 1024.0;
            let mut s = format!("vRAM {:.0}/{:.0}GB", u_gb, t_gb);
            if let Some(pk) = g.vram_peak_mb {
                if show_vram_peak_parenthetical(used, pk, total) {
                    let pk_gb = pk as f64 / 1024.0;
                    s.push_str(&format!(" (peak {:.0}GB)", pk_gb));
                }
            }
            s
        }
        _ => "vRAM —".to_string(),
    };

    format!("{util} | {power} | {mem}")
}

fn vllm_requests_value(v: &VllmRawMetrics) -> String {
    let run = match v.num_requests_running.filter(|x| x.is_finite()) {
        Some(avg) => {
            let rounded = avg.round();
            if let Some(max_n) = v.max_num_seqs.filter(|&m| m > 0) {
                let pct = (avg / f64::from(max_n)) * 100.0;
                format!("run {:.0} ({:.1}%)", rounded, pct)
            } else {
                format!("run {:.0}", rounded)
            }
        }
        None => "run —".to_string(),
    };
    let wait = match v.num_requests_waiting.filter(|x| x.is_finite()) {
        Some(w) => format!("wait {:.0}", w.round()),
        None => "wait —".to_string(),
    };
    let max_seq = v
        .max_num_seqs
        .map(|n| format!("max {n}"))
        .unwrap_or_else(|| "max —".to_string());

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
        .unwrap_or_else(|| "—".to_string());
    let tpot = v
        .tpot_ms
        .map(fmt_seconds_from_ms)
        .unwrap_or_else(|| "—".to_string());
    if !verbose {
        return format!("ttft {ttft} | tpot {tpot}");
    }
    let prefill = v
        .prefill_latency_ms
        .map(fmt_seconds_from_ms)
        .unwrap_or_else(|| "—".to_string());
    let queue = v
        .queue_delay_ms
        .map(fmt_seconds_from_ms)
        .unwrap_or_else(|| "—".to_string());
    format!("ttft {ttft} | tpot {tpot} | prefill {prefill} | queue {queue}")
}

fn vllm_prompt_kv_fragment(v: &VllmRawMetrics) -> String {
    match v.kv_cache_usage_perc.filter(|x| x.is_finite()) {
        Some(p) => {
            let mut s = format!("kv_cache {:.1}%", p);
            if let Some(pk) = v.kv_cache_peak_perc.filter(|x| x.is_finite()) {
                if show_kv_cache_peak_parenthetical(p, pk) {
                    s.push_str(&format!(" (peak {:.1}%)", pk));
                }
            }
            s
        }
        None => "kv_cache —".to_string(),
    }
}

fn vllm_prompt_value(v: &VllmRawMetrics, verbose: bool) -> String {
    let kv = vllm_prompt_kv_fragment(v);
    if !verbose {
        return kv;
    }
    let n = v
        .prompt_tokens_mean
        .map(fmt_tok)
        .unwrap_or_else(|| "—".to_string());
    format!("{n} tok | {kv}")
}

fn fmt_tok(t: f64) -> String {
    if (t - t.round()).abs() < 1e-6 {
        format!("{:.0}", t)
    } else {
        format!("{:.1}", t)
    }
}

fn vllm_throughput_value(v: &VllmRawMetrics) -> String {
    let tps = v
        .generation_tokens_per_sec
        .map(|t| format!("{:.0} tok/s", t))
        .unwrap_or_else(|| "— tok/s".to_string());
    let cache = cache_use_fragment(v);
    format!("{tps} | {cache}")
}

fn cache_use_fragment(v: &VllmRawMetrics) -> String {
    match v.prefix_cache_hit_rate {
        Some(0.0) => "pfix_cache 0%".to_string(),
        Some(r) => format!("pfix_cache {:.1}%", r * 100.0),
        None => "pfix_cache —".to_string(),
    }
}

fn gpu_detail_line(g: &GpuRawMetrics) -> String {
    let mem_util = g
        .mem_util_pct
        .map(|u| format!("mem_util {:.1}%", u))
        .unwrap_or_else(|| "mem_util —".to_string());
    let temp = match g.temperature_c.filter(|t| t.is_finite()) {
        Some(cur) => {
            let mut s = format!("temp {:.0}°C", cur);
            if let Some(pk) = g.temperature_peak_c.filter(|t| t.is_finite()) {
                if show_gpu_temp_peak_parenthetical(cur, pk) {
                    s.push_str(&format!(" (peak {:.0}°C)", pk));
                }
            }
            s
        }
        None => "temp —".to_string(),
    };
    let sm = g
        .sm_clock_mhz
        .map(|c| format!("sm {}MHz", c))
        .unwrap_or_else(|| "sm —".to_string());
    let limit = g
        .power_limit_watts
        .map(|l| format!("limit {:.0}W", l))
        .unwrap_or_else(|| "limit —".to_string());
    format!("{mem_util} | {temp} | {sm} | {limit}")
}

fn vllm_memory_value(v: &VllmRawMetrics) -> String {
    let swapped = v
        .num_requests_swapped
        .map(fmt_gauge)
        .map(|s| format!("swapped {s}"))
        .unwrap_or_else(|| "swapped —".to_string());
    let cpu_cache = match v.cpu_cache_usage_perc.filter(|x| x.is_finite()) {
        Some(p) => format!("cpu_cache {:.1}%", p),
        None => "cpu_cache —".to_string(),
    };
    format!("{swapped} | {cpu_cache}")
}

fn vllm_traffic_value(v: &VllmRawMetrics) -> String {
    let qps = v
        .request_success_per_sec
        .map(|q| format!("qps {:.1}", q))
        .unwrap_or_else(|| "qps —".to_string());
    let req_total = v
        .request_success_total
        .map(|t| format!("req_total {:.0}", t))
        .unwrap_or_else(|| "req_total —".to_string());
    let gen_total = v
        .generation_tokens_total
        .map(|t| format!("gen_total {:.0}", t))
        .unwrap_or_else(|| "gen_total —".to_string());
    let preempt_rate = v
        .num_preemptions_per_sec
        .map(|p| format!("preempt/s {:.2}", p))
        .unwrap_or_else(|| "preempt/s —".to_string());
    let preempt_total = v
        .num_preemptions_total
        .map(|t| format!("preempt_total {:.0}", t))
        .unwrap_or_else(|| "preempt_total —".to_string());
    format!("{qps} | {req_total} | {gen_total} | {preempt_rate} | {preempt_total}")
}

fn vllm_cache_cfg_value(v: &VllmRawMetrics) -> String {
    let block = v
        .cache_config
        .block_size
        .map(|b| format!("block {b}"))
        .unwrap_or_else(|| "block —".to_string());
    let dtype = v.cache_config.cache_dtype.as_deref().unwrap_or("—");
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
        .unwrap_or("prefix_cache —");
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
        .unwrap_or("chunked_prefill —");
    format!("{block} | dtype {dtype} | {prefix} | {chunked}")
}

fn config_parallel_value(cfg: &VllmConfig) -> String {
    let tp = cfg
        .tensor_parallel_size
        .map(|v| format!("tp {v}"))
        .unwrap_or_else(|| "tp —".to_string());
    let pp = cfg
        .pipeline_parallel_size
        .map(|v| format!("pp {v}"))
        .unwrap_or_else(|| "pp —".to_string());
    format!("{tp} | {pp}")
}

fn config_model_value(cfg: &VllmConfig) -> String {
    let max_len = cfg
        .max_model_len
        .map(|v| format!("max_len {v}"))
        .unwrap_or_else(|| "max_len —".to_string());
    let dtype = cfg.dtype.as_deref().unwrap_or("—");
    let quant = cfg.quantization.as_deref().unwrap_or("—");
    let gpu_mem = cfg
        .gpu_memory_utilization
        .map(|v| format!("gpu_mem_util {:.2}", v))
        .unwrap_or_else(|| "gpu_mem_util —".to_string());
    format!("{max_len} | dtype {dtype} | quant {quant} | {gpu_mem}")
}

fn config_kv_value(cfg: &VllmConfig) -> String {
    let kv_dtype = cfg.kv_cache_dtype.as_deref().unwrap_or("—");
    let block = cfg
        .block_size
        .map(|b| format!("block {b}"))
        .unwrap_or_else(|| "block —".to_string());
    let prefix = cfg
        .enable_prefix_caching
        .map(|b| {
            if b {
                "prefix_cache on"
            } else {
                "prefix_cache off"
            }
        })
        .unwrap_or("prefix_cache —");
    let chunked = cfg
        .enable_chunked_prefill
        .map(|b| {
            if b {
                "chunked_prefill on"
            } else {
                "chunked_prefill off"
            }
        })
        .unwrap_or("chunked_prefill —");
    format!("dtype {kv_dtype} | {block} | {prefix} | {chunked}")
}

fn fmt_seconds_from_ms(ms: f64) -> String {
    if ms >= 1000.0 {
        format!("{:.1}s", ms / 1000.0)
    } else {
        format!("{:.0}ms", ms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collectors::{GpuRawMetrics, RawSnapshot, VllmRawMetrics};
    use crate::context::{RuntimeWindow, StaticContext};
    use crate::profiler::DiagnoseResult;
    use std::time::{Duration, UNIX_EPOCH};

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
    fn fmt_seconds_from_ms_prefers_seconds_when_large() {
        assert_eq!(fmt_seconds_from_ms(1200.0), "1.2s");
        assert_eq!(fmt_seconds_from_ms(50.0), "50ms");
    }

    #[test]
    fn over_ceiling_hint_prefix_cache_when_under_effective_ceiling() {
        // decode 100, raw 150% → actual 150 tok/s; hit 0.4 → effective 166.67
        assert_eq!(
            over_ceiling_hint(150.0, 100.0, 40.0, None, Some(0.4)),
            "(prefix cache inflating throughput)"
        );
    }

    #[test]
    fn over_ceiling_hint_dtype_when_above_effective_ceiling() {
        // decode 100, raw 200% → actual 200 tok/s; hit 0.4 → effective 166.67; running below ridge → dtype
        assert_eq!(
            over_ceiling_hint(200.0, 100.0, 40.0, Some(10.0), Some(0.4)),
            "(verify weight dtype)"
        );
    }

    #[test]
    fn over_ceiling_hint_dtype_when_hit_rate_missing() {
        assert_eq!(
            over_ceiling_hint(150.0, 100.0, 40.0, Some(10.0), None),
            "(verify weight dtype)"
        );
    }

    #[test]
    fn over_ceiling_hint_dtype_when_hit_rate_out_of_band() {
        assert_eq!(
            over_ceiling_hint(150.0, 100.0, 40.0, Some(10.0), Some(f64::NAN)),
            "(verify weight dtype)"
        );
        assert_eq!(
            over_ceiling_hint(150.0, 100.0, 40.0, Some(10.0), Some(-0.1)),
            "(verify weight dtype)"
        );
        assert_eq!(
            over_ceiling_hint(150.0, 100.0, 40.0, Some(10.0), Some(1.0)),
            "(verify weight dtype)"
        );
    }

    #[test]
    fn over_ceiling_hint_large_batch_compute_bound_when_no_cache_explanation() {
        assert_eq!(
            over_ceiling_hint(150.0, 100.0, 40.0, Some(50.0), None),
            "(large batch — compute-bound)"
        );
    }

    #[test]
    fn over_ceiling_hint_cache_wins_before_large_batch() {
        // actual 150 ≤ effective 166.67 — prefix cache; running is high but irrelevant
        assert_eq!(
            over_ceiling_hint(150.0, 100.0, 40.0, Some(100.0), Some(0.4)),
            "(prefix cache inflating throughput)"
        );
    }

    #[test]
    fn over_ceiling_hint_dtype_when_running_none() {
        assert_eq!(
            over_ceiling_hint(150.0, 100.0, 40.0, None, None),
            "(verify weight dtype)"
        );
    }

    #[test]
    fn baseline_lines_over_ceiling_wires_hint_into_first_line() {
        // 150% efficiency at decode 100 tok/s → actual 150; hit 0.4 → effective 166.67 → prefix-cache hint.
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
            efficiency_pct: Some(150.0),
            headroom_pct: Some(0.0),
            weight_dtype_source: engine::WeightDtypeSource::EnvVar,
            weight_gb: 16.0,
            kv_headroom_gb: Some(8.0),
            tpot_floor_ms: 10.0,
            prefill_latency_floor_ms: Some(20.0),
            ridge_batch_size: 40.0,
        };
        let lines = baseline_lines(Some(b), Some(0.4), None);
        assert_eq!(
            lines[0],
            "HW LIMITS  >100% of decode ceiling (prefix cache inflating throughput) | decode ~100 tok/s (est) | prefill ~50 tok/s (est)"
        );
        assert_eq!(
            lines[1],
            "           weight 16GB | kv_headroom 8GB | tpot_floor ~10ms"
        );
        assert_eq!(lines.len(), 2);
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
            kv_headroom_gb: Some(8.0),
            tpot_floor_ms: 10.0,
            prefill_latency_floor_ms: Some(42.0),
            ridge_batch_size: 40.0,
        };
        let above = baseline_lines(Some(base()), None, Some(40.0));
        assert!(
            !above[1].contains("prefill_floor"),
            "expected no prefill_floor at ridge: {}",
            above[1]
        );
        let below = baseline_lines(Some(base()), None, Some(39.0));
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
            kv_headroom_gb: Some(8.0),
            tpot_floor_ms: 10.0,
            prefill_latency_floor_ms: Some(200.0),
            ridge_batch_size: 40.0,
        };
        let lines = baseline_lines(Some(b), None, Some(5.0));
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
            kv_headroom_gb: Some(8.0),
            tpot_floor_ms: 10.0,
            prefill_latency_floor_ms: Some(20.0),
            ridge_batch_size: 40.0,
        };
        let lines = baseline_lines(Some(b), None, Some(10.0));
        assert!(
            lines[0].contains("prefill ~50 tok/s (est)"),
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
        assert_eq!(
            cache_use_fragment(&VllmRawMetrics::default()),
            "pfix_cache —"
        );
        assert_eq!(
            cache_use_fragment(&VllmRawMetrics {
                prefix_cache_hit_rate: Some(0.0),
                ..Default::default()
            }),
            "pfix_cache 0%"
        );
        assert_eq!(
            cache_use_fragment(&VllmRawMetrics {
                prefix_cache_hit_rate: Some(0.728),
                ..Default::default()
            }),
            "pfix_cache 72.8%"
        );
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
        let s = gpu_gauges_line(&g);
        assert!(s.contains("UTIL 28.0%"));
        assert!(s.contains("POWER 310W"));
        assert!(s.contains("vRAM 72/80GB"));
        let g_peak = GpuRawMetrics {
            gpu_util_pct: Some(28.0),
            power_watts: Some(310.0),
            vram_used_mb: Some(60 * 1024),
            vram_peak_mb: Some(78 * 1024),
            vram_total_mb: Some(80 * 1024),
            ..Default::default()
        };
        let s_peak = gpu_gauges_line(&g_peak);
        assert!(s_peak.contains("vRAM 60/80GB (peak 78GB)"));
        let g_peak_below_frac = GpuRawMetrics {
            gpu_util_pct: Some(28.0),
            power_watts: Some(310.0),
            vram_used_mb: Some(60 * 1024),
            vram_peak_mb: Some(70 * 1024),
            vram_total_mb: Some(80 * 1024),
            ..Default::default()
        };
        assert!(!gpu_gauges_line(&g_peak_below_frac).contains("peak"));
        let g_no_recovery = GpuRawMetrics {
            gpu_util_pct: Some(28.0),
            power_watts: Some(310.0),
            vram_used_mb: Some(78 * 1024),
            vram_peak_mb: Some(78 * 1024),
            vram_total_mb: Some(80 * 1024),
            ..Default::default()
        };
        assert!(!gpu_gauges_line(&g_no_recovery).contains("peak"));
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
        let line = gpu_detail_line(&g);
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
        assert!(!gpu_detail_line(&below_80).contains("peak"));
        let not_recovered = GpuRawMetrics {
            mem_util_pct: Some(50.0),
            temperature_c: Some(86.0),
            temperature_peak_c: Some(86.0),
            ..Default::default()
        };
        assert!(!gpu_detail_line(&not_recovered).contains("peak"));
    }

    #[test]
    fn vllm_requests_value_run_wait_max() {
        let v = VllmRawMetrics {
            num_requests_running: Some(2.0),
            num_requests_waiting: Some(1.0),
            max_num_seqs: Some(256),
            ..Default::default()
        };
        assert_eq!(vllm_requests_value(&v), "run 2 (0.8%) | wait 1 | max 256");
    }

    #[test]
    fn vllm_requests_value_omits_pct_when_max_unknown() {
        let v = VllmRawMetrics {
            num_requests_running: Some(4.0),
            num_requests_waiting: Some(0.0),
            max_num_seqs: None,
            ..Default::default()
        };
        assert_eq!(vllm_requests_value(&v), "run 4 | wait 0 | max —");
    }

    #[test]
    fn vllm_throughput_value_tok_s_and_cache() {
        let v = VllmRawMetrics {
            generation_tokens_per_sec: Some(59.0),
            prefix_cache_hit_rate: Some(0.5),
            ..Default::default()
        };
        assert_eq!(vllm_throughput_value(&v), "59 tok/s | pfix_cache 50.0%");
    }

    #[test]
    fn vllm_prompt_value_default_is_kv_only() {
        let v = VllmRawMetrics {
            prompt_tokens_mean: Some(18.0),
            kv_cache_usage_perc: Some(45.25),
            ..Default::default()
        };
        assert_eq!(vllm_prompt_value(&v, false), "kv_cache 45.2%");
    }

    #[test]
    fn vllm_prompt_value_verbose_includes_prompt_tok_and_kv() {
        let v = VllmRawMetrics {
            prompt_tokens_mean: Some(18.0),
            kv_cache_usage_perc: Some(45.25),
            ..Default::default()
        };
        assert_eq!(vllm_prompt_value(&v, true), "18 tok | kv_cache 45.2%");
        let v_peak = VllmRawMetrics {
            prompt_tokens_mean: Some(18.0),
            kv_cache_usage_perc: Some(40.0),
            kv_cache_peak_perc: Some(92.0),
            ..Default::default()
        };
        assert_eq!(
            vllm_prompt_value(&v_peak, true),
            "18 tok | kv_cache 40.0% (peak 92.0%)"
        );
        let peak_below_threshold = VllmRawMetrics {
            prompt_tokens_mean: Some(18.0),
            kv_cache_usage_perc: Some(40.0),
            kv_cache_peak_perc: Some(84.0),
            ..Default::default()
        };
        assert_eq!(
            vllm_prompt_value(&peak_below_threshold, true),
            "18 tok | kv_cache 40.0%"
        );
        let peak_not_above_last = VllmRawMetrics {
            prompt_tokens_mean: Some(18.0),
            kv_cache_usage_perc: Some(92.0),
            kv_cache_peak_perc: Some(92.0),
            ..Default::default()
        };
        assert_eq!(
            vllm_prompt_value(&peak_not_above_last, true),
            "18 tok | kv_cache 92.0%"
        );
        let no_kv = VllmRawMetrics {
            prompt_tokens_mean: Some(512.0),
            ..Default::default()
        };
        assert_eq!(vllm_prompt_value(&no_kv, true), "512 tok | kv_cache —");
    }

    #[test]
    fn vllm_latency_value_default_ttft_tpot_only() {
        let v = VllmRawMetrics {
            ttft_ms: Some(120.0),
            tpot_ms: Some(50.0),
            prefill_latency_ms: Some(200.0),
            queue_delay_ms: Some(10.0),
            ..Default::default()
        };
        assert_eq!(vllm_latency_value(&v, false), "ttft 120ms | tpot 50ms");
        assert_eq!(
            vllm_latency_value(&v, true),
            "ttft 120ms | tpot 50ms | prefill 200ms | queue 10ms"
        );
    }

    #[test]
    fn vllm_label_row_aligns_labels_and_gap_before_metrics() {
        let line = vllm_label_row("REQUESTS", "run 2 (0.8%) | wait 1 | max 256");
        assert!(line.starts_with("REQUESTS"));
        assert!(line.contains(" run 2"));
        let t = vllm_label_row("THROUGHPUT", "59 tok/s | pfix_cache 72.8%");
        assert!(t.starts_with("THROUGHPUT"));
        assert!(t.contains(" 59 tok/s"));
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
            gpu: GpuRawMetrics {
                gpu_name: Some("Test GPU".into()),
                ..Default::default()
            },
        };
        let result = DiagnoseResult {
            snapshot: idle_snap.clone(),
            windows: vec![RuntimeWindow::from_snapshot(idle_snap)],
            static_ctx: StaticContext::default(),
            duration: Duration::from_secs(2),
            started_at: UNIX_EPOCH,
            any_evaluable: false,
            metrics_input: "http://127.0.0.1:8000/metrics".into(),
        };
        let lines = build_diagnose_lines(&result, false);
        let text = lines.join("\n");
        assert!(text.contains("Target:") && text.contains("127.0.0.1"));
        assert!(text.contains("No qualifying load"));
        assert!(!text.contains("GPU =>"));
    }
}
