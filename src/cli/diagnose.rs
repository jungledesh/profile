//! `diagnose` subcommand: render snapshot as a boxed table (metrics + rule diagnose blocks).
//!
//! Layout: **GPU =>** (NVML); **vLLM:** REQUESTS / LATENCY / PROMPT / THROUGHPUT rows (aligned labels).

use crate::collectors::{GpuRawMetrics, RawSnapshot, VllmRawMetrics};
use crate::engine;
use crate::profiler;

/// Width for REQUESTS / LATENCY / PROMPT / THROUGHPUT label column (matches **THROUGHPUT**).
const VLLM_LABEL_W: usize = 10;

/// Space between label column and metric values (after `vLLM:` block labels).
const VLLM_LABEL_METRICS_GAP: &str = " ";

pub fn execute(
    vllm_metrics_input: &str,
    max_num_seqs: u32,
    verbose_rules: bool,
) -> anyhow::Result<()> {
    let result = profiler::run_diagnose(vllm_metrics_input, max_num_seqs)?;
    print_diagnose_table(&result.snapshot, verbose_rules);
    Ok(())
}

fn print_diagnose_table(snapshot: &RawSnapshot, verbose_rules: bool) {
    let lines = build_diagnose_lines(snapshot, verbose_rules);
    print_boxed(&lines);
}

fn build_diagnose_lines(snapshot: &RawSnapshot, verbose_rules: bool) -> Vec<String> {
    let v = &snapshot.vllm;
    let g = &snapshot.gpu;

    let model = v.model_name.as_deref().unwrap_or("(unknown model)");
    let gpu_label = g.gpu_name.as_deref().unwrap_or("(no GPU)");

    let mut lines = vec![format!(
        "PROFILE v{} [{}] [{}]",
        env!("CARGO_PKG_VERSION"),
        model,
        gpu_label
    )];

    lines.push(format!("GPU => {}", gpu_gauges_line(g)));
    lines.push(String::new());
    lines.push("vLLM:".to_string());
    lines.push(vllm_label_row("REQUESTS", &vllm_requests_value(v)));
    lines.push(vllm_label_row("LATENCY", &vllm_latency_value(v)));
    lines.push(vllm_label_row("PROMPT", &vllm_prompt_value(v)));
    lines.push(vllm_label_row("THROUGHPUT", &vllm_throughput_value(v)));

    let rule_lines = engine::format_diagnose_rules(snapshot, verbose_rules);
    if !rule_lines.is_empty() {
        lines.push(String::new());
        lines.extend(rule_lines);
    }

    lines
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

/// Print `lines` in a single ASCII box (inner width = longest line).
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

/// GPU row: UTIL | POWER | MEM (NVML).
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
            format!("MEM {:.0}/{:.0}GB", u_gb, t_gb)
        }
        _ => "MEM —".to_string(),
    };

    format!("{util} | {power} | {mem}")
}

/// #1 / #2 / #9 — `run | wait | max`
fn vllm_requests_value(v: &VllmRawMetrics) -> String {
    let run = v
        .num_requests_running
        .map(fmt_gauge)
        .map(|s| format!("run {s}"))
        .unwrap_or_else(|| "run —".to_string());
    let wait = v
        .num_requests_waiting
        .map(fmt_gauge)
        .map(|s| format!("wait {s}"))
        .unwrap_or_else(|| "wait —".to_string());
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

/// Histogram means — `ttft | tpot | prefill | queue`
fn vllm_latency_value(v: &VllmRawMetrics) -> String {
    let ttft = v
        .ttft_ms
        .map(fmt_seconds_from_ms)
        .unwrap_or_else(|| "—".to_string());
    let tpot = v
        .tpot_ms
        .map(fmt_seconds_from_ms)
        .unwrap_or_else(|| "—".to_string());
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

/// Prompt mean + KV cache fill (`vllm_kv_cache_usage_perc` gauge, 0–100).
fn vllm_prompt_value(v: &VllmRawMetrics) -> String {
    let n = v
        .prompt_tokens_mean
        .map(fmt_tok)
        .unwrap_or_else(|| "—".to_string());
    let kv = match v.kv_cache_usage_perc.filter(|x| x.is_finite()) {
        Some(p) => format!("kv_cache {:.1}%", p),
        None => "kv_cache —".to_string(),
    };
    format!("{n} tok | {kv}")
}

fn fmt_tok(t: f64) -> String {
    if (t - t.round()).abs() < 1e-6 {
        format!("{:.0}", t)
    } else {
        format!("{:.1}", t)
    }
}

/// Output tok/s from **Δ** `vllm_generation_tokens_total` (or iteration-token fallback) over the
/// scrape window, divided by wall time — not the cumulative counter value itself.
/// Prefix hit rate: `pfix_cache` (Δhits/Δqueries in window).
fn vllm_throughput_value(v: &VllmRawMetrics) -> String {
    let tps = v
        .generation_tokens_per_sec
        .map(|t| format!("{:.0} tok/s", t))
        .unwrap_or_else(|| "— tok/s".to_string());
    let cache = cache_use_fragment(v);
    format!("{tps} | {cache}")
}

/// Prefix cache use % from #7/#8 (Δhits/Δqueries).
fn cache_use_fragment(v: &VllmRawMetrics) -> String {
    match v.prefix_cache_hit_rate {
        Some(0.0) => "pfix_cache 0%".to_string(),
        Some(r) => format!("pfix_cache {:.1}%", r * 100.0),
        None => "pfix_cache —".to_string(),
    }
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

    #[test]
    fn fmt_seconds_from_ms_prefers_seconds_when_large() {
        assert_eq!(fmt_seconds_from_ms(1200.0), "1.2s");
        assert_eq!(fmt_seconds_from_ms(50.0), "50ms");
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
        assert!(s.contains("MEM 72/80GB"));
    }

    #[test]
    fn vllm_requests_value_run_wait_max() {
        let v = VllmRawMetrics {
            num_requests_running: Some(2.0),
            num_requests_waiting: Some(1.0),
            max_num_seqs: Some(256),
            ..Default::default()
        };
        assert_eq!(vllm_requests_value(&v), "run 2 | wait 1 | max 256");
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
    fn vllm_prompt_value_includes_kv_cache() {
        let v = VllmRawMetrics {
            prompt_tokens_mean: Some(18.0),
            kv_cache_usage_perc: Some(45.25),
            ..Default::default()
        };
        assert_eq!(vllm_prompt_value(&v), "18 tok | kv_cache 45.2%");
        let no_kv = VllmRawMetrics {
            prompt_tokens_mean: Some(512.0),
            ..Default::default()
        };
        assert_eq!(vllm_prompt_value(&no_kv), "512 tok | kv_cache —");
    }

    #[test]
    fn vllm_label_row_aligns_labels_and_gap_before_metrics() {
        let line = vllm_label_row("REQUESTS", "run 2 | wait 1 | max 256");
        assert!(line.starts_with("REQUESTS"));
        assert!(line.contains(" run 2"));
        let t = vllm_label_row("THROUGHPUT", "59 tok/s | pfix_cache 72.8%");
        assert!(t.starts_with("THROUGHPUT"));
        assert!(t.contains(" 59 tok/s"));
    }
}
