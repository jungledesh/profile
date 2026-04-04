//! `diagnose` subcommand: render snapshot as a compact table (metrics + stub WASTE/FIX).

use crate::collectors::{GpuRawMetrics, RawSnapshot, VllmRawMetrics};
use crate::profiler;

const DIVIDER: &str = "---------------------------------------------------------------------------";

/// Placeholder until the rule engine fills WASTE/FIX dynamically.
const WASTE_FIX_STUB: &str = r"WASTE
01 MOVEMENT   70% GPU idle (decode)
02 SCHEDULER  batch collapse (2/16)

FIX
+-----------------------------------------------------------------------+
| ENABLE CONTINUOUS BATCHING                                            |
| +45% TPS   | decode underutilized   | queue 200ms → 0ms               |
+-----------------------------------------------------------------------+
| INCREASE BATCH WINDOW (15ms)                                          |
| -12% cost/token   | small prompt overhead                             |
+-----------------------------------------------------------------------+
";

pub fn execute(vllm_metrics_input: &str, max_num_seqs: u32) -> anyhow::Result<()> {
    let result = profiler::run_diagnose(vllm_metrics_input, max_num_seqs)?;
    print_diagnose_table(&result.snapshot);
    Ok(())
}

fn print_diagnose_table(snapshot: &RawSnapshot) {
    let v = &snapshot.vllm;
    let g = &snapshot.gpu;

    let model = v.model_name.as_deref().unwrap_or("(unknown model)");
    let gpu_label = g.gpu_name.as_deref().unwrap_or("(no GPU)");

    println!(
        "PROFILE v{} [{}] [{}]",
        env!("CARGO_PKG_VERSION"),
        model,
        gpu_label
    );
    println!("{DIVIDER}");
    println!("{}", gauges_line(g));
    println!("{}", counters_line(v));
    println!("{}", histograms_line(v));
    println!("{DIVIDER}");
    println!();
    println!("{WASTE_FIX_STUB}");
}

/// Row 2 — gauges (GPU util, power, VRAM).
fn gauges_line(g: &GpuRawMetrics) -> String {
    let util = g
        .gpu_util_pct
        .map(|u| format!("UTIL {:.0}%", u))
        .unwrap_or_else(|| "UTIL —".to_string());

    let power = match (g.power_watts, g.power_limit_watts) {
        (Some(draw), Some(limit)) if limit > 0.0 => {
            let pct = (draw / limit) * 100.0;
            format!("POWER {:.0}W ({:.0}%)", draw, pct)
        }
        (Some(draw), _) => format!("POWER {:.0}W", draw),
        _ => "POWER —".to_string(),
    };

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

/// Row 3 — counters (throughput, TTFT, P99 placeholder).
fn counters_line(v: &VllmRawMetrics) -> String {
    let tps = v
        .generation_tokens_per_sec
        .map(|t| format!("TPS {:.0}", t))
        .unwrap_or_else(|| "TPS —".to_string());

    let ttft = v
        .ttft_ms
        .map(fmt_seconds_from_ms)
        .map(|s| format!("TTFT {s}"))
        .unwrap_or_else(|| "TTFT —".to_string());

    let p99 = "P99 —";

    format!("{tps} | {ttft} | {p99}")
}

/// Row 4 — histogram window means (same fields as vLLM window estimates).
fn histograms_line(v: &VllmRawMetrics) -> String {
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

    let prefix_cache_use = prefix_cache_use_pct(v);

    format!("HIST TTFT {ttft} | TPOT {tpot} | PREF {prefill} | QUEUE {queue} | {prefix_cache_use}")
}

/// Prefix cache: window hit rate only (no per-scrape hits/queries).
fn prefix_cache_use_pct(v: &VllmRawMetrics) -> String {
    match v.prefix_cache_hit_rate {
        Some(0.0) => "prefix cache use 0%".to_string(),
        Some(r) => format!("prefix cache use {:.1}%", r * 100.0),
        None => "prefix cache use —".to_string(),
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
    fn prefix_cache_use_pct_formats_hit_rate_only() {
        assert_eq!(
            prefix_cache_use_pct(&VllmRawMetrics::default()),
            "prefix cache use —"
        );
        assert_eq!(
            prefix_cache_use_pct(&VllmRawMetrics {
                prefix_cache_hit_rate: Some(0.0),
                ..Default::default()
            }),
            "prefix cache use 0%"
        );
        assert_eq!(
            prefix_cache_use_pct(&VllmRawMetrics {
                prefix_cache_hit_rate: Some(0.728),
                ..Default::default()
            }),
            "prefix cache use 72.8%"
        );
    }

    #[test]
    fn gauges_line_formats_mem_gb() {
        let g = GpuRawMetrics {
            gpu_util_pct: Some(28.0),
            power_watts: Some(310.0),
            power_limit_watts: Some(400.0),
            vram_used_mb: Some(72 * 1024),
            vram_total_mb: Some(80 * 1024),
            ..Default::default()
        };
        let s = gauges_line(&g);
        assert!(s.contains("UTIL 28%"));
        assert!(s.contains("POWER 310W"));
        assert!(s.contains("MEM 72/80GB"));
    }
}
