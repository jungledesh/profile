use super::DiagnoseArgs;
use crate::profiler;

const LABEL_W: usize = 24;

fn row(label: &str, value: impl std::fmt::Display) {
    println!("  {:<w$} : {}", label, value, w = LABEL_W);
}

pub fn execute(args: &DiagnoseArgs) -> anyhow::Result<()> {
    let result = profiler::run_diagnose(&args.url, args.max_num_seqs)?;

    match &result.snapshot.gpu.gpu_name {
        Some(name) => row("GPU name", name),
        None => row("GPU name", "(no GPU / NVML not ready)"),
    }

    match result.snapshot.gpu.gpu_index {
        Some(i) => row("GPU index", i),
        None => row("GPU index", "(no GPU / NVML not ready)"),
    }

    match &result.snapshot.gpu.gpu_uuid {
        Some(id) => row("GPU ID (UUID)", id),
        None => row("GPU ID (UUID)", "(no GPU / NVML not ready)"),
    }

    match result.snapshot.gpu.gpu_util_pct {
        Some(util) => row("GPU util %", format!("{:.1}", util)),
        None => row("GPU util %", "(no GPU / NVML not ready)"),
    }

    match result.snapshot.gpu.mem_util_pct {
        Some(pct) => row("Mem ctrl util %", format!("{:.1}", pct)),
        None => row("Mem ctrl util %", "(no GPU / NVML not ready)"),
    }

    match (
        result.snapshot.gpu.vram_used_mb,
        result.snapshot.gpu.vram_total_mb,
    ) {
        (Some(used), Some(total)) => {
            if total > 0 {
                let pct = (used as f64 / total as f64) * 100.0;
                row(
                    "VRAM % used",
                    format!("{} / {} MiB ({:.1})", used, total, pct),
                );
            } else {
                row("VRAM % used", format!("{} / {} MiB", used, total));
            }
        }
        (Some(used), None) => row("VRAM % used", format!("{} MiB used (total n/a)", used)),
        (None, Some(total)) => row("VRAM % used", format!("(n/a) / {} MiB", total)),
        (None, None) => row("VRAM % used", "(no GPU / NVML not ready)"),
    }

    match (
        result.snapshot.gpu.power_watts,
        result.snapshot.gpu.power_limit_watts,
    ) {
        (Some(draw), Some(limit)) => {
            row("Power draw", format!("{:.0} / {:.0} W", draw, limit));
        }
        (Some(draw), None) => row("Power draw", format!("{:.1} W (limit n/a)", draw)),
        (None, Some(limit)) => row("Power draw", format!("(n/a) / {:.0} W", limit)),
        (None, None) => row("Power draw", "(no GPU / NVML not ready)"),
    }

    match result.snapshot.gpu.sm_clock_mhz {
        Some(mhz) => row("SM clock", format!("{} MHz", mhz)),
        None => row("SM clock", "(not available)"),
    }

    match result.snapshot.vllm.num_requests_running {
        Some(n) => row("In-batch reqs", format!("{:.1} (avg 2s)", n)),
        None => row("In-batch reqs", "(not parsed)"),
    }

    match result.snapshot.vllm.num_requests_waiting {
        Some(n) => row("Waiting reqs", format!("{:.1} (avg 2s)", n)),
        None => row("Waiting reqs", "(not parsed)"),
    }

    match result.snapshot.vllm.max_num_seqs {
        Some(n) => row("Max seqs", n),
        None => row("Max seqs", "—"),
    }

    match result.snapshot.vllm.ttft_ms {
        Some(ms) => row("TTFT (est. ms)", format!("{:.1} (window)", ms)),
        None => row("TTFT (est. ms)", "(n/a)"),
    }

    match result.snapshot.vllm.prefill_latency_ms {
        Some(ms) => row("Prefill ms", format!("{:.1} (window)", ms)),
        None => row("Prefill ms", "(n/a)"),
    }

    match result.snapshot.vllm.queue_delay_ms {
        Some(ms) => row("Queue ms", format!("{:.1} (window)", ms)),
        None => row("Queue ms", "(n/a)"),
    }

    match result.snapshot.vllm.tpot_ms {
        Some(ms) => row("TPOT ms", format!("{:.1} (window)", ms)),
        None => row("TPOT ms", "(n/a)"),
    }

    match result.snapshot.vllm.prompt_tokens_mean {
        Some(t) => row("Prompt mean", format!("{:.1} tok (window)", t)),
        None => row("Prompt mean", "(n/a)"),
    }

    match result.snapshot.vllm.generation_tokens_per_sec {
        Some(tps) => row("Gen tok/s", format!("{:.1} (window)", tps)),
        None => row("Gen tok/s", "(n/a)"),
    }

    match result.snapshot.vllm.prefix_cache_hit_rate {
        Some(0.0) => row("Prefix cache hit rate (last scrape)", "0%"),
        Some(r) => row(
            "Prefix cache hit rate (last scrape)",
            format!("{:.1}%", r * 100.0),
        ),
        None => row("Prefix cache hit rate (last scrape)", "(n/a)"),
    }

    match result.snapshot.vllm.generation_tokens_total {
        Some(n) => row("Gen tokens", format!("{:.0} (total)", n)),
        None => row("Gen tokens", "(not parsed)"),
    }

    if result.snapshot.gpu.gpu_util_pct.is_none() && !result.snapshot.vllm.has_scrape_data() {
        println!(
            "\n  (No metrics in snapshot — NVML unavailable or vLLM scrape not implemented yet.)"
        );
    } else {
        println!("\n  Snapshot collected; rule engine and richer /metrics fields still TODO.");
    }

    Ok(())
}

#[cfg(test)]
#[test]
fn metric_line_padding_matches_longest_label() {
    let s = format!("  {:<w$} : {}", "Mem ctrl util %", "12.0", w = LABEL_W);
    assert_eq!(s.find(" : ").unwrap(), 2 + LABEL_W);
    assert_eq!(s[2..2 + LABEL_W].trim_end(), "Mem ctrl util %");
}

#[cfg(test)]
#[test]
fn metric_line_shorter_label_pads_before_colon() {
    let s = format!("  {:<w$} : {}", "VRAM % used", "1 / 2 MiB", w = LABEL_W);
    assert!(s.starts_with("  VRAM % used") && s.contains(" : ") && s.ends_with("1 / 2 MiB"));
}
