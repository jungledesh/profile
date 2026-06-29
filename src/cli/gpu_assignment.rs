#[cfg(target_os = "linux")]
use std::collections::HashMap;
use std::collections::{BTreeSet, HashSet};
use std::io::{self, BufRead, IsTerminal, Write};

use nvml_wrapper::Nvml;

use crate::cli::diagnose::{TP_ABORT_HINT, prompt_u32_required};

const VRAM_ACTIVE_THRESHOLD: f64 = 0.20;

/// Resolved tensor-parallel degree and the exact GPU indices to collect from.
/// `indices` is always non-empty after resolution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GpuAssignment {
    pub tp: u32,
    pub indices: Vec<u32>,
}

struct GpuSnapshot {
    idx: u32,
    name: String,
    vram_pct: f64,
    vram_total_mb: u64,
    pids: Vec<u32>,
}

struct Term;

impl Term {
    fn new() -> Self {
        Self
    }

    fn scan_header(&self, n: u32) -> String {
        format!("Scanning {n} GPUs via NVML...")
    }

    fn format_gpu_row(&self, row: &GpuSnapshot) -> String {
        let filled = ((row.vram_pct / 10.0).round() as usize).min(10);
        let empty = 10usize.saturating_sub(filled);
        let bar = format!("{}{}", "█".repeat(filled), "░".repeat(empty));
        let pct = format!("{:>3.0}% vRAM", row.vram_pct.round());
        let status = if row.vram_pct >= VRAM_ACTIVE_THRESHOLD * 100.0 {
            if !row.pids.is_empty() {
                let pids_str = row
                    .pids
                    .iter()
                    .map(|p| p.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                let prefix = if row.pids.len() > 1 { "pids" } else { "pid" };
                format!("{prefix} {pids_str}")
            } else {
                "active".to_string()
            }
        } else {
            "idle".to_string()
        };
        format!(
            "  [{}] {:<12}  {}  {}  {}",
            row.idx,
            short_gpu_name(&row.name),
            bar,
            pct,
            status
        )
    }

    fn step1_success(&self, indices: &[u32], tp: u32) -> String {
        let idxs = indices
            .iter()
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        format!("Detected vLLM on GPUs [{idxs}]: TP = {tp} inferred")
    }

    #[cfg(target_os = "linux")]
    fn step2_progress(&self) -> String {
        "Scanning GPU processes via ps...".to_string()
    }

    #[cfg(target_os = "linux")]
    fn step2_success(&self, indices: &[u32], tp: u32) -> String {
        let idxs = indices
            .iter()
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        format!("Detected vLLM on GPUs [{idxs}]: TP = {tp} inferred")
    }

    fn step3_header(&self, pid_list: &str) -> String {
        if pid_list.is_empty() {
            "[!] No GPU processes detected. Enter TP and GPU indices manually.".to_string()
        } else {
            format!(
                "[!] Cannot find vLLM GPUs. Run: ps -f -p {pid_list}, match vLLM PID to GPUs above."
            )
        }
    }

    fn prompt_tp(&self, host_gpus: u32) -> String {
        format!("Enter value for --tensor-parallel-size ({host_gpus} gpus detected): ")
    }

    fn prompt_indices(&self) -> String {
        "Enter value for --gpu-indices (e.g. 0,1): ".to_string()
    }
}

fn short_gpu_name(name: &str) -> String {
    name.chars().take(12).collect()
}

pub(crate) fn resolve_gpu_assignment(
    cli_tp: Option<u32>,
    url: &str,
) -> anyhow::Result<GpuAssignment> {
    resolve_gpu_assignment_for_host(crate::collectors::gpu::host_gpu_count(), cli_tp, url)
}

pub(crate) fn resolve_gpu_assignment_for_host(
    host_count: Option<u32>,
    cli_tp: Option<u32>,
    url: &str,
) -> anyhow::Result<GpuAssignment> {
    match host_count {
        None => {
            if let Some(tp) = cli_tp {
                if tp > 1 {
                    anyhow::bail!(
                        "NVML unavailable. Cannot verify hardware for TP {}. Is the GPU driver installed?",
                        tp
                    );
                } else {
                    return Ok(GpuAssignment {
                        tp,
                        indices: (0..tp).collect(),
                    });
                }
            }
            anyhow::bail!("NVML unavailable. Is the GPU driver installed?")
        }
        Some(0) => anyhow::bail!("No GPUs detected."),
        Some(1) => {
            if let Some(tp) = cli_tp
                && tp > 1
            {
                anyhow::bail!("--tensor-parallel-size {tp} exceeds detected GPU count (1).")
            }
            Ok(GpuAssignment {
                tp: 1,
                indices: vec![0],
            })
        }
        Some(n) => {
            if let Some(tp) = cli_tp {
                if tp > n {
                    anyhow::bail!("--tensor-parallel-size {tp} exceeds detected GPU count ({n}).")
                } else {
                    run_pipeline(n, Some(tp), url)
                }
            } else {
                run_pipeline(n, None, url)
            }
        }
    }
}

fn run_pipeline(
    host_count: u32,
    known_tp: Option<u32>,
    #[cfg_attr(not(target_os = "linux"), allow(unused_variables))] url: &str,
) -> anyhow::Result<GpuAssignment> {
    let nvml = Nvml::init()?;
    let term = Term::new();

    eprintln!("{}", term.scan_header(host_count));

    let snapshots = collect_gpu_snapshots(&nvml, host_count);
    let fracs: Vec<Option<f64>> = snapshots.iter().map(|s| Some(s.vram_pct / 100.0)).collect();
    let model_weight_gb = preflight_model_weight_gb(url);
    let vram_total_gb: Vec<f64> = snapshots
        .iter()
        .map(|s| s.vram_total_mb as f64 / 1024.0)
        .collect();

    if let Some(a) = vram_heuristic_from_fracs_inner(
        host_count,
        &fracs,
        known_tp,
        model_weight_gb,
        &vram_total_gb,
    ) {
        for row in snapshots {
            eprintln!("{}", term.format_gpu_row(&row));
        }
        eprintln!();
        eprintln!("{}", term.step1_success(&a.indices, a.tp));
        eprintln!();
        return Ok(a);
    }

    for row in &snapshots {
        eprintln!("{}", term.format_gpu_row(row));
    }
    eprintln!();

    #[cfg(target_os = "linux")]
    if let Some(a) = ps_tiebreaker(&snapshots, known_tp, &term) {
        return Ok(a);
    }

    interactive_gpu_prompt(host_count, known_tp, &snapshots, &term)
}

fn collect_gpu_snapshots(nvml: &Nvml, host_count: u32) -> Vec<GpuSnapshot> {
    let mut out = Vec::with_capacity(host_count as usize);
    for idx in 0..host_count {
        let device = nvml.device_by_index(idx).ok();
        let name = device
            .as_ref()
            .and_then(|d| d.name().ok())
            .unwrap_or_else(|| "GPU".to_string());
        let mem = device.as_ref().and_then(|d| d.memory_info().ok());
        let vram_pct = mem
            .as_ref()
            .map(|m| {
                if m.total == 0 {
                    0.0
                } else {
                    m.used as f64 / m.total as f64 * 100.0
                }
            })
            .unwrap_or(0.0);
        let vram_total_mb = mem.map(|m| m.total / (1024 * 1024)).unwrap_or(0);
        let pids = device
            .as_ref()
            .and_then(|d| d.running_compute_processes().ok())
            .map(|procs| procs.iter().map(|p| p.pid).collect())
            .unwrap_or_default();
        out.push(GpuSnapshot {
            idx,
            name,
            vram_pct,
            vram_total_mb,
            pids,
        });
    }
    out
}

/// Fetch model weight in GB from the vLLM `/v1/models` endpoint + catalog.
/// Returns None on any failure — callers fall back to VRAM_ACTIVE_THRESHOLD.
fn preflight_model_weight_gb(url: &str) -> Option<f64> {
    let base = {
        let (scheme, rest) = url.split_once("://")?;
        let host = rest.split('/').next()?;
        format!("{scheme}://{host}")
    };
    let models_url = format!("{base}/v1/models");
    let body = reqwest::blocking::Client::builder()
        .use_rustls_tls()
        .timeout(std::time::Duration::from_secs(2))
        .build()
        .ok()?
        .get(&models_url)
        .send()
        .ok()?
        .text()
        .ok()?;
    let json: serde_json::Value = serde_json::from_str(&body).ok()?;
    let model_id = json["data"][0]["id"].as_str()?;
    let entry = crate::context::model_catalog::lookup_model(model_id)?;
    let bits: u8 = match entry.default_weight_dtype {
        "fp8" | "e4m3" | "e5m2" => 8,
        "fp16" | "bf16" => 16,
        "fp32" => 32,
        _ => 16,
    };
    Some(crate::engine::baseline::weight_gb(entry.param_count, bits))
}

#[cfg(test)]
pub(crate) fn vram_heuristic_from_fracs(
    host_count: u32,
    fracs: &[Option<f64>],
    known_tp: Option<u32>,
) -> Option<GpuAssignment> {
    vram_heuristic_from_fracs_inner(host_count, fracs, known_tp, None, &[])
}

/// Internal implementation. `model_weight_gb` and `vram_total_gb_per_gpu` enable a
/// physics-based active threshold; both fall back to VRAM_ACTIVE_THRESHOLD when absent.
fn vram_heuristic_from_fracs_inner(
    host_count: u32,
    fracs: &[Option<f64>],
    known_tp: Option<u32>,
    model_weight_gb: Option<f64>,
    vram_total_gb_per_gpu: &[f64],
) -> Option<GpuAssignment> {
    let mut active = Vec::new();
    for idx in 0..host_count {
        let frac = fracs.get(idx as usize).copied().flatten()?;
        // Physics-based threshold: weight_gb × 0.80 / host_count / gpu_vram_gb.
        // Dividing by host_count is the worst-case (max TP) — ensures we catch
        // any GPU holding even a fraction of model weights.
        // Buffer of 0.80 covers catalog dtype/quantization uncertainty.
        // Falls back to VRAM_ACTIVE_THRESHOLD when model or VRAM data unavailable.
        let threshold = match (model_weight_gb, vram_total_gb_per_gpu.get(idx as usize)) {
            (Some(w_gb), Some(&total_gb)) if total_gb > 0.0 => {
                (w_gb * 0.80 / host_count as f64 / total_gb).min(VRAM_ACTIVE_THRESHOLD)
            }
            _ => VRAM_ACTIVE_THRESHOLD,
        };
        if frac >= threshold {
            active.push(idx);
        }
    }
    if active.is_empty() {
        return None;
    }
    // All GPUs active: unambiguous — only one workload can own all of them.
    if active.len() == host_count as usize {
        return Some(GpuAssignment {
            tp: active.len() as u32,
            indices: active,
        });
    }
    match known_tp {
        Some(tp) if active.len() != tp as usize => return None,
        // No --tp hint: auto-resolve only when exactly 1 GPU is active.
        // Multiple active GPUs without a TP hint is ambiguous — fall through to Step 2.
        None if active.len() != 1 => return None,
        _ => {}
    }
    Some(GpuAssignment {
        tp: active.len() as u32,
        indices: active,
    })
}

/// Tiebreaker: run `ps -f -p <all GPU PIDs>`, find lines containing "vllm",
/// map those PIDs back to GPU indices from the snapshots.
/// More reliable than /proc socket walking — /proc/{pid}/cmdline is world-readable
/// even when /proc/{pid}/fd/ is not.
#[cfg(target_os = "linux")]
fn ps_tiebreaker(
    snapshots: &[GpuSnapshot],
    known_tp: Option<u32>,
    term: &Term,
) -> Option<GpuAssignment> {
    // Build pid → [gpu_indices] map from what NVML already gave us.
    let mut pid_to_gpus: HashMap<u32, Vec<u32>> = HashMap::new();
    for snap in snapshots {
        for &pid in &snap.pids {
            pid_to_gpus.entry(pid).or_default().push(snap.idx);
        }
    }
    if pid_to_gpus.is_empty() {
        return None;
    }

    eprintln!("{}", term.step2_progress());

    let pid_list = pid_to_gpus
        .keys()
        .map(|p| p.to_string())
        .collect::<Vec<_>>()
        .join(",");

    let output = std::process::Command::new("ps")
        .args(["-f", "-p", &pid_list])
        .output()
        .ok()?;

    let stdout = String::from_utf8_lossy(&output.stdout);

    // ps -f columns: UID PID PPID C STIME TTY TIME CMD...
    // Skip header; match lines where the full command contains "vllm".
    let mut gpu_indices: BTreeSet<u32> = BTreeSet::new();
    for line in stdout.lines().skip(1) {
        if !line.contains("vllm") {
            continue;
        }
        let mut fields = line.split_whitespace();
        let _uid = fields.next();
        let pid: u32 = fields.next().and_then(|s| s.parse().ok())?;
        if let Some(gpus) = pid_to_gpus.get(&pid) {
            gpu_indices.extend(gpus);
        }
    }

    if gpu_indices.is_empty() {
        return None;
    }

    let indices: Vec<u32> = gpu_indices.into_iter().collect();
    let tp = indices.len() as u32;

    if let Some(expected) = known_tp
        && tp != expected
    {
        return None;
    }

    eprintln!("{}", term.step2_success(&indices, tp));
    eprintln!();
    Some(GpuAssignment { tp, indices })
}

fn interactive_gpu_prompt(
    host_count: u32,
    known_tp: Option<u32>,
    snapshots: &[GpuSnapshot],
    term: &Term,
) -> anyhow::Result<GpuAssignment> {
    let mut hint_pids = BTreeSet::new();
    for row in snapshots {
        for pid in &row.pids {
            hint_pids.insert(*pid);
        }
    }
    let pid_list = hint_pids
        .iter()
        .map(|p| p.to_string())
        .collect::<Vec<_>>()
        .join(",");

    eprintln!();
    eprintln!("{}", term.step3_header(&pid_list));
    eprintln!();

    if !io::stdin().is_terminal() {
        anyhow::bail!(
            "Cannot infer GPU assignment automatically. Please pass --tensor-parallel-size and/or run on Linux."
        );
    }

    let tp = if let Some(tp) = known_tp {
        tp
    } else {
        let prompt = term.prompt_tp(host_count);
        let v = prompt_u32_required(&mut io::stdin().lock(), &mut io::stderr(), &prompt)?;
        eprintln!();
        v
    };

    let indices = prompt_gpu_indices(
        &mut io::stdin().lock(),
        &mut io::stderr(),
        tp,
        host_count,
        term,
    )?;
    let idxs = indices
        .iter()
        .map(|i| i.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    eprintln!("Locked to GPUs [{idxs}]: TP = {tp}");
    eprintln!();
    Ok(GpuAssignment { tp, indices })
}

fn prompt_gpu_indices<R: BufRead, W: Write>(
    reader: &mut R,
    writer: &mut W,
    tp: u32,
    host_count: u32,
    term: &Term,
) -> anyhow::Result<Vec<u32>> {
    const MAX_ATTEMPTS: u8 = 4;
    let mut attempts: u8 = 0;
    let prompt = term.prompt_indices();
    loop {
        write!(writer, "{prompt}")?;
        writer.flush()?;
        let mut line = String::new();
        reader.read_line(&mut line)?;
        match parse_gpu_indices_line(&line, tp, host_count) {
            Ok(indices) => return Ok(indices),
            Err(msg) => {
                attempts += 1;
                if attempts >= MAX_ATTEMPTS {
                    anyhow::bail!("Too many invalid attempts. {TP_ABORT_HINT}");
                }
                writeln!(writer, "{msg}")?;
            }
        }
    }
}

pub(crate) fn parse_gpu_indices_line(
    line: &str,
    tp: u32,
    host_count: u32,
) -> Result<Vec<u32>, String> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return Err("enter at least one GPU index.".into());
    }
    let max_idx = host_count.saturating_sub(1);
    let mut indices = Vec::new();
    for part in trimmed.split(',') {
        let part = part.trim();
        if part.is_empty() {
            return Err("expected a positive integer, got empty token".into());
        }
        let i: u32 = part
            .parse()
            .map_err(|_| format!("expected a positive integer, got {part:?}"))?;
        if i >= host_count {
            return Err(format!("index {i} out of range (0–{max_idx})"));
        }
        indices.push(i);
    }
    let mut seen = HashSet::new();
    for &i in &indices {
        if !seen.insert(i) {
            return Err(format!("duplicate index {i}"));
        }
    }
    if indices.len() != tp as usize {
        return Err(format!("expected {tp} indices, got {}", indices.len()));
    }
    Ok(indices)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vram_heuristic_clean_split_returns_assignment() {
        let fracs = [Some(0.92), Some(0.91), Some(0.01), Some(0.02)];
        // With active.len() > 1 and known_tp == None, we now return None to force Step 2.
        let a = vram_heuristic_from_fracs(4, &fracs, None);
        assert!(a.is_none());
    }

    #[test]
    fn vram_heuristic_all_active_returns_assignment() {
        let fracs = [Some(0.92), Some(0.91)];
        let a = vram_heuristic_from_fracs(2, &fracs, None).expect("all active = unambiguous");
        assert_eq!(a.tp, 2);
        assert_eq!(a.indices, vec![0, 1]);
    }

    #[test]
    fn vram_heuristic_all_idle_returns_none() {
        let fracs = [Some(0.01), Some(0.02), Some(0.03), Some(0.04)];
        assert!(vram_heuristic_from_fracs(4, &fracs, None).is_none());
    }

    #[test]
    fn vram_heuristic_known_tp_mismatch_returns_none() {
        let fracs = [Some(0.92), Some(0.91), Some(0.01), Some(0.02)];
        assert!(vram_heuristic_from_fracs(4, &fracs, Some(4)).is_none());
    }

    #[test]
    fn interactive_prompt_valid_indices_parses() {
        assert_eq!(parse_gpu_indices_line("0,1", 2, 4).unwrap(), vec![0, 1]);
    }

    #[test]
    fn interactive_prompt_rejects_count_mismatch() {
        assert!(parse_gpu_indices_line("0,1,2", 2, 4).is_err());
    }

    #[test]
    fn interactive_prompt_rejects_out_of_range_index() {
        let err = parse_gpu_indices_line("0,4", 2, 4).unwrap_err();
        assert!(err.contains("out of range"));
    }

    #[test]
    fn interactive_prompt_rejects_duplicate_indices() {
        let err = parse_gpu_indices_line("0,0", 2, 4).unwrap_err();
        assert!(err.contains("duplicate index 0"));
    }

    #[test]
    fn interactive_prompt_aborts_after_four_bad_attempts() {
        let term = Term::new();
        let mut out = Vec::new();
        let err = prompt_gpu_indices(
            &mut std::io::Cursor::new(b"\n\n\n\n"),
            &mut out,
            2,
            4,
            &term,
        )
        .unwrap_err();
        assert!(err.to_string().contains("Pass --tensor-parallel-size"));
    }

    #[test]
    fn resolve_single_gpu_host_without_cli() {
        let a = resolve_gpu_assignment_for_host(Some(1), None, "http://localhost:8000/metrics")
            .unwrap();
        assert_eq!(
            a,
            GpuAssignment {
                tp: 1,
                indices: vec![0],
            }
        );
    }

    #[test]
    fn resolve_cli_tp_exceeds_host_bails() {
        let err =
            resolve_gpu_assignment_for_host(Some(2), Some(4), "http://localhost:8000/metrics")
                .unwrap_err();
        assert!(err.to_string().contains("exceeds detected GPU count"));
    }

    #[test]
    fn resolve_cli_tp_without_nvml_uses_flag() {
        let a = resolve_gpu_assignment_for_host(None, Some(1), "http://localhost:8000/metrics")
            .unwrap();
        assert_eq!(a.tp, 1);
        assert_eq!(a.indices, vec![0]);
    }

    #[test]
    fn resolve_aborts_when_nvml_unavailable_and_no_cli() {
        let err = resolve_gpu_assignment_for_host(None, None, "http://localhost:8000/metrics")
            .unwrap_err();
        assert!(err.to_string().contains("NVML unavailable"));
    }

    #[test]
    fn resolve_aborts_when_no_gpus_and_no_cli() {
        let err = resolve_gpu_assignment_for_host(Some(0), None, "http://localhost:8000/metrics")
            .unwrap_err();
        assert!(err.to_string().contains("No GPUs detected"));
    }

    #[test]
    fn resolve_single_gpu_rejects_cli_tp_above_one() {
        let err =
            resolve_gpu_assignment_for_host(Some(1), Some(2), "http://localhost:8000/metrics")
                .unwrap_err();
        assert!(err.to_string().contains("exceeds detected GPU count (1)"));
    }

    #[test]
    fn resolve_no_gpus_bails_even_with_cli_tp() {
        let err =
            resolve_gpu_assignment_for_host(Some(0), Some(1), "http://localhost:8000/metrics")
                .unwrap_err();
        assert!(err.to_string().contains("No GPUs detected"));
    }

    #[test]
    fn gpu_row_formats_correctly() {
        let term = Term::new();
        let row = GpuSnapshot {
            idx: 0,
            name: "NVIDIA A100".to_string(),
            vram_pct: 92.0,
            vram_total_mb: 80 * 1024,
            pids: vec![40592],
        };
        let line = term.format_gpu_row(&row);
        assert!(!line.contains('\x1b'));
        assert!(line.contains("[0]"));
        assert!(line.contains("92% vRAM"));
    }
}
