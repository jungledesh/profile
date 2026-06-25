#[cfg(target_os = "linux")]
use std::collections::HashMap;
use std::collections::{BTreeSet, HashSet};
use std::io::{self, BufRead, IsTerminal, Write};

use nvml_wrapper::Nvml;

use crate::cli::diagnose::{TP_ABORT_HINT, prompt_u32_required};

const VRAM_ACTIVE_THRESHOLD: f64 = 0.70;

const DIM: &str = "\x1b[38;5;240m";
const GREEN: &str = "\x1b[38;5;114m";
const YELLOW: &str = "\x1b[38;5;178m";
const RED: &str = "\x1b[38;5;203m";
const BLUE: &str = "\x1b[38;5;110m";
const RESET: &str = "\x1b[0m";

/// Resolved tensor-parallel degree and the exact GPU indices to collect from.
/// `indices` is always non-empty after resolution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GpuAssignment {
    pub tp: u32,
    pub indices: Vec<u32>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum RowColorMode {
    Step1,
    Ambiguous,
}

struct GpuSnapshot {
    idx: u32,
    name: String,
    vram_pct: f64,
    pids: Vec<u32>,
}

struct Term {
    color: bool,
}

impl Term {
    fn new() -> Self {
        Self {
            color: color_enabled(),
        }
    }

    fn wrap(&self, color: &str, text: &str) -> String {
        if self.color {
            format!("{color}{text}{RESET}")
        } else {
            text.to_string()
        }
    }

    fn scan_header(&self, n: u32) -> String {
        self.wrap(DIM, &format!("scanning {n} gpus via nvml..."))
    }

    fn format_gpu_row(&self, row: &GpuSnapshot, mode: RowColorMode, highlight: bool) -> String {
        let filled = ((row.vram_pct / 10.0).round() as usize).min(10);
        let empty = 10usize.saturating_sub(filled);
        let bar_body = format!("{}{}", "█".repeat(filled), "░".repeat(empty));

        let pct_color = if mode == RowColorMode::Step1 && highlight && row.vram_pct >= 70.0 {
            GREEN
        } else if row.vram_pct >= 70.0 {
            YELLOW
        } else {
            DIM
        };

        let bar = self.wrap(pct_color, &bar_body);
        let pct = self.wrap(pct_color, &format!("{:>3.0}% vram", row.vram_pct.round()));

        let status = if row.vram_pct >= VRAM_ACTIVE_THRESHOLD * 100.0 {
            if !row.pids.is_empty() {
                let pids_str = row
                    .pids
                    .iter()
                    .map(|p| p.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                let prefix = if row.pids.len() > 1 { "pids" } else { "pid" };
                if self.color {
                    format!("{DIM}{prefix} {RESET}{BLUE}{pids_str}{RESET}")
                } else {
                    format!("{prefix} {pids_str}")
                }
            } else {
                self.wrap(DIM, "active")
            }
        } else {
            self.wrap(DIM, "idle")
        };

        let name = short_gpu_name(&row.name);
        if self.color {
            format!(
                "{DIM}  [{idx}] {name:<12}{RESET}  {bar}  {pct}  {status}",
                idx = row.idx,
                name = name,
            )
        } else {
            format!(
                "  [{idx}] {name:<12}  {bar}  {pct}  {status}",
                idx = row.idx,
                name = name,
            )
        }
    }

    fn step1_success(&self, indices: &[u32], tp: u32) -> String {
        format!(
            "{} isolated workload on gpus {indices:?} — tp={tp} inferred",
            self.wrap(GREEN, "✓")
        )
    }

    #[cfg(target_os = "linux")]
    fn step2_progress(&self, port: u16) -> String {
        self.wrap(
            DIM,
            &format!("vram ambiguous — tracing port {port} via /proc..."),
        )
    }

    #[cfg(target_os = "linux")]
    fn step2_port_line(
        &self,
        port: u16,
        pid: u32,
        indices: &[u32],
        highlight: bool,
        other_process: bool,
    ) -> String {
        let suffix = if other_process {
            if self.color {
                format!(" {DIM}(other process){RESET}")
            } else {
                " (other process)".to_string()
            }
        } else {
            String::new()
        };
        if self.color {
            if highlight {
                format!(
                    "{DIM}  port {port} →{RESET} {BLUE}pid {pid}{RESET} {DIM}→ gpus{RESET} {indices:?}{suffix}"
                )
            } else {
                format!(
                    "{DIM}  port {port} →{RESET} {DIM}pid {pid}{RESET}  {DIM}→ gpus {indices:?}{RESET}{suffix}"
                )
            }
        } else if highlight {
            format!("  port {port} → pid {pid} → gpus {indices:?}{suffix}")
        } else {
            format!("  port {port} → pid {pid}  → gpus {indices:?}{suffix}")
        }
    }

    #[cfg(target_os = "linux")]
    fn step2_success(&self, port: u16, indices: &[u32], tp: u32) -> String {
        format!(
            "{} port {port} maps to gpus {indices:?} — tp={tp} inferred",
            self.wrap(GREEN, "✓")
        )
    }

    fn step3_header(&self, pid_list: &str) -> String {
        if pid_list.is_empty() {
            format!(
                "{} ambiguous — PIDs not visible. Try: {}",
                self.wrap(RED, "[!]"),
                self.wrap(BLUE, "ps aux | grep vllm")
            )
        } else {
            format!(
                "{} ambiguous — run {} to trace your instance.",
                self.wrap(RED, "[!]"),
                self.wrap(BLUE, &format!("ps -f -p {pid_list}"))
            )
        }
    }

    fn prompt_tp(&self, host_gpus: u32) -> String {
        self.wrap(BLUE, &tensor_parallel_prompt(host_gpus))
    }

    fn prompt_indices(&self) -> String {
        self.wrap(BLUE, "--gpu-indices (e.g. 0,1): ")
    }

    fn step3_success(&self, indices: &[u32], tp: u32) -> String {
        format!(
            "{} locked to gpus {indices:?} — tp={tp}",
            self.wrap(GREEN, "✓")
        )
    }
}

pub(crate) fn tensor_parallel_prompt(host_gpus: u32) -> String {
    format!("--tensor-parallel-size ({host_gpus} gpus detected): ")
}

fn color_enabled() -> bool {
    if std::env::var_os("NO_COLOR").is_some() {
        return false;
    }
    io::stderr().is_terminal()
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

    if let Some(a) = vram_heuristic_from_fracs(host_count, &fracs, known_tp) {
        let active_set: HashSet<u32> = a.indices.iter().copied().collect();
        for row in &snapshots {
            let highlight = active_set.contains(&row.idx);
            eprintln!(
                "{}",
                term.format_gpu_row(row, RowColorMode::Step1, highlight)
            );
        }
        eprintln!("{}", term.step1_success(&a.indices, a.tp));
        return Ok(a);
    }

    for row in &snapshots {
        eprintln!(
            "{}",
            term.format_gpu_row(row, RowColorMode::Ambiguous, row.vram_pct >= 70.0)
        );
    }

    #[cfg(target_os = "linux")]
    if let Some(a) = proc_tiebreaker(&nvml, host_count, known_tp, url, &term, &snapshots) {
        return Ok(a);
    }

    eprintln!(
        "\n[i] Could not resolve GPU assignment automatically (port trace failed, ambiguous, or unsupported OS). Falling back to interactive prompt."
    );

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
        let vram_pct = device
            .as_ref()
            .and_then(|d| d.memory_info().ok())
            .map(|m| {
                if m.total == 0 {
                    0.0
                } else {
                    m.used as f64 / m.total as f64 * 100.0
                }
            })
            .unwrap_or(0.0);
        let pids = device
            .as_ref()
            .and_then(|d| d.running_compute_processes().ok())
            .map(|procs| procs.iter().map(|p| p.pid).collect())
            .unwrap_or_default();
        out.push(GpuSnapshot {
            idx,
            name,
            vram_pct,
            pids,
        });
    }
    out
}

pub(crate) fn vram_heuristic_from_fracs(
    host_count: u32,
    fracs: &[Option<f64>],
    known_tp: Option<u32>,
) -> Option<GpuAssignment> {
    let mut active = Vec::new();
    for idx in 0..host_count {
        let frac = fracs.get(idx as usize).copied().flatten()?;
        if frac >= VRAM_ACTIVE_THRESHOLD {
            active.push(idx);
        }
    }
    if active.is_empty() || active.len() >= host_count as usize {
        return None;
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

#[cfg(target_os = "linux")]
fn proc_tiebreaker(
    nvml: &Nvml,
    host_count: u32,
    known_tp: Option<u32>,
    url: &str,
    term: &Term,
    _snapshots: &[GpuSnapshot],
) -> Option<GpuAssignment> {
    let port = port_from_metrics_url(url)?;
    eprintln!("{}", term.step2_progress(port));

    let inode = find_inode_for_port(port)?;
    let vllm_pid = find_pid_for_socket_inode(inode)?;
    let matching = gpu_indices_for_vllm_pid(nvml, host_count, vllm_pid)?;
    if matching.is_empty() {
        return None;
    }
    if let Some(tp) = known_tp
        && matching.len() != tp as usize
    {
        return None;
    }

    eprintln!(
        "{}",
        term.step2_port_line(port, vllm_pid, &matching, true, false)
    );

    if let Some(other) = find_other_port_line(nvml, host_count, port) {
        eprintln!(
            "{}",
            term.step2_port_line(other.port, other.pid, &other.indices, false, true)
        );
    }

    let tp = matching.len() as u32;
    eprintln!("{}", term.step2_success(port, &matching, tp));
    Some(GpuAssignment {
        tp,
        indices: matching,
    })
}

#[cfg(target_os = "linux")]
struct PortGpuLine {
    port: u16,
    pid: u32,
    indices: Vec<u32>,
}

#[cfg(target_os = "linux")]
fn find_other_port_line(nvml: &Nvml, host_count: u32, skip_port: u16) -> Option<PortGpuLine> {
    let skip_needle = format!(":{:04X}", skip_port);
    let mut tcp_data = std::fs::read_to_string("/proc/net/tcp").unwrap_or_default();
    if let Ok(tcp6) = std::fs::read_to_string("/proc/net/tcp6") {
        tcp_data.push('\n');
        if let Some((_, rest)) = tcp6.split_once('\n') {
            tcp_data.push_str(rest);
        } else {
            tcp_data.push_str(&tcp6);
        }
    }
    if tcp_data.is_empty() {
        return None;
    }
    for line in tcp_data.lines() {
        let cols: Vec<&str> = line.split_whitespace().collect();
        if cols.len() < 10 {
            continue;
        }
        let local = cols[1];
        if local.ends_with(&skip_needle) {
            continue;
        }
        let Some(port_hex) = local.rsplit(':').next() else {
            continue;
        };
        let Some(port) = u16::from_str_radix(port_hex, 16).ok() else {
            continue;
        };
        if port == 0 {
            continue;
        }
        let Some(inode) = cols[9].parse().ok() else {
            continue;
        };
        let Some(pid) = find_pid_for_socket_inode(inode) else {
            continue;
        };
        let Some(indices) = gpu_indices_for_vllm_pid(nvml, host_count, pid) else {
            continue;
        };
        if indices.is_empty() {
            continue;
        }
        return Some(PortGpuLine { port, pid, indices });
    }
    None
}

#[cfg(target_os = "linux")]
fn gpu_indices_for_vllm_pid(nvml: &Nvml, host_count: u32, vllm_pid: u32) -> Option<Vec<u32>> {
    let mut by_pid: HashMap<u32, Vec<u32>> = HashMap::new();
    let mut by_ppid: HashMap<u32, Vec<u32>> = HashMap::new();

    for gpu_idx in 0..host_count {
        let Ok(device) = nvml.device_by_index(gpu_idx) else {
            continue;
        };
        let Ok(procs) = device.running_compute_processes() else {
            continue;
        };
        for proc in procs {
            let pid = proc.pid;
            by_pid.entry(pid).or_default().push(gpu_idx);
            if let Some(ppid) = read_ppid(pid) {
                by_ppid.entry(ppid).or_default().push(gpu_idx);
            }
        }
    }

    let mut matching = by_pid
        .get(&vllm_pid)
        .cloned()
        .or_else(|| by_ppid.get(&vllm_pid).cloned())?;
    matching.sort_unstable();
    matching.dedup();
    if matching.is_empty() {
        None
    } else {
        Some(matching)
    }
}

#[cfg(target_os = "linux")]
fn read_ppid(pid: u32) -> Option<u32> {
    let status = std::fs::read_to_string(format!("/proc/{pid}/status")).ok()?;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("PPid:") {
            return rest.trim().parse().ok();
        }
    }
    None
}

#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn port_from_metrics_url(url: &str) -> Option<u16> {
    let after_scheme = url.split("://").nth(1).unwrap_or(url);
    let host_port = after_scheme.split('/').next().unwrap_or(after_scheme);
    if let Some((_, port_str)) = host_port.rsplit_once(':') {
        port_str.parse().ok()
    } else {
        None
    }
}

#[cfg(target_os = "linux")]
fn find_inode_for_port(port: u16) -> Option<u64> {
    let needle = format!(":{:04X}", port);
    let mut tcp_data = std::fs::read_to_string("/proc/net/tcp").unwrap_or_default();
    if let Ok(tcp6) = std::fs::read_to_string("/proc/net/tcp6") {
        tcp_data.push('\n');
        if let Some((_, rest)) = tcp6.split_once('\n') {
            tcp_data.push_str(rest);
        } else {
            tcp_data.push_str(&tcp6);
        }
    }
    if tcp_data.is_empty() {
        return None;
    }
    for line in tcp_data.lines() {
        let cols: Vec<&str> = line.split_whitespace().collect();
        if cols.len() < 10 {
            continue;
        }
        if cols[1].ends_with(&needle) {
            return cols[9].parse().ok();
        }
    }
    None
}

#[cfg(target_os = "linux")]
fn find_pid_for_socket_inode(target_inode: u64) -> Option<u32> {
    let proc = std::fs::read_dir("/proc").ok()?;
    for entry in proc.flatten() {
        let name = entry.file_name();
        let pid: u32 = match name.to_string_lossy().parse() {
            Ok(p) => p,
            Err(_) => continue,
        };
        let fd_dir = format!("/proc/{pid}/fd");
        let fds = match std::fs::read_dir(&fd_dir) {
            Ok(d) => d,
            Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => continue,
            Err(_) => continue,
        };
        for fd in fds.flatten() {
            let Ok(target) = std::fs::read_link(fd.path()) else {
                continue;
            };
            let s = target.to_string_lossy();
            if let Some(inode) = s.strip_prefix("socket:[").and_then(|x| x.strip_suffix(']'))
                && inode.parse::<u64>().ok() == Some(target_inode)
            {
                return Some(pid);
            }
        }
    }
    None
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
    eprintln!("{}", term.step3_success(&indices, tp));
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
    fn vram_heuristic_all_active_returns_none() {
        let fracs = [Some(0.92), Some(0.91)];
        assert!(vram_heuristic_from_fracs(2, &fracs, None).is_none());
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
    fn port_from_metrics_url_parses() {
        assert_eq!(
            port_from_metrics_url("http://localhost:8000/metrics"),
            Some(8000)
        );
        assert_eq!(port_from_metrics_url("http://localhost:9000/"), Some(9000));
        assert_eq!(port_from_metrics_url("http://localhost/metrics"), None);
        assert_eq!(
            port_from_metrics_url("http://localhost:invalid/metrics"),
            None
        );
    }

    #[test]
    fn plain_text_when_no_color() {
        let term = Term { color: false };
        let row = GpuSnapshot {
            idx: 0,
            name: "NVIDIA A100".to_string(),
            vram_pct: 92.0,
            pids: vec![40592],
        };
        let line = term.format_gpu_row(&row, RowColorMode::Step1, true);
        assert!(!line.contains('\x1b'));
        assert!(line.contains("[0]"));
        assert!(line.contains("92% vram"));
    }
}
