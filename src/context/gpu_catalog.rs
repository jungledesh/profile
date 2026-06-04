/// Static GPU catalog for `GPUModel` field resolution.
///
/// Lookup is token-based: every token in `tokens` must appear as a substring
/// in the normalized GPU name (lowercase, non-alphanumeric → space).
/// Entries are ordered most-specific to least-specific so the first match wins.
///
/// `peak_flops_f32_tflops` is non-tensor-core FP32 throughput — the conservative
/// roofline input. LLMs execute in BF16/FP16 so the real compute ceiling is
/// higher; the output always labels this as an estimate.
pub struct GpuCatalogEntry {
    pub arch: &'static str,
    /// Non-tensor-core FP32 TFLOPS. Used for prefill roofline ceiling.
    pub peak_flops_f32_tflops: f64,
    /// Peak memory bandwidth GB/s. Used for decode roofline ceiling.
    pub peak_bw_gbps: f64,
}

struct GpuEntry {
    tokens: &'static [&'static str],
    entry: GpuCatalogEntry,
}

static CATALOG: &[GpuEntry] = &[
    // ── H100 ─────────────────────────────────────────────────────────────────
    // PCIe before generic H100 — "pcie" is the discriminating token.
    GpuEntry {
        tokens: &["h100", "pcie"],
        entry: GpuCatalogEntry {
            arch: "hopper",
            peak_flops_f32_tflops: 60.0,
            peak_bw_gbps: 2000.0,
        },
    },
    // SXM: "hbm3" is exclusive to the SXM variant; "sxm" also works.
    GpuEntry {
        tokens: &["h100", "hbm3"],
        entry: GpuCatalogEntry {
            arch: "hopper",
            peak_flops_f32_tflops: 67.0,
            peak_bw_gbps: 3350.0,
        },
    },
    GpuEntry {
        tokens: &["h100", "sxm"],
        entry: GpuCatalogEntry {
            arch: "hopper",
            peak_flops_f32_tflops: 67.0,
            peak_bw_gbps: 3350.0,
        },
    },
    // ── H200 ─────────────────────────────────────────────────────────────────
    // Same compute die as H100 SXM; HBM3e doubles the bandwidth.
    GpuEntry {
        tokens: &["h200"],
        entry: GpuCatalogEntry {
            arch: "hopper",
            peak_flops_f32_tflops: 67.0,
            peak_bw_gbps: 4800.0,
        },
    },
    // ── A100 ─────────────────────────────────────────────────────────────────
    // NVML names include VRAM size: "A100-SXM4-80GB", "A100-PCIE-40GB", etc.
    // Require explicit size token; ambiguous "a100" alone returns None.
    GpuEntry {
        tokens: &["a100", "80gb"],
        entry: GpuCatalogEntry {
            arch: "ampere",
            peak_flops_f32_tflops: 19.5,
            peak_bw_gbps: 2039.0,
        },
    },
    GpuEntry {
        tokens: &["a100", "40gb"],
        entry: GpuCatalogEntry {
            arch: "ampere",
            peak_flops_f32_tflops: 19.5,
            peak_bw_gbps: 1555.0,
        },
    },
    // ── B200 SXM ─────────────────────────────────────────────────────────────
    // Blackwell deprioritised FP32 in favour of FP4/FP8/BF16 tensor throughput.
    // FP32 ~20 TFLOPS; BW 8000 GB/s HBM3e (estimated — not yet widely published).
    GpuEntry {
        tokens: &["b200"],
        entry: GpuCatalogEntry {
            arch: "blackwell",
            peak_flops_f32_tflops: 20.0,
            peak_bw_gbps: 8000.0,
        },
    },
    // ── DGX Spark (GB10 Grace Blackwell Superchip) ───────────────────────────
    // LPDDR5X unified memory shared with Grace CPU — bandwidth is ~273 GB/s,
    // far below HBM. This significantly lowers the decode ceiling vs datacenter
    // cards. nvmlDeviceGetName expected to contain "gb10".
    // FP32 ~20 TFLOPS (estimated); BW estimated from published LPDDR5X specs.
    GpuEntry {
        tokens: &["gb10"],
        entry: GpuCatalogEntry {
            arch: "blackwell",
            peak_flops_f32_tflops: 20.0,
            peak_bw_gbps: 273.0,
        },
    },
    // ── RTX PRO 6000 Blackwell ────────────────────────────────────────────────
    // 96GB GDDR7; consumer/workstation Blackwell with high FP32 CUDA core count.
    // Before generic "blackwell" token entries to prevent false matches.
    GpuEntry {
        tokens: &["rtx", "pro", "6000", "blackwell"],
        entry: GpuCatalogEntry {
            arch: "blackwell",
            peak_flops_f32_tflops: 125.0,
            peak_bw_gbps: 960.0,
        },
    },
    // ── L40S ─────────────────────────────────────────────────────────────────
    GpuEntry {
        tokens: &["l40s"],
        entry: GpuCatalogEntry {
            arch: "ada",
            peak_flops_f32_tflops: 91.6,
            peak_bw_gbps: 864.0,
        },
    },
    // ── RTX 4090 ─────────────────────────────────────────────────────────────
    GpuEntry {
        tokens: &["rtx", "4090"],
        entry: GpuCatalogEntry {
            arch: "ada",
            peak_flops_f32_tflops: 82.6,
            peak_bw_gbps: 1008.0,
        },
    },
];

/// Lowercase the name; replace non-alphanumeric characters (except `.`) with spaces.
pub(crate) fn normalize_gpu_name(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '.' {
                c.to_ascii_lowercase()
            } else {
                ' '
            }
        })
        .collect()
}

/// Return the first catalog entry whose tokens all appear as substrings in
/// the normalized GPU name, or `None` if no entry matches.
pub fn lookup_gpu(name: &str) -> Option<&'static GpuCatalogEntry> {
    let norm = normalize_gpu_name(name);
    CATALOG.iter().find_map(|e| {
        if e.tokens.iter().all(|t| norm.contains(*t)) {
            Some(&e.entry)
        } else {
            None
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn h100_sxm_hbm3() {
        let e = lookup_gpu("NVIDIA H100 80GB HBM3").expect("no match");
        assert_eq!(e.arch, "hopper");
        assert_eq!(e.peak_bw_gbps, 3350.0);
    }

    #[test]
    fn h100_sxm_explicit() {
        let e = lookup_gpu("NVIDIA H100-SXM5-80GB").expect("no match");
        assert_eq!(e.arch, "hopper");
        assert_eq!(e.peak_bw_gbps, 3350.0);
    }

    #[test]
    fn h100_pcie() {
        let e = lookup_gpu("NVIDIA H100 PCIe").expect("no match");
        assert_eq!(e.peak_bw_gbps, 2000.0);
    }

    #[test]
    fn h100_pcie_not_sxm() {
        // PCIe variant must not match the SXM entry (3350 GB/s).
        let e = lookup_gpu("NVIDIA H100 PCIe").expect("no match");
        assert!(
            e.peak_bw_gbps < 3000.0,
            "PCIe matched SXM bandwidth {}",
            e.peak_bw_gbps
        );
    }

    #[test]
    fn h200() {
        let e = lookup_gpu("NVIDIA H200 SXM").expect("no match");
        assert_eq!(e.peak_bw_gbps, 4800.0);
        assert_eq!(e.peak_flops_f32_tflops, 67.0);
    }

    #[test]
    fn a100_80gb() {
        let e = lookup_gpu("NVIDIA A100-SXM4-80GB").expect("no match");
        assert_eq!(e.arch, "ampere");
        assert_eq!(e.peak_bw_gbps, 2039.0);
    }

    #[test]
    fn a100_40gb() {
        let e = lookup_gpu("NVIDIA A100-PCIE-40GB").expect("no match");
        assert_eq!(e.peak_bw_gbps, 1555.0);
    }

    #[test]
    fn a100_ambiguous_returns_none() {
        // "NVIDIA A100" with no size token — must not guess.
        assert!(lookup_gpu("NVIDIA A100").is_none());
    }

    #[test]
    fn b200() {
        let e = lookup_gpu("NVIDIA B200 SXM").expect("no match");
        assert_eq!(e.arch, "blackwell");
        assert_eq!(e.peak_bw_gbps, 8000.0);
    }

    #[test]
    fn dgx_spark_gb10() {
        let e = lookup_gpu("NVIDIA GB10").expect("no match");
        assert_eq!(e.arch, "blackwell");
        assert_eq!(e.peak_bw_gbps, 273.0);
    }

    #[test]
    fn rtx_pro_6000_blackwell() {
        let e = lookup_gpu("NVIDIA RTX PRO 6000 Blackwell").expect("no match");
        assert_eq!(e.arch, "blackwell");
        assert_eq!(e.peak_bw_gbps, 960.0);
        assert_eq!(e.peak_flops_f32_tflops, 125.0);
    }

    #[test]
    fn l40s() {
        let e = lookup_gpu("NVIDIA L40S").expect("no match");
        assert_eq!(e.arch, "ada");
        assert_eq!(e.peak_flops_f32_tflops, 91.6);
    }

    #[test]
    fn rtx_4090() {
        let e = lookup_gpu("NVIDIA GeForce RTX 4090").expect("no match");
        assert_eq!(e.arch, "ada");
        assert_eq!(e.peak_flops_f32_tflops, 82.6);
    }

    #[test]
    fn unknown_gpu_returns_none() {
        assert!(lookup_gpu("NVIDIA Tesla V100").is_none());
    }
}
