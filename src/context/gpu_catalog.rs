/// Static GPU catalog for `GPUModel` field resolution.
///
/// Lookup is token-based: every token in `tokens` must appear as a substring
/// in the normalized GPU name (lowercase, non-alphanumeric → space).
/// Entries are ordered most-specific to least-specific so the first match wins.
pub struct GpuCatalogEntry {
    pub arch: &'static str,
    /// BF16/FP16 Tensor Core TFLOPS. Used for prefill roofline ceiling.
    pub peak_flops_tc_tflops: f64,
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
            peak_flops_tc_tflops: 756.0,
            peak_bw_gbps: 2000.0,
        },
    },
    // SXM: "hbm3" is exclusive to the SXM variant; "sxm" also works.
    GpuEntry {
        tokens: &["h100", "hbm3"],
        entry: GpuCatalogEntry {
            arch: "hopper",
            peak_flops_tc_tflops: 989.0,
            peak_bw_gbps: 3350.0,
        },
    },
    GpuEntry {
        tokens: &["h100", "sxm"],
        entry: GpuCatalogEntry {
            arch: "hopper",
            peak_flops_tc_tflops: 989.0,
            peak_bw_gbps: 3350.0,
        },
    },
    // ── H200 ─────────────────────────────────────────────────────────────────
    // Same compute die as H100 SXM; HBM3e doubles the bandwidth.
    GpuEntry {
        tokens: &["h200"],
        entry: GpuCatalogEntry {
            arch: "hopper",
            peak_flops_tc_tflops: 989.0,
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
            peak_flops_tc_tflops: 312.0,
            peak_bw_gbps: 2039.0,
        },
    },
    GpuEntry {
        tokens: &["a100", "40gb"],
        entry: GpuCatalogEntry {
            arch: "ampere",
            peak_flops_tc_tflops: 312.0,
            peak_bw_gbps: 1555.0,
        },
    },
    // ── B200 SXM ─────────────────────────────────────────────────────────────
    // Blackwell deprioritised FP32 in favour of FP4/FP8/BF16 tensor throughput.
    GpuEntry {
        tokens: &["b200"],
        entry: GpuCatalogEntry {
            arch: "blackwell",
            peak_flops_tc_tflops: 2250.0, // BF16 Dense Tensor Core (unverified — no production B200 to test against)
            peak_bw_gbps: 8000.0,
        },
    },
    // ── RTX PRO 6000 Blackwell ────────────────────────────────────────────────
    // 96GB GDDR7; consumer/workstation Blackwell.
    // Before generic "blackwell" token entries to prevent false matches.
    GpuEntry {
        tokens: &["rtx", "pro", "6000", "blackwell"],
        entry: GpuCatalogEntry {
            arch: "blackwell",
            peak_flops_tc_tflops: 181.0,
            peak_bw_gbps: 960.0,
        },
    },
    // ── L40S ─────────────────────────────────────────────────────────────────
    GpuEntry {
        tokens: &["l40s"],
        entry: GpuCatalogEntry {
            arch: "ada",
            peak_flops_tc_tflops: 366.0,
            peak_bw_gbps: 864.0,
        },
    },
    // ── RTX 4090 ─────────────────────────────────────────────────────────────
    GpuEntry {
        tokens: &["rtx", "4090"],
        entry: GpuCatalogEntry {
            arch: "ada",
            peak_flops_tc_tflops: 165.0,
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
        assert_eq!(e.peak_flops_tc_tflops, 989.0);
    }

    #[test]
    fn a100_80gb() {
        let e = lookup_gpu("NVIDIA A100-SXM4-80GB").expect("no match");
        assert_eq!(e.arch, "ampere");
        assert_eq!(e.peak_bw_gbps, 2039.0);
        assert_eq!(e.peak_flops_tc_tflops, 312.0);
    }

    #[test]
    fn a100_40gb() {
        let e = lookup_gpu("NVIDIA A100-PCIE-40GB").expect("no match");
        assert_eq!(e.peak_bw_gbps, 1555.0);
        assert_eq!(e.peak_flops_tc_tflops, 312.0);
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
        assert_eq!(e.peak_flops_tc_tflops, 2250.0);
    }

    #[test]
    fn rtx_pro_6000_blackwell() {
        let e = lookup_gpu("NVIDIA RTX PRO 6000 Blackwell").expect("no match");
        assert_eq!(e.arch, "blackwell");
        assert_eq!(e.peak_bw_gbps, 960.0);
        assert_eq!(e.peak_flops_tc_tflops, 181.0);
    }

    #[test]
    fn l40s() {
        let e = lookup_gpu("NVIDIA L40S").expect("no match");
        assert_eq!(e.arch, "ada");
        assert_eq!(e.peak_flops_tc_tflops, 366.0);
    }

    #[test]
    fn rtx_4090() {
        let e = lookup_gpu("NVIDIA GeForce RTX 4090").expect("no match");
        assert_eq!(e.arch, "ada");
        assert_eq!(e.peak_flops_tc_tflops, 165.0);
    }

    #[test]
    fn unknown_gpu_returns_none() {
        assert!(lookup_gpu("NVIDIA Tesla V100").is_none());
    }

    #[test]
    fn gb200_matches_b200_entry_via_substring() {
        // GB200 is the superchip (B200 GPU + Grace CPU). Same physics specs as B200.
        let e = lookup_gpu("NVIDIA GB200 NVL72").expect("no match");
        assert_eq!(e.arch, "blackwell");
        assert_eq!(e.peak_bw_gbps, 8000.0);
    }
}
