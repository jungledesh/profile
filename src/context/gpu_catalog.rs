/// Static GPU catalog for `GPUModel` field resolution.
///
/// Lookup is token-based: every token in `tokens` must appear as a substring
/// in the normalized GPU name (lowercase, non-alphanumeric → space).
/// Entries are ordered most-specific to least-specific so the first match wins.
pub struct GpuCatalogEntry {
    /// Test-only discriminator to verify token matching order.
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
    // Dense BF16 Tensor Core = half the sparsity-marked datasheet figure.
    // Sources:
    //   https://www.nvidia.com/en-us/data-center/h100/
    //   https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/h100/PB-11773-001_v01.pdf
    // PCIe before generic H100 - "pcie" is the discriminating token.
    GpuEntry {
        tokens: &["h100", "pcie"],
        entry: GpuCatalogEntry {
            arch: "hopper",
            peak_flops_tc_tflops: 756.0, // dense BF16 (1513 sparse / 2)
            peak_bw_gbps: 2000.0,
        },
    },
    // NVL: 94 GB HBM3 PCIe with NVLink bridge. Must precede hbm3/sxm: "nvl" is
    // the discriminating token; NVML name is "NVIDIA H100 NVL".
    // Dense BF16 835.5 TFLOPS (1671 sparse / 2); BW 3.9 TB/s.
    GpuEntry {
        tokens: &["h100", "nvl"],
        entry: GpuCatalogEntry {
            arch: "hopper",
            peak_flops_tc_tflops: 835.5,
            peak_bw_gbps: 3900.0,
        },
    },
    // SXM: "hbm3" is exclusive to the SXM variant; "sxm" also works.
    GpuEntry {
        tokens: &["h100", "hbm3"],
        entry: GpuCatalogEntry {
            arch: "hopper",
            peak_flops_tc_tflops: 989.0, // dense BF16 (1979 sparse / 2)
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
    // Driver-reported names include VRAM size: "A100-SXM4-80GB", "A100-PCIE-40GB", etc.
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
            peak_flops_tc_tflops: 2250.0, // BF16 Dense Tensor Core (unverified - no production B200 to test against)
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
            peak_flops_tc_tflops: 250.0,
            peak_bw_gbps: 1792.0,
        },
    },
    // ── L40S ─────────────────────────────────────────────────────────────────
    GpuEntry {
        tokens: &["l40s"],
        entry: GpuCatalogEntry {
            arch: "ada",
            peak_flops_tc_tflops: 362.05,
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
    // ── RTX A6000 (Ampere) ───────────────────────────────────────────────────
    GpuEntry {
        tokens: &["rtx", "a6000"],
        entry: GpuCatalogEntry {
            arch: "ampere",
            peak_flops_tc_tflops: 77.4,
            peak_bw_gbps: 768.0,
        },
    },
    // ── RTX 3090 Ti ──────────────────────────────────────────────────────────
    GpuEntry {
        tokens: &["rtx", "3090", "ti"],
        entry: GpuCatalogEntry {
            arch: "ampere",
            peak_flops_tc_tflops: 80.0,
            peak_bw_gbps: 1008.0,
        },
    },
    // ── RTX 3090 ─────────────────────────────────────────────────────────────
    GpuEntry {
        tokens: &["rtx", "3090"],
        entry: GpuCatalogEntry {
            arch: "ampere",
            peak_flops_tc_tflops: 71.16,
            peak_bw_gbps: 936.0,
        },
    },
    // ── RTX 5090 ─────────────────────────────────────────────────────────────
    GpuEntry {
        tokens: &["rtx", "5090"],
        entry: GpuCatalogEntry {
            arch: "blackwell",
            peak_flops_tc_tflops: 209.5,
            peak_bw_gbps: 1792.0,
        },
    },
    // ── GB10 (DGX Spark) ─────────────────────────────────────────────────────
    // Grace-Blackwell Superchip: 128 GB unified LPDDR5X. BW is system-level (~273 GB/s);
    // decode ceiling is fundamentally different from HBM parts - treat as approximate.
    // NVML reports device name "NVIDIA GB10" - token "gb10" matches.
    // BF16 Dense TC: 212.9 TFLOPS measured (mma_bf16bf16f32 via mmapeak on real hardware).
    // The marketed "1000 TOPS" figure is FP4 2:4 sparse - not BF16 dense.
    GpuEntry {
        tokens: &["gb10"],
        entry: GpuCatalogEntry {
            arch: "blackwell",
            peak_flops_tc_tflops: 212.9, // BF16 Dense TC - measured on production hardware
            peak_bw_gbps: 273.0,         // LPDDR5X system BW - confirmed
        },
    },
    // ── A10G ─────────────────────────────────────────────────────────────────
    GpuEntry {
        tokens: &["a10g"],
        entry: GpuCatalogEntry {
            arch: "ampere",
            peak_flops_tc_tflops: 126.0,
            peak_bw_gbps: 600.0,
        },
    },
    // ── AMD Instinct MI355X (CDNA4, HBM3e) ─────────────────────────────────
    // Same die as MI350X; higher clock (2400 MHz) and TBP (1400W).
    GpuEntry {
        tokens: &["mi355x"],
        entry: GpuCatalogEntry {
            arch: "cdna4",
            peak_flops_tc_tflops: 2500.0,
            peak_bw_gbps: 8000.0,
        },
    },
    // ── AMD Instinct MI350X (CDNA4, HBM3e) ─────────────────────────────────
    // Current AMD datacenter flagship. Launched June 2025.
    GpuEntry {
        tokens: &["mi350x"],
        entry: GpuCatalogEntry {
            arch: "cdna4",
            peak_flops_tc_tflops: 2300.0,
            peak_bw_gbps: 8000.0,
        },
    },
    // ── AMD Instinct MI325X (CDNA3, HBM3e) ─────────────────────────────────
    // Same compute die as MI300X; HBM3e increases bandwidth.
    GpuEntry {
        tokens: &["mi325x"],
        entry: GpuCatalogEntry {
            arch: "cdna3",
            peak_flops_tc_tflops: 1307.4,
            peak_bw_gbps: 6000.0,
        },
    },
    // ── AMD Instinct MI300A (CDNA3 APU) ─────────────────────────────────────
    // 228 GPU CUs (vs 304 on MI300X). 128 GB HBM3 shared with CPU.
    GpuEntry {
        tokens: &["mi300a"],
        entry: GpuCatalogEntry {
            arch: "cdna3",
            peak_flops_tc_tflops: 980.6,
            // 5300 GB/s is total HBM3 bandwidth shared with CPU; GPU-available portion depends on workload.
            peak_bw_gbps: 5300.0,
        },
    },
    // ── AMD Instinct MI300X (CDNA3) ─────────────────────────────────────────
    GpuEntry {
        tokens: &["mi300x"],
        entry: GpuCatalogEntry {
            arch: "cdna3",
            peak_flops_tc_tflops: 1307.4,
            peak_bw_gbps: 5300.0,
        },
    },
    // ── AMD Instinct MI250X (CDNA2) ─────────────────────────────────────────
    // OAM has 2 GCDs; ROCm sees each as a separate device.
    // Values are per-GCD (half of full-OAM: 383 / 2 = 191.5, 3276.8 / 2 = 1638.4).
    GpuEntry {
        tokens: &["mi250x"],
        entry: GpuCatalogEntry {
            arch: "cdna2",
            peak_flops_tc_tflops: 191.5,
            peak_bw_gbps: 1638.4,
        },
    },
    // ── AMD Instinct MI250 (CDNA2) ──────────────────────────────────────────
    // Dual-GCD OAM like MI250X but 208 CUs (104 per GCD) vs 220.
    // Values are per-GCD (half of full-OAM: 362.1 / 2 = 181.0, 3200 / 2 = 1600).
    // Must be after MI250X: "mi250" is a substring of "mi250x".
    GpuEntry {
        tokens: &["mi250"],
        entry: GpuCatalogEntry {
            arch: "cdna2",
            peak_flops_tc_tflops: 181.0,
            peak_bw_gbps: 1600.0,
        },
    },
    // ── AMD Instinct MI210 (CDNA2, PCIe) ────────────────────────────────────
    // Single-GCD PCIe card. 104 CUs, 64GB HBM2e.
    GpuEntry {
        tokens: &["mi210"],
        entry: GpuCatalogEntry {
            arch: "cdna2",
            peak_flops_tc_tflops: 181.0,
            peak_bw_gbps: 1638.4,
        },
    },
    // ── AMD Radeon RX 9070 XT (RDNA4) ───────────────────────────────────────
    // 64 CUs, 128 AI Accelerators, 16GB GDDR6.
    // RDNA4 WMMA matrix ops. vLLM native RDNA4 kernel support in progress;
    // roofline ceiling may be optimistic until kernels land.
    GpuEntry {
        tokens: &["rx", "9070", "xt"],
        entry: GpuCatalogEntry {
            arch: "rdna4",
            peak_flops_tc_tflops: 194.6,
            peak_bw_gbps: 644.0,
        },
    },
    // ── AMD Radeon RX 9070 (RDNA4) ───────────────────────────────────────────
    // 56 CUs, 112 AI Accelerators, 16GB GDDR6.
    // Must be after RX 9070 XT: ["rx", "9070"] matches "RX 9070 XT" names.
    GpuEntry {
        tokens: &["rx", "9070"],
        entry: GpuCatalogEntry {
            arch: "rdna4",
            peak_flops_tc_tflops: 144.5,
            peak_bw_gbps: 644.0,
        },
    },
    // ── AMD Radeon PRO W7900 (RDNA3) ─────────────────────────────────────────
    // 96 CUs, 48GB GDDR6. Dense FP16 matrix TFLOPS / BW from AMD product page:
    // https://www.amd.com/en/products/graphics/workstations/radeon-pro/w7000-series/amd-radeon-pro-w7900.html
    // https://www.amd.com/content/dam/amd/en/documents/products/graphics/workstation/radeon-pro-w7900-datasheet.pdf
    // No cloud $/GPU-hr row in gpu_prices.json → cost fields stay None.
    GpuEntry {
        tokens: &["w7900"],
        entry: GpuCatalogEntry {
            arch: "rdna3",
            peak_flops_tc_tflops: 123.0, // peak FP16 matrix
            peak_bw_gbps: 864.0,
        },
    },
    // ── AMD Radeon PRO W7800 (RDNA3) ─────────────────────────────────────────
    // 70 CUs, 32GB GDDR6. Dense FP16 matrix / BW:
    // https://www.amd.com/en/products/graphics/workstations/radeon-pro/w7000-series/amd-radeon-pro-w7800.html
    GpuEntry {
        tokens: &["w7800"],
        entry: GpuCatalogEntry {
            arch: "rdna3",
            peak_flops_tc_tflops: 90.5,
            peak_bw_gbps: 576.0,
        },
    },
    // ── AMD Radeon RX 7900 XTX (RDNA3) ─────────────────────────────────────
    // RDNA3 WMMA (shader-unit matrix ops, not dedicated tensor cores).
    // Roofline ceiling may be optimistic for vLLM workloads.
    // XTX before XT: "xt" is a substring of "xtx", so XTX must match first.
    // Dense FP16 matrix 123 TFLOPS, BW 960 GB/s:
    // https://www.amd.com/en/products/graphics/desktops/radeon/7000-series/amd-radeon-rx-7900xtx.html
    GpuEntry {
        tokens: &["rx", "7900", "xtx"],
        entry: GpuCatalogEntry {
            arch: "rdna3",
            peak_flops_tc_tflops: 123.0,
            peak_bw_gbps: 960.0,
        },
    },
    // ── AMD Radeon RX 7900 GRE (RDNA3) ───────────────────────────────────────
    // 80 CUs, 16GB GDDR6. Budget 16GB option for local LLM inference.
    // https://www.amd.com/en/products/graphics/desktops/radeon/7000-series/amd-radeon-rx-7900gre.html
    GpuEntry {
        tokens: &["rx", "7900", "gre"],
        entry: GpuCatalogEntry {
            arch: "rdna3",
            peak_flops_tc_tflops: 92.0,
            peak_bw_gbps: 576.0,
        },
    },
    // ── AMD Radeon RX 7900 XT (RDNA3) ───────────────────────────────────────
    // Dense FP16 matrix ~103 TFLOPS, BW 800 GB/s:
    // https://www.amd.com/en/products/graphics/desktops/radeon/7000-series/amd-radeon-rx-7900xt.html
    GpuEntry {
        tokens: &["rx", "7900", "xt"],
        entry: GpuCatalogEntry {
            arch: "rdna3",
            peak_flops_tc_tflops: 103.0,
            peak_bw_gbps: 800.0,
        },
    },
    // ── AMD Radeon RX 7800 XT (RDNA3) ───────────────────────────────────────
    GpuEntry {
        tokens: &["rx", "7800", "xt"],
        entry: GpuCatalogEntry {
            arch: "rdna3",
            peak_flops_tc_tflops: 74.6,
            peak_bw_gbps: 624.0,
        },
    },
    // ── AMD Radeon RX 7700 XT (RDNA3) ────────────────────────────────────────
    // 54 CUs, 12GB GDDR6.
    GpuEntry {
        tokens: &["rx", "7700", "xt"],
        entry: GpuCatalogEntry {
            arch: "rdna3",
            peak_flops_tc_tflops: 70.3,
            peak_bw_gbps: 432.0,
        },
    },
    // ── AMD Radeon RX 7600 XT (RDNA3) ────────────────────────────────────────
    // 32 CUs, 16GB GDDR6. Cheapest 16GB RDNA3 option for local LLM inference.
    GpuEntry {
        tokens: &["rx", "7600", "xt"],
        entry: GpuCatalogEntry {
            arch: "rdna3",
            peak_flops_tc_tflops: 45.1,
            peak_bw_gbps: 288.0,
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
    #![allow(clippy::float_cmp)]
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
    fn h100_nvl() {
        // NVML name for the 94 GB HBM3 NVL SKU.
        let e = lookup_gpu("NVIDIA H100 NVL").expect("no match");
        assert_eq!(e.arch, "hopper");
        assert_eq!(e.peak_flops_tc_tflops, 835.5);
        assert_eq!(e.peak_bw_gbps, 3900.0);
    }

    #[test]
    fn h100_nvl_not_pcie() {
        let e = lookup_gpu("NVIDIA H100 NVL").expect("no match");
        assert_ne!(e.peak_bw_gbps, 2000.0);
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
        // "NVIDIA A100" with no size token - must not guess.
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
        assert_eq!(e.peak_bw_gbps, 1792.0);
        assert_eq!(e.peak_flops_tc_tflops, 250.0);
    }

    #[test]
    fn l40s() {
        let e = lookup_gpu("NVIDIA L40S").expect("no match");
        assert_eq!(e.arch, "ada");
        assert_eq!(e.peak_flops_tc_tflops, 362.05);
    }

    #[test]
    fn rtx_4090() {
        let e = lookup_gpu("NVIDIA GeForce RTX 4090").expect("no match");
        assert_eq!(e.arch, "ada");
        assert_eq!(e.peak_flops_tc_tflops, 165.0);
    }

    #[test]
    fn rtx_a6000() {
        let e = lookup_gpu("NVIDIA RTX A6000").expect("no match");
        assert_eq!(e.arch, "ampere");
        assert_eq!(e.peak_flops_tc_tflops, 77.4);
        assert_eq!(e.peak_bw_gbps, 768.0);
    }

    #[test]
    fn rtx_3090_ti() {
        let e = lookup_gpu("NVIDIA GeForce RTX 3090 Ti").expect("no match");
        assert_eq!(e.arch, "ampere");
        assert_eq!(e.peak_flops_tc_tflops, 80.0);
        assert_eq!(e.peak_bw_gbps, 1008.0);
    }

    #[test]
    fn rtx_3090() {
        let e = lookup_gpu("NVIDIA GeForce RTX 3090").expect("no match");
        assert_eq!(e.arch, "ampere");
        assert_eq!(e.peak_flops_tc_tflops, 71.16);
        assert_eq!(e.peak_bw_gbps, 936.0);
    }

    #[test]
    fn rtx_5090() {
        let e = lookup_gpu("NVIDIA GeForce RTX 5090").expect("no match");
        assert_eq!(e.arch, "blackwell");
        assert_eq!(e.peak_flops_tc_tflops, 209.5);
        assert_eq!(e.peak_bw_gbps, 1792.0);
    }

    #[test]
    fn a10g() {
        let e = lookup_gpu("NVIDIA A10G").expect("no match");
        assert_eq!(e.arch, "ampere");
        assert_eq!(e.peak_flops_tc_tflops, 126.0);
        assert_eq!(e.peak_bw_gbps, 600.0);
    }

    #[test]
    fn gb10_dgx_spark() {
        // NVML reports "NVIDIA GB10" on production DGX Spark hardware.
        let e = lookup_gpu("NVIDIA GB10").expect("no match");
        assert_eq!(e.arch, "blackwell");
        assert_eq!(e.peak_flops_tc_tflops, 212.9);
        assert_eq!(e.peak_bw_gbps, 273.0);
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

    #[test]
    fn mi300x() {
        let e = lookup_gpu("AMD Instinct MI300X").expect("no match");
        assert_eq!(e.arch, "cdna3");
        assert_eq!(e.peak_flops_tc_tflops, 1307.4);
        assert_eq!(e.peak_bw_gbps, 5300.0);
    }

    #[test]
    fn mi300x_with_variant_suffix() {
        // Some firmware reports "Aqua Vanjaram [Instinct MI300X VF]"
        let e = lookup_gpu("Aqua Vanjaram [Instinct MI300X VF]").expect("no match");
        assert_eq!(e.arch, "cdna3");
    }

    #[test]
    fn mi325x() {
        let e = lookup_gpu("AMD Instinct MI325X").expect("no match");
        assert_eq!(e.arch, "cdna3");
        assert_eq!(e.peak_bw_gbps, 6000.0);
    }

    #[test]
    fn mi300a() {
        let e = lookup_gpu("AMD Instinct MI300A").expect("no match");
        assert_eq!(e.arch, "cdna3");
        assert_eq!(e.peak_flops_tc_tflops, 980.6);
    }

    #[test]
    fn mi250x_per_gcd() {
        // Per-GCD values (half of full OAM).
        let e = lookup_gpu("AMD INSTINCT MI250X (MCM) OAM AC MBA MSFT").expect("no match");
        assert_eq!(e.arch, "cdna2");
        assert_eq!(e.peak_flops_tc_tflops, 191.5);
        assert_eq!(e.peak_bw_gbps, 1638.4);
    }

    #[test]
    fn rx_7900_xtx() {
        let e = lookup_gpu("AMD Radeon RX 7900 XTX").expect("no match");
        assert_eq!(e.arch, "rdna3");
        assert_eq!(e.peak_flops_tc_tflops, 123.0);
        assert_eq!(e.peak_bw_gbps, 960.0);
    }

    #[test]
    fn rx_7900_xt() {
        let e = lookup_gpu("AMD Radeon RX 7900 XT").expect("no match");
        assert_eq!(e.arch, "rdna3");
        assert_eq!(e.peak_flops_tc_tflops, 103.0);
        assert_eq!(e.peak_bw_gbps, 800.0);
    }

    #[test]
    fn rx_7900_xtx_not_matched_by_xt_entry() {
        // XTX must match the XTX entry, not the XT entry.
        let e = lookup_gpu("AMD Radeon RX 7900 XTX").expect("no match");
        assert_eq!(e.peak_bw_gbps, 960.0, "XTX matched XT entry");
    }

    #[test]
    fn rx_7800_xt() {
        let e = lookup_gpu("AMD Radeon RX 7800 XT").expect("no match");
        assert_eq!(e.arch, "rdna3");
        assert_eq!(e.peak_flops_tc_tflops, 74.6);
        assert_eq!(e.peak_bw_gbps, 624.0);
    }

    #[test]
    fn mi355x() {
        let e = lookup_gpu("AMD Instinct MI355X").expect("no match");
        assert_eq!(e.arch, "cdna4");
        assert_eq!(e.peak_flops_tc_tflops, 2500.0);
        assert_eq!(e.peak_bw_gbps, 8000.0);
    }

    #[test]
    fn mi350x() {
        let e = lookup_gpu("AMD Instinct MI350X").expect("no match");
        assert_eq!(e.arch, "cdna4");
        assert_eq!(e.peak_flops_tc_tflops, 2300.0);
        assert_eq!(e.peak_bw_gbps, 8000.0);
    }

    #[test]
    fn mi250_per_gcd() {
        // Per-GCD values (half of full OAM).
        // Must not match MI250X entry.
        let e = lookup_gpu("AMD Instinct MI250").expect("no match");
        assert_eq!(e.arch, "cdna2");
        assert_eq!(e.peak_flops_tc_tflops, 181.0);
        assert_eq!(e.peak_bw_gbps, 1600.0);
    }

    #[test]
    fn mi250x_not_matched_by_mi250_entry() {
        // MI250X must still match MI250X entry, not MI250.
        let e = lookup_gpu("AMD Instinct MI250X").expect("no match");
        assert_eq!(e.peak_flops_tc_tflops, 191.5);
    }

    #[test]
    fn mi210() {
        let e = lookup_gpu("AMD Instinct MI210").expect("no match");
        assert_eq!(e.arch, "cdna2");
        assert_eq!(e.peak_flops_tc_tflops, 181.0);
        assert_eq!(e.peak_bw_gbps, 1638.4);
    }

    #[test]
    fn radeon_pro_w7900() {
        let e = lookup_gpu("AMD Radeon PRO W7900").expect("no match");
        assert_eq!(e.arch, "rdna3");
        assert_eq!(e.peak_flops_tc_tflops, 123.0);
        assert_eq!(e.peak_bw_gbps, 864.0);
    }

    #[test]
    fn radeon_pro_w7900_ds() {
        // Dual-slot variant reports same GPU name.
        let e = lookup_gpu("AMD Radeon PRO W7900 DS").expect("no match");
        assert_eq!(e.peak_flops_tc_tflops, 123.0);
    }

    #[test]
    fn radeon_pro_w7800() {
        let e = lookup_gpu("AMD Radeon PRO W7800").expect("no match");
        assert_eq!(e.arch, "rdna3");
        assert_eq!(e.peak_flops_tc_tflops, 90.5);
        assert_eq!(e.peak_bw_gbps, 576.0);
    }

    #[test]
    fn rx_9070_xt() {
        let e = lookup_gpu("AMD Radeon RX 9070 XT").expect("no match");
        assert_eq!(e.arch, "rdna4");
        assert_eq!(e.peak_flops_tc_tflops, 194.6);
        assert_eq!(e.peak_bw_gbps, 644.0);
    }

    #[test]
    fn rx_9070() {
        let e = lookup_gpu("AMD Radeon RX 9070").expect("no match");
        assert_eq!(e.arch, "rdna4");
        assert_eq!(e.peak_flops_tc_tflops, 144.5);
        assert_eq!(e.peak_bw_gbps, 644.0);
    }

    #[test]
    fn rx_9070_xt_not_matched_by_9070_entry() {
        // 9070 XT must match XT entry, not generic 9070.
        let e = lookup_gpu("AMD Radeon RX 9070 XT").expect("no match");
        assert_eq!(e.peak_flops_tc_tflops, 194.6);
    }

    #[test]
    fn rx_7900_gre() {
        let e = lookup_gpu("AMD Radeon RX 7900 GRE").expect("no match");
        assert_eq!(e.arch, "rdna3");
        assert_eq!(e.peak_flops_tc_tflops, 92.0);
        assert_eq!(e.peak_bw_gbps, 576.0);
    }

    #[test]
    fn rx_7700_xt() {
        let e = lookup_gpu("AMD Radeon RX 7700 XT").expect("no match");
        assert_eq!(e.arch, "rdna3");
        assert_eq!(e.peak_flops_tc_tflops, 70.3);
        assert_eq!(e.peak_bw_gbps, 432.0);
    }

    #[test]
    fn rx_7600_xt() {
        let e = lookup_gpu("AMD Radeon RX 7600 XT").expect("no match");
        assert_eq!(e.arch, "rdna3");
        assert_eq!(e.peak_flops_tc_tflops, 45.1);
        assert_eq!(e.peak_bw_gbps, 288.0);
    }
}
