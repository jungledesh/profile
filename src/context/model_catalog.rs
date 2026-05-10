/// Static model catalog for `ModelArch` field resolution.
///
/// Lookup is token-based: every token in `tokens` must appear as a substring
/// in the normalized model name (lowercase, non-alphanumeric → space).
/// Entries are ordered most-specific to least-specific within each family so
/// the first match wins.
///
/// param_count and active_param_count are in raw parameter counts (not billions).
/// e.g. 8B → 8_000_000_000u64.
pub struct CatalogEntry {
    pub family: &'static str,
    /// Total parameter count.
    pub param_count: u64,
    /// Active parameter count for MoE models; None for dense.
    pub active_param_count: Option<u64>,
    pub num_layers: u32,
    pub hidden_dim: u32,
    pub is_moe: bool,
}

struct ModelEntry {
    /// All tokens must appear as substrings in the normalized name.
    tokens: &'static [&'static str],
    entry: CatalogEntry,
}

const B: u64 = 1_000_000_000;

static CATALOG: &[ModelEntry] = &[
    // ── Llama 4 ──────────────────────────────────────────────────────────────
    // Maverick: 17B active / 400B total MoE
    ModelEntry {
        tokens: &["llama", "4", "maverick"],
        entry: CatalogEntry {
            family: "llama4",
            param_count: 400 * B,
            active_param_count: Some(17 * B),
            num_layers: 48,
            hidden_dim: 5120,
            is_moe: true,
        },
    },
    // Scout: 17B active / 109B total MoE
    ModelEntry {
        tokens: &["llama", "4", "scout"],
        entry: CatalogEntry {
            family: "llama4",
            param_count: 109 * B,
            active_param_count: Some(17 * B),
            num_layers: 48,
            hidden_dim: 5120,
            is_moe: true,
        },
    },
    // ── Nemotron (before generic llama entries — names contain "llama" + size) ─
    ModelEntry {
        tokens: &["nemotron", "70b"],
        entry: CatalogEntry {
            family: "nemotron",
            param_count: 70 * B,
            active_param_count: None,
            num_layers: 80,
            hidden_dim: 8192,
            is_moe: false,
        },
    },
    ModelEntry {
        tokens: &["nemotron", "8b"],
        entry: CatalogEntry {
            family: "nemotron",
            param_count: 8 * B,
            active_param_count: None,
            num_layers: 32,
            hidden_dim: 4096,
            is_moe: false,
        },
    },
    // ── Llama 3.x ────────────────────────────────────────────────────────────
    // "3" token required: "3.1" contains "3"; guards against a hypothetical
    // "Llama-4-70B" (no version tag) matching as llama3.
    ModelEntry {
        tokens: &["llama", "3", "405b"],
        entry: CatalogEntry {
            family: "llama3",
            param_count: 405 * B,
            active_param_count: None,
            num_layers: 126,
            hidden_dim: 16384,
            is_moe: false,
        },
    },
    ModelEntry {
        tokens: &["llama", "3", "70b"],
        entry: CatalogEntry {
            family: "llama3",
            param_count: 70 * B,
            active_param_count: None,
            num_layers: 80,
            hidden_dim: 8192,
            is_moe: false,
        },
    },
    ModelEntry {
        tokens: &["llama", "3", "8b"],
        entry: CatalogEntry {
            family: "llama3",
            param_count: 8 * B,
            active_param_count: None,
            num_layers: 32,
            hidden_dim: 4096,
            is_moe: false,
        },
    },
    // ── Qwen 3 MoE ───────────────────────────────────────────────────────────
    ModelEntry {
        tokens: &["qwen3", "235b"],
        entry: CatalogEntry {
            family: "qwen3",
            param_count: 235 * B,
            active_param_count: Some(22 * B),
            num_layers: 94,
            hidden_dim: 7168,
            is_moe: true,
        },
    },
    ModelEntry {
        tokens: &["qwen3", "30b"],
        entry: CatalogEntry {
            family: "qwen3",
            param_count: 30 * B,
            active_param_count: Some(3 * B),
            num_layers: 48,
            hidden_dim: 2048,
            is_moe: true,
        },
    },
    // ── Qwen 3 dense ─────────────────────────────────────────────────────────
    ModelEntry {
        tokens: &["qwen3", "72b"],
        entry: CatalogEntry {
            family: "qwen3",
            param_count: 72 * B,
            active_param_count: None,
            num_layers: 80,
            hidden_dim: 8192,
            is_moe: false,
        },
    },
    ModelEntry {
        tokens: &["qwen3", "32b"],
        entry: CatalogEntry {
            family: "qwen3",
            param_count: 32 * B,
            active_param_count: None,
            num_layers: 64,
            hidden_dim: 5120,
            is_moe: false,
        },
    },
    ModelEntry {
        tokens: &["qwen3", "14b"],
        entry: CatalogEntry {
            family: "qwen3",
            param_count: 14 * B,
            active_param_count: None,
            num_layers: 40,
            hidden_dim: 5120,
            is_moe: false,
        },
    },
    ModelEntry {
        tokens: &["qwen3", "7b"],
        entry: CatalogEntry {
            family: "qwen3",
            param_count: 7 * B,
            active_param_count: None,
            num_layers: 28,
            hidden_dim: 3584,
            is_moe: false,
        },
    },
    // ── Qwen 2.5 dense ───────────────────────────────────────────────────────
    ModelEntry {
        tokens: &["qwen2.5", "72b"],
        entry: CatalogEntry {
            family: "qwen2.5",
            param_count: 72 * B,
            active_param_count: None,
            num_layers: 80,
            hidden_dim: 8192,
            is_moe: false,
        },
    },
    ModelEntry {
        tokens: &["qwen2.5", "32b"],
        entry: CatalogEntry {
            family: "qwen2.5",
            param_count: 32 * B,
            active_param_count: None,
            num_layers: 64,
            hidden_dim: 5120,
            is_moe: false,
        },
    },
    ModelEntry {
        tokens: &["qwen2.5", "14b"],
        entry: CatalogEntry {
            family: "qwen2.5",
            param_count: 14 * B,
            active_param_count: None,
            num_layers: 48,
            hidden_dim: 5120,
            is_moe: false,
        },
    },
    ModelEntry {
        tokens: &["qwen2.5", "7b"],
        entry: CatalogEntry {
            family: "qwen2.5",
            param_count: 7 * B,
            active_param_count: None,
            num_layers: 28,
            hidden_dim: 3584,
            is_moe: false,
        },
    },
    // ── DeepSeek V3 / R1 ─────────────────────────────────────────────────────
    // 671B MoE (37B active)
    ModelEntry {
        tokens: &["deepseek", "671b"],
        entry: CatalogEntry {
            family: "deepseek",
            param_count: 671 * B,
            active_param_count: Some(37 * B),
            num_layers: 61,
            hidden_dim: 7168,
            is_moe: true,
        },
    },
    // R1 without size token defaults to 671B
    ModelEntry {
        tokens: &["deepseek", "r1"],
        entry: CatalogEntry {
            family: "deepseek",
            param_count: 671 * B,
            active_param_count: Some(37 * B),
            num_layers: 61,
            hidden_dim: 7168,
            is_moe: true,
        },
    },
    // V3 without size token defaults to 671B
    ModelEntry {
        tokens: &["deepseek", "v3"],
        entry: CatalogEntry {
            family: "deepseek",
            param_count: 671 * B,
            active_param_count: Some(37 * B),
            num_layers: 61,
            hidden_dim: 7168,
            is_moe: true,
        },
    },
    ModelEntry {
        tokens: &["deepseek", "70b"],
        entry: CatalogEntry {
            family: "deepseek",
            param_count: 70 * B,
            active_param_count: None,
            num_layers: 80,
            hidden_dim: 8192,
            is_moe: false,
        },
    },
    ModelEntry {
        tokens: &["deepseek", "7b"],
        entry: CatalogEntry {
            family: "deepseek",
            param_count: 7 * B,
            active_param_count: None,
            num_layers: 30,
            hidden_dim: 4096,
            is_moe: false,
        },
    },
    // ── Mistral Large 3 ──────────────────────────────────────────────────────
    // 675B MoE (52B active est.)
    ModelEntry {
        tokens: &["mistral", "large", "675b"],
        entry: CatalogEntry {
            family: "mistral",
            param_count: 675 * B,
            active_param_count: Some(52 * B),
            num_layers: 88,
            hidden_dim: 8192,
            is_moe: true,
        },
    },
    // 123B dense
    ModelEntry {
        tokens: &["mistral", "large", "123b"],
        entry: CatalogEntry {
            family: "mistral",
            param_count: 123 * B,
            active_param_count: None,
            num_layers: 88,
            hidden_dim: 8192,
            is_moe: false,
        },
    },
    // Mistral Large without explicit size — 123B default
    ModelEntry {
        tokens: &["mistral", "large"],
        entry: CatalogEntry {
            family: "mistral",
            param_count: 123 * B,
            active_param_count: None,
            num_layers: 88,
            hidden_dim: 8192,
            is_moe: false,
        },
    },
    // ── Mixtral MoE ──────────────────────────────────────────────────────────
    // 8x22B — 141B total, ~39B active
    ModelEntry {
        tokens: &["mixtral", "8x22b"],
        entry: CatalogEntry {
            family: "mistral",
            param_count: 141 * B,
            active_param_count: Some(39 * B),
            num_layers: 56,
            hidden_dim: 6144,
            is_moe: true,
        },
    },
    // 8x7B — 47B total, ~13B active
    ModelEntry {
        tokens: &["mixtral", "8x7b"],
        entry: CatalogEntry {
            family: "mistral",
            param_count: 47 * B,
            active_param_count: Some(13 * B),
            num_layers: 32,
            hidden_dim: 4096,
            is_moe: true,
        },
    },
    // ── Mistral 7B dense ─────────────────────────────────────────────────────
    ModelEntry {
        tokens: &["mistral", "7b"],
        entry: CatalogEntry {
            family: "mistral",
            param_count: 7 * B,
            active_param_count: None,
            num_layers: 32,
            hidden_dim: 4096,
            is_moe: false,
        },
    },
    // ── Gemma 2 / 3 ──────────────────────────────────────────────────────────
    ModelEntry {
        tokens: &["gemma", "27b"],
        entry: CatalogEntry {
            family: "gemma",
            param_count: 27 * B,
            active_param_count: None,
            num_layers: 46,
            hidden_dim: 4608,
            is_moe: false,
        },
    },
    ModelEntry {
        tokens: &["gemma", "9b"],
        entry: CatalogEntry {
            family: "gemma",
            param_count: 9 * B,
            active_param_count: None,
            num_layers: 42,
            hidden_dim: 3584,
            is_moe: false,
        },
    },
    // ── Kimi K2.5 ────────────────────────────────────────────────────────────
    // 1T total MoE, ~32B active
    ModelEntry {
        tokens: &["kimi", "k2"],
        entry: CatalogEntry {
            family: "kimi",
            param_count: 1_000 * B,
            active_param_count: Some(32 * B),
            num_layers: 96,
            hidden_dim: 7168,
            is_moe: true,
        },
    },
    // ── GLM-5 / 5.1 MoE ──────────────────────────────────────────────────────
    // 744B total MoE, ~56B active (est.)
    ModelEntry {
        tokens: &["glm", "744b"],
        entry: CatalogEntry {
            family: "glm",
            param_count: 744 * B,
            active_param_count: Some(56 * B),
            num_layers: 80,
            hidden_dim: 8192,
            is_moe: true,
        },
    },
    // 32B dense variant
    ModelEntry {
        tokens: &["glm", "32b"],
        entry: CatalogEntry {
            family: "glm",
            param_count: 32 * B,
            active_param_count: None,
            num_layers: 64,
            hidden_dim: 5120,
            is_moe: false,
        },
    },
    // ── Phi-4 ────────────────────────────────────────────────────────────────
    ModelEntry {
        tokens: &["phi", "4", "32b"],
        entry: CatalogEntry {
            family: "phi4",
            param_count: 32 * B,
            active_param_count: None,
            num_layers: 40,
            hidden_dim: 5120,
            is_moe: false,
        },
    },
    ModelEntry {
        tokens: &["phi", "4", "14b"],
        entry: CatalogEntry {
            family: "phi4",
            param_count: 14 * B,
            active_param_count: None,
            num_layers: 40,
            hidden_dim: 5120,
            is_moe: false,
        },
    },
    // Phi-4 without explicit size — 14B default
    ModelEntry {
        tokens: &["phi", "4"],
        entry: CatalogEntry {
            family: "phi4",
            param_count: 14 * B,
            active_param_count: None,
            num_layers: 40,
            hidden_dim: 5120,
            is_moe: false,
        },
    },
];

/// Lowercase the name; replace non-alphanumeric characters (except `.`) with spaces.
/// "Qwen2.5-72B-Instruct" → "qwen2.5 72b instruct". Dots are kept so that
/// version tokens like "qwen2.5" match literally.
fn normalize(name: &str) -> String {
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
/// the normalized model name, or `None` if no entry matches.
pub fn lookup_model(name: &str) -> Option<&'static CatalogEntry> {
    let norm = normalize(name);
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
    fn llama3_70b_variants() {
        for name in &[
            "meta-llama/Llama-3.1-70B-Instruct",
            "meta-llama/Meta-Llama-3-70B",
            "llama-3.3-70b-instruct",
        ] {
            let e = lookup_model(name).unwrap_or_else(|| panic!("no match for {name}"));
            assert_eq!(e.family, "llama3");
            assert_eq!(e.param_count, 70 * B);
            assert!(!e.is_moe);
        }
    }

    #[test]
    fn llama4_maverick() {
        let e = lookup_model("meta-llama/Llama-4-Maverick-17B-128E-Instruct")
            .expect("should match maverick");
        assert_eq!(e.family, "llama4");
        assert_eq!(e.active_param_count, Some(17 * B));
        assert!(e.is_moe);
    }

    #[test]
    fn llama4_scout() {
        let e =
            lookup_model("meta-llama/Llama-4-Scout-17B-16E-Instruct").expect("should match scout");
        assert_eq!(e.family, "llama4");
        assert!(e.is_moe);
    }

    #[test]
    fn qwen3_moe_235b() {
        let e = lookup_model("Qwen/Qwen3-235B-A22B").expect("should match qwen3 235b moe");
        assert_eq!(e.family, "qwen3");
        assert_eq!(e.param_count, 235 * B);
        assert_eq!(e.active_param_count, Some(22 * B));
        assert!(e.is_moe);
    }

    #[test]
    fn qwen25_72b() {
        let e = lookup_model("Qwen/Qwen2.5-72B-Instruct").expect("no match");
        assert_eq!(e.family, "qwen2.5");
        assert_eq!(e.param_count, 72 * B);
        assert!(!e.is_moe);
    }

    #[test]
    fn deepseek_r1_671b() {
        let e = lookup_model("deepseek-ai/DeepSeek-R1").expect("no match");
        assert_eq!(e.family, "deepseek");
        assert_eq!(e.param_count, 671 * B);
        assert!(e.is_moe);
    }

    #[test]
    fn deepseek_v3_671b() {
        let e = lookup_model("deepseek-ai/DeepSeek-V3").expect("no match");
        assert_eq!(e.family, "deepseek");
        assert_eq!(e.param_count, 671 * B);
        assert!(e.is_moe);
    }

    #[test]
    fn mixtral_8x7b() {
        let e = lookup_model("mistralai/Mixtral-8x7B-Instruct-v0.1").expect("no match");
        assert_eq!(e.family, "mistral");
        assert_eq!(e.param_count, 47 * B);
        assert!(e.is_moe);
    }

    #[test]
    fn mixtral_8x22b() {
        let e = lookup_model("mistralai/Mixtral-8x22B-Instruct-v0.1").expect("no match");
        assert_eq!(e.family, "mistral");
        assert_eq!(e.param_count, 141 * B);
        assert!(e.is_moe);
    }

    #[test]
    fn mistral_large_123b() {
        let e = lookup_model("mistralai/Mistral-Large-Instruct-2411").expect("no match");
        assert_eq!(e.family, "mistral");
        assert_eq!(e.param_count, 123 * B);
        assert!(!e.is_moe);
    }

    #[test]
    fn nemotron_70b() {
        let e = lookup_model("nvidia/Llama-3.1-Nemotron-70B-Instruct").expect("no match");
        assert_eq!(e.family, "nemotron");
        assert_eq!(e.param_count, 70 * B);
    }

    #[test]
    fn kimi_k2() {
        let e = lookup_model("moonshotai/Kimi-K2-Instruct").expect("no match");
        assert_eq!(e.family, "kimi");
        assert_eq!(e.param_count, 1_000 * B);
        assert!(e.is_moe);
    }

    #[test]
    fn phi4_14b() {
        let e = lookup_model("microsoft/phi-4").expect("no match");
        assert_eq!(e.family, "phi4");
        assert_eq!(e.param_count, 14 * B);
    }

    #[test]
    fn unknown_model_returns_none() {
        assert!(lookup_model("somevendor/mystery-model-99B").is_none());
    }

    #[test]
    fn llama_70b_does_not_match_llama4() {
        // "llama-3.1-70B" must NOT match the llama4 entries
        let e = lookup_model("meta-llama/Llama-3.1-70B-Instruct").expect("no match");
        assert_eq!(e.family, "llama3", "should be llama3, got {}", e.family);
    }

    #[test]
    fn llama4_untagged_70b_returns_none() {
        // A hypothetical "Llama-4-70B" (no Scout/Maverick) must NOT match as
        // llama3. Before the "3" token guard this would have returned llama3.
        assert!(lookup_model("meta-llama/Llama-4-70B-Instruct").is_none());
    }
}
