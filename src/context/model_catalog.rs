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
    /// Test-only discriminator to verify token matching order.
    pub family: &'static str,
    /// Total parameter count.
    pub param_count: u64,
    /// Active parameter count for MoE models; None for dense.
    pub active_param_count: Option<u64>,
    pub num_layers: u32,
    pub hidden_dim: u32,
    pub default_weight_dtype: &'static str,
    /// Number of KV heads (num_key_value_heads from config.json).
    /// None for MLA (DeepSeek V3/R1), interleaved attention (Llama 4), or any
    /// architecture where per-head KV semantics don't apply.
    /// Hybrid models (Qwen3.6) set this field. Use num_kv_layers to restrict to
    /// attention-only layers.
    pub num_kv_heads: Option<u32>,
    /// KV cache head dimension in elements. Separate from hidden_dim/num_heads
    /// (e.g. Gemma 2 9B uses head_dim=256 despite hidden_dim=3584).
    /// None when architecture is non-standard or unknown.
    pub head_dim: Option<u32>,
    /// KV-relevant layer count. For hybrid architectures (Qwen3.6 DeltaNet, future
    /// interleaved models) where only a subset of layers use KV cache.
    /// None → fall back to num_layers in KV math (correct for pure-attention models).
    pub num_kv_layers: Option<u32>,
    /// Pre-computed per-layer attention FLOPs coefficient for the quadratic seq_len² term.
    /// Total attention FLOPs = attn_flops_coeff × num_layers × seq_len².
    /// None → standard MHA/GQA: use 2 × hidden_dim.
    /// Some(0) → architecture has no quadratic attention term (skip correction).
    pub attn_flops_coeff: Option<u64>,
    /// Hybrid (linear-attention/mamba-class) state facts, from config.json verbatim.
    /// Used to derive fixed per-sequence state bytes. None => pure-attention model.
    pub linear_num_layers: Option<u32>,
    pub linear_key_heads: Option<u32>,
    pub linear_value_heads: Option<u32>,
    pub linear_key_head_dim: Option<u32>,
    pub linear_value_head_dim: Option<u32>,
    pub linear_conv_kernel_dim: Option<u32>,
    pub state_dtype: Option<&'static str>,
}

/// Pure-attention catalog row. Named fields are transposition-proof; hybrid/linear
/// state fields default to `None` inside the expansion.
macro_rules! catalog_dense {
    (
        family: $family:expr,
        param_count: $param_count:expr,
        active_param_count: $active_param_count:expr,
        num_layers: $num_layers:expr,
        hidden_dim: $hidden_dim:expr,
        default_weight_dtype: $default_weight_dtype:expr,
        num_kv_heads: $num_kv_heads:expr,
        head_dim: $head_dim:expr,
        num_kv_layers: $num_kv_layers:expr,
        attn_flops_coeff: $attn_flops_coeff:expr $(,)?
    ) => {
        CatalogEntry {
            family: $family,
            param_count: $param_count,
            active_param_count: $active_param_count,
            num_layers: $num_layers,
            hidden_dim: $hidden_dim,
            default_weight_dtype: $default_weight_dtype,
            num_kv_heads: $num_kv_heads,
            head_dim: $head_dim,
            num_kv_layers: $num_kv_layers,
            attn_flops_coeff: $attn_flops_coeff,
            linear_num_layers: None,
            linear_key_heads: None,
            linear_value_heads: None,
            linear_key_head_dim: None,
            linear_value_head_dim: None,
            linear_conv_kernel_dim: None,
            state_dtype: None,
        }
    };
}

/// Hybrid catalog row (linear_* / state_dtype set by name).
macro_rules! catalog_hybrid {
    (
        family: $family:expr,
        param_count: $param_count:expr,
        active_param_count: $active_param_count:expr,
        num_layers: $num_layers:expr,
        hidden_dim: $hidden_dim:expr,
        default_weight_dtype: $default_weight_dtype:expr,
        num_kv_heads: $num_kv_heads:expr,
        head_dim: $head_dim:expr,
        num_kv_layers: $num_kv_layers:expr,
        attn_flops_coeff: $attn_flops_coeff:expr,
        linear_num_layers: $linear_num_layers:expr,
        linear_key_heads: $linear_key_heads:expr,
        linear_value_heads: $linear_value_heads:expr,
        linear_key_head_dim: $linear_key_head_dim:expr,
        linear_value_head_dim: $linear_value_head_dim:expr,
        linear_conv_kernel_dim: $linear_conv_kernel_dim:expr,
        state_dtype: $state_dtype:expr $(,)?
    ) => {
        CatalogEntry {
            family: $family,
            param_count: $param_count,
            active_param_count: $active_param_count,
            num_layers: $num_layers,
            hidden_dim: $hidden_dim,
            default_weight_dtype: $default_weight_dtype,
            num_kv_heads: $num_kv_heads,
            head_dim: $head_dim,
            num_kv_layers: $num_kv_layers,
            attn_flops_coeff: $attn_flops_coeff,
            linear_num_layers: $linear_num_layers,
            linear_key_heads: $linear_key_heads,
            linear_value_heads: $linear_value_heads,
            linear_key_head_dim: $linear_key_head_dim,
            linear_value_head_dim: $linear_value_head_dim,
            linear_conv_kernel_dim: $linear_conv_kernel_dim,
            state_dtype: $state_dtype,
        }
    };
}

struct ModelEntry {
    /// All tokens must appear as substrings in the normalized name.
    tokens: &'static [&'static str],
    entry: CatalogEntry,
}

const B: u64 = 1_000_000_000;

static CATALOG: &[ModelEntry] = &[
    // ── Llama 4 ──────────────────────────────────────────────────────────────
    // Interleaved attention architecture: standard KV formula doesn't apply.
    // Maverick: 17B active / 400B total MoE (model card; not confirmed from config.json here).
    // Source: https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct/raw/main/config.json (fetch timed out — gated; unverified)
    // config.json gated from this environment; layer/hidden dims unverified — do not trust for physics.
    ModelEntry {
        tokens: &["llama", "4", "maverick"],
        entry: catalog_dense! {
            family: "llama4",
            param_count: 400 * B,
            active_param_count: Some(17 * B),
            num_layers: 48,
            hidden_dim: 5120,
            default_weight_dtype: "bf16",
            num_kv_heads: None,
            head_dim: None,
            num_kv_layers: None,
            attn_flops_coeff: Some(0),
        },
    },
    // Scout: 17B active / 109B total MoE (model card; not confirmed from config.json here).
    // Source: https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct/raw/main/config.json (fetch timed out — gated; unverified)
    // config.json gated from this environment; layer/hidden dims unverified — do not trust for physics.
    ModelEntry {
        tokens: &["llama", "4", "scout"],
        entry: catalog_dense! {
            family: "llama4",
            param_count: 109 * B,
            active_param_count: Some(17 * B),
            num_layers: 48,
            hidden_dim: 5120,
            default_weight_dtype: "bf16",
            num_kv_heads: None,
            head_dim: None,
            num_kv_layers: None,
            attn_flops_coeff: Some(0),
        },
    },
    // ── Nemotron (before generic llama entries, names contain "llama" + size) ─
    // Source: https://huggingface.co/nvidia/Llama-3.1-Nemotron-70B-Instruct/raw/main/config.json (accessed 2026-07-16)
    ModelEntry {
        tokens: &["nemotron", "70b"],
        entry: catalog_dense! {
            family: "nemotron",
            param_count: 70 * B,
            active_param_count: None,
            num_layers: 80,
            hidden_dim: 8192,
            default_weight_dtype: "bf16",
            num_kv_heads: Some(8),
            head_dim: Some(128),
            num_kv_layers: None,
            attn_flops_coeff: None,
        },
    },
    // Source: https://huggingface.co/nvidia/Llama-3.1-Nemotron-8B-Instruct/raw/main/config.json (accessed 2026-07-16)
    ModelEntry {
        tokens: &["nemotron", "8b"],
        entry: catalog_dense! {
            family: "nemotron",
            param_count: 8 * B,
            active_param_count: None,
            num_layers: 32,
            hidden_dim: 4096,
            default_weight_dtype: "bf16",
            num_kv_heads: Some(8),
            head_dim: Some(128),
            num_kv_layers: None,
            attn_flops_coeff: None,
        },
    },
    // ── Llama 3.x ────────────────────────────────────────────────────────────
    // "3" token required: "3.1" contains "3"; guards against a hypothetical
    // "Llama-4-70B" (no version tag) matching as llama3.
    // Source: https://huggingface.co/meta-llama/Meta-Llama-3.1-405B-Instruct/raw/main/config.json (accessed 2026-07-16)
    ModelEntry {
        tokens: &["llama", "3", "405b"],
        entry: catalog_dense! {
            family: "llama3",
            param_count: 405 * B,
            active_param_count: None,
            num_layers: 126,
            hidden_dim: 16384,
            default_weight_dtype: "bf16",
            num_kv_heads: Some(8),
            head_dim: Some(128),
            num_kv_layers: None,
            attn_flops_coeff: None,
        },
    },
    // Source: https://huggingface.co/meta-llama/Meta-Llama-3.1-70B-Instruct/raw/main/config.json (accessed 2026-07-16)
    ModelEntry {
        tokens: &["llama", "3", "70b"],
        entry: catalog_dense! {
            family: "llama3",
            param_count: 70 * B,
            active_param_count: None,
            num_layers: 80,
            hidden_dim: 8192,
            default_weight_dtype: "bf16",
            num_kv_heads: Some(8),
            head_dim: Some(128),
            num_kv_layers: None,
            attn_flops_coeff: None,
        },
    },
    // Source: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct/raw/main/config.json (accessed 2026-07-16)
    ModelEntry {
        tokens: &["llama", "3", "8b"],
        entry: catalog_dense! {
            family: "llama3",
            param_count: 8 * B,
            active_param_count: None,
            num_layers: 32,
            hidden_dim: 4096,
            default_weight_dtype: "bf16",
            num_kv_heads: Some(8),
            head_dim: Some(128),
            num_kv_layers: None,
            attn_flops_coeff: None,
        },
    },
    // ── Qwen 3 MoE ───────────────────────────────────────────────────────────
    // Source: https://huggingface.co/Qwen/Qwen3-235B-A22B/raw/main/config.json (accessed 2026-07-16)
    ModelEntry {
        tokens: &["qwen3", "235b"],
        entry: catalog_dense! {
            family: "qwen3",
            param_count: 235 * B,
            active_param_count: Some(22 * B),
            num_layers: 94,
            hidden_dim: 7168,
            default_weight_dtype: "bf16",
            num_kv_heads: Some(4),
            head_dim: Some(128),
            num_kv_layers: None,
            attn_flops_coeff: None,
        },
    },
    // Qwen3-30B-A3B MoE.
    // num_key_value_heads=4, head_dim=128, hidden_size=2048, num_hidden_layers=48.
    // Source: https://huggingface.co/Qwen/Qwen3-30B-A3B/raw/main/config.json (accessed 2026-07-16)
    ModelEntry {
        tokens: &["qwen3", "30b"],
        entry: catalog_dense! {
            family: "qwen3",
            param_count: 30 * B,
            active_param_count: Some(3 * B),
            num_layers: 48,
            hidden_dim: 2048,
            default_weight_dtype: "bf16",
            num_kv_heads: Some(4),
            head_dim: Some(128),
            num_kv_layers: None,
            attn_flops_coeff: None,
        },
    },
    // ── Qwen 3.6 dense / hybrid ──────────────────────────────────────────────
    // (Qwen3.6 35B entry removed: no official release)
    // Before generic qwen3 size entries, "7b" is a substring of "27b".
    // Released April 2026. Dense 27B; hybrid gated-DeltaNet + attention blocks.
    // Only attention layers use KV cache. Standard num_layers formula overstates.
    // Source: https://huggingface.co/Qwen/Qwen3.6-27B/raw/main/config.json (accessed 2026-07-16)
    ModelEntry {
        tokens: &["qwen3.6", "27b"],
        entry: catalog_hybrid! {
            family: "qwen3.6",
            param_count: 27 * B,
            active_param_count: None,
            num_layers: 64,
            hidden_dim: 5120,
            default_weight_dtype: "bf16",
            num_kv_heads: Some(4),
            head_dim: Some(256),
            // 64 total layers, 3:1 DeltaNet/attention interleave (full_attention_interval: 4)
            num_kv_layers: Some(16),
            attn_flops_coeff: None,
            linear_num_layers: Some(48),
            linear_key_heads: Some(16),
            linear_value_heads: Some(48),
            linear_key_head_dim: Some(128),
            linear_value_head_dim: Some(128),
            linear_conv_kernel_dim: Some(4),
            state_dtype: Some("fp32"),
        },
    },
    // ── Qwen 3 dense ─────────────────────────────────────────────────────────
    // Official dense SKUs: 0.6B, 1.7B, 4B, 8B, 14B, 32B (no dense 7B/72B).
    // Larger sizes first; first-match wins.
    // Source: https://huggingface.co/Qwen/Qwen3-32B/raw/main/config.json (accessed 2026-07-16)
    ModelEntry {
        tokens: &["qwen3", "32b"],
        entry: catalog_dense! {
            family: "qwen3",
            param_count: 32 * B,
            active_param_count: None,
            num_layers: 64,
            hidden_dim: 5120,
            default_weight_dtype: "bf16",
            num_kv_heads: Some(8),
            head_dim: Some(128),
            num_kv_layers: None,
            attn_flops_coeff: None,
        },
    },
    // Source: https://huggingface.co/Qwen/Qwen3-14B/raw/main/config.json (accessed 2026-07-16)
    ModelEntry {
        tokens: &["qwen3", "14b"],
        entry: catalog_dense! {
            family: "qwen3",
            param_count: 14 * B,
            active_param_count: None,
            num_layers: 40,
            hidden_dim: 5120,
            default_weight_dtype: "bf16",
            num_kv_heads: Some(8),
            head_dim: Some(128),
            num_kv_layers: None,
            attn_flops_coeff: None,
        },
    },
    // Source: https://huggingface.co/Qwen/Qwen3-8B/raw/main/config.json (accessed 2026-07-16)
    ModelEntry {
        // https://huggingface.co/Qwen/Qwen3-8B/raw/main/config.json
        tokens: &["qwen3", "8b"],
        entry: catalog_dense! {
            family: "qwen3",
            param_count: 8 * B,
            active_param_count: None,
            num_layers: 36,
            hidden_dim: 4096,
            default_weight_dtype: "bf16",
            num_kv_heads: Some(8),
            head_dim: Some(128),
            num_kv_layers: None,
            attn_flops_coeff: None,
        },
    },
    // Source: https://huggingface.co/Qwen/Qwen3-4B/raw/main/config.json (accessed 2026-07-16)
    ModelEntry {
        // https://huggingface.co/Qwen/Qwen3-4B/raw/main/config.json
        tokens: &["qwen3", "4b"],
        entry: catalog_dense! {
            family: "qwen3",
            param_count: 4 * B,
            active_param_count: None,
            num_layers: 36,
            hidden_dim: 2560,
            default_weight_dtype: "bf16",
            num_kv_heads: Some(8),
            head_dim: Some(128),
            num_kv_layers: None,
            attn_flops_coeff: None,
        },
    },
    // Source: https://huggingface.co/Qwen/Qwen3-1.7B/raw/main/config.json (accessed 2026-07-16)
    ModelEntry {
        // https://huggingface.co/Qwen/Qwen3-1.7B/raw/main/config.json
        tokens: &["qwen3", "1.7b"],
        entry: catalog_dense! {
            family: "qwen3",
            param_count: 1_700_000_000,
            active_param_count: None,
            num_layers: 28,
            hidden_dim: 2048,
            default_weight_dtype: "bf16",
            num_kv_heads: Some(8),
            head_dim: Some(128),
            num_kv_layers: None,
            attn_flops_coeff: None,
        },
    },
    // Source: https://huggingface.co/Qwen/Qwen3-0.6B/raw/main/config.json (accessed 2026-07-16)
    ModelEntry {
        // https://huggingface.co/Qwen/Qwen3-0.6B/raw/main/config.json
        tokens: &["qwen3", "0.6b"],
        entry: catalog_dense! {
            family: "qwen3",
            param_count: 600_000_000,
            active_param_count: None,
            num_layers: 28,
            hidden_dim: 1024,
            default_weight_dtype: "bf16",
            num_kv_heads: Some(8),
            head_dim: Some(128),
            num_kv_layers: None,
            attn_flops_coeff: None,
        },
    },
    // ── Qwen 2.5 dense ───────────────────────────────────────────────────────
    // Source: https://huggingface.co/Qwen/Qwen2.5-72B-Instruct/raw/main/config.json (accessed 2026-07-16)
    ModelEntry {
        tokens: &["qwen2.5", "72b"],
        entry: catalog_dense! {
            family: "qwen2.5",
            param_count: 72 * B,
            active_param_count: None,
            num_layers: 80,
            hidden_dim: 8192,
            default_weight_dtype: "bf16",
            num_kv_heads: Some(8),
            head_dim: Some(128),
            num_kv_layers: None,
            attn_flops_coeff: None,
        },
    },
    // Source: https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/raw/main/config.json (accessed 2026-07-16)
    ModelEntry {
        tokens: &["qwen2.5", "32b"],
        entry: catalog_dense! {
            family: "qwen2.5",
            param_count: 32 * B,
            active_param_count: None,
            num_layers: 64,
            hidden_dim: 5120,
            default_weight_dtype: "bf16",
            num_kv_heads: Some(8),
            head_dim: Some(128),
            num_kv_layers: None,
            attn_flops_coeff: None,
        },
    },
    // Source: https://huggingface.co/Qwen/Qwen2.5-14B-Instruct/raw/main/config.json (accessed 2026-07-16)
    ModelEntry {
        tokens: &["qwen2.5", "14b"],
        entry: catalog_dense! {
            family: "qwen2.5",
            param_count: 14 * B,
            active_param_count: None,
            num_layers: 48,
            hidden_dim: 5120,
            default_weight_dtype: "bf16",
            num_kv_heads: Some(8),
            head_dim: Some(128),
            num_kv_layers: None,
            attn_flops_coeff: None,
        },
    },
    // Source: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/raw/main/config.json (accessed 2026-07-16)
    ModelEntry {
        tokens: &["qwen2.5", "7b"],
        entry: catalog_dense! {
            family: "qwen2.5",
            param_count: 7 * B,
            active_param_count: None,
            num_layers: 28,
            hidden_dim: 3584,
            default_weight_dtype: "bf16",
            num_kv_heads: Some(4),
            head_dim: Some(128),
            num_kv_layers: None,
            attn_flops_coeff: None,
        },
    },
    // ── DeepSeek V3 / R1 ─────────────────────────────────────────────────────
    // Uses MLA (Multi-head Latent Attention): KV cache is a compressed latent,
    // not num_kv_heads × head_dim. Standard formula doesn't apply.
    // 671B MoE (37B active)
    // Source: https://huggingface.co/deepseek-ai/DeepSeek-V3/raw/main/config.json (accessed 2026-07-16)
    ModelEntry {
        tokens: &["deepseek", "671b"],
        entry: catalog_dense! {
            family: "deepseek",
            param_count: 671 * B,
            active_param_count: Some(37 * B),
            num_layers: 61,
            hidden_dim: 7168,
            default_weight_dtype: "bf16",
            num_kv_heads: None,
            head_dim: None,
            num_kv_layers: None,
            attn_flops_coeff: Some(139_264),
        },
    },
    // R1 distills are Llama/Qwen dense arches (NOT MLA). Place before generic
    // ["deepseek","r1"] so "DeepSeek-R1-Distill-*" never inherits 671B MLA params.
    // Source: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B/raw/main/config.json (accessed 2026-07-16)
    ModelEntry {
        // https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B/raw/main/config.json
        tokens: &["deepseek", "r1", "distill", "70b"],
        entry: catalog_dense! {
            family: "deepseek",
            param_count: 70 * B,
            active_param_count: None,
            num_layers: 80,
            hidden_dim: 8192,
            default_weight_dtype: "bf16",
            num_kv_heads: Some(8),
            head_dim: Some(128),
            num_kv_layers: None,
            attn_flops_coeff: None,
        },
    },
    // Source: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B/raw/main/config.json (accessed 2026-07-16)
    ModelEntry {
        // https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B/raw/main/config.json
        tokens: &["deepseek", "r1", "distill", "32b"],
        entry: catalog_dense! {
            family: "deepseek",
            param_count: 32 * B,
            active_param_count: None,
            num_layers: 64,
            hidden_dim: 5120,
            default_weight_dtype: "bf16",
            num_kv_heads: Some(8),
            head_dim: Some(128),
            num_kv_layers: None,
            attn_flops_coeff: None,
        },
    },
    // Source: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B/raw/main/config.json (accessed 2026-07-16)
    ModelEntry {
        // https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B/raw/main/config.json
        tokens: &["deepseek", "r1", "distill", "14b"],
        entry: catalog_dense! {
            family: "deepseek",
            param_count: 14 * B,
            active_param_count: None,
            num_layers: 48,
            hidden_dim: 5120,
            default_weight_dtype: "bf16",
            num_kv_heads: Some(8),
            head_dim: Some(128),
            num_kv_layers: None,
            attn_flops_coeff: None,
        },
    },
    // Source: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/raw/main/config.json (accessed 2026-07-16)
    ModelEntry {
        // https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/raw/main/config.json
        tokens: &["deepseek", "r1", "distill", "7b"],
        entry: catalog_dense! {
            family: "deepseek",
            param_count: 7 * B,
            active_param_count: None,
            num_layers: 28,
            hidden_dim: 3584,
            default_weight_dtype: "bf16",
            num_kv_heads: Some(4),
            head_dim: Some(128),
            num_kv_layers: None,
            attn_flops_coeff: None,
        },
    },
    // Source: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B/raw/main/config.json (accessed 2026-07-16)
    ModelEntry {
        // https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B/raw/main/config.json
        tokens: &["deepseek", "r1", "distill", "8b"],
        entry: catalog_dense! {
            family: "deepseek",
            param_count: 8 * B,
            active_param_count: None,
            num_layers: 32,
            hidden_dim: 4096,
            default_weight_dtype: "bf16",
            num_kv_heads: Some(8),
            head_dim: Some(128),
            num_kv_layers: None,
            attn_flops_coeff: None,
        },
    },
    // Source: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B/raw/main/config.json (accessed 2026-07-16)
    ModelEntry {
        // https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B/raw/main/config.json
        tokens: &["deepseek", "r1", "distill", "1.5b"],
        entry: catalog_dense! {
            family: "deepseek",
            param_count: 1_500_000_000,
            active_param_count: None,
            num_layers: 28,
            hidden_dim: 1536,
            default_weight_dtype: "bf16",
            num_kv_heads: Some(2),
            head_dim: Some(128),
            num_kv_layers: None,
            attn_flops_coeff: None,
        },
    },
    // R1 without size token defaults to 671B MLA
    // Source: https://huggingface.co/deepseek-ai/DeepSeek-R1/raw/main/config.json (accessed 2026-07-16)
    ModelEntry {
        tokens: &["deepseek", "r1"],
        entry: catalog_dense! {
            family: "deepseek",
            param_count: 671 * B,
            active_param_count: Some(37 * B),
            num_layers: 61,
            hidden_dim: 7168,
            default_weight_dtype: "bf16",
            num_kv_heads: None,
            head_dim: None,
            num_kv_layers: None,
            attn_flops_coeff: Some(139_264),
        },
    },
    // V3 without size token defaults to 671B
    // Source: https://huggingface.co/deepseek-ai/DeepSeek-V3/raw/main/config.json (accessed 2026-07-16)
    ModelEntry {
        tokens: &["deepseek", "v3"],
        entry: catalog_dense! {
            family: "deepseek",
            param_count: 671 * B,
            active_param_count: Some(37 * B),
            num_layers: 61,
            hidden_dim: 7168,
            default_weight_dtype: "bf16",
            num_kv_heads: None,
            head_dim: None,
            num_kv_layers: None,
            attn_flops_coeff: Some(139_264),
        },
    },
    // Source: https://huggingface.co/deepseek-ai/deepseek-llm-70b-chat/raw/main/config.json (accessed 2026-07-16)
    ModelEntry {
        tokens: &["deepseek", "70b"],
        entry: catalog_dense! {
            family: "deepseek",
            param_count: 70 * B,
            active_param_count: None,
            num_layers: 80,
            hidden_dim: 8192,
            default_weight_dtype: "bf16",
            num_kv_heads: None,
            head_dim: None,
            num_kv_layers: None,
            attn_flops_coeff: Some(69_632),
        },
    },
    // Source: https://huggingface.co/deepseek-ai/deepseek-llm-7b-chat/raw/main/config.json (accessed 2026-07-16)
    ModelEntry {
        tokens: &["deepseek", "7b"],
        entry: catalog_dense! {
            family: "deepseek",
            param_count: 7 * B,
            active_param_count: None,
            num_layers: 30,
            hidden_dim: 4096,
            default_weight_dtype: "bf16",
            num_kv_heads: None,
            head_dim: None,
            num_kv_layers: None,
            attn_flops_coeff: Some(34_816),
        },
    },
    // ── Mistral Large 3 ──────────────────────────────────────────────────────
    // MLA (kv_lora_rank in params.json): standard KV formula doesn't apply.
    // 675B total / 41B active from model card. dim=7168, n_layers=61 from params.json.
    // Source: https://huggingface.co/mistralai/Mistral-Large-3-675B-Instruct-2512/raw/main/params.json (accessed 2026-07-16)
    ModelEntry {
        tokens: &["mistral", "large", "675b"],
        entry: catalog_dense! {
            family: "mistral",
            param_count: 675 * B,
            active_param_count: Some(41 * B),
            num_layers: 61,
            hidden_dim: 7168,
            default_weight_dtype: "bf16",
            num_kv_heads: None,
            head_dim: None,
            num_kv_layers: None,
            attn_flops_coeff: None,
        },
    },
    // 123B dense
    // Source: https://huggingface.co/mistralai/Mistral-Large-Instruct-2407/raw/main/config.json (accessed 2026-07-16)
    ModelEntry {
        tokens: &["mistral", "large", "123b"],
        entry: catalog_dense! {
            family: "mistral",
            param_count: 123 * B,
            active_param_count: None,
            num_layers: 88,
            hidden_dim: 8192,
            default_weight_dtype: "bf16",
            num_kv_heads: Some(8),
            head_dim: Some(128),
            num_kv_layers: None,
            attn_flops_coeff: None,
        },
    },
    // Mistral Large without explicit size, 123B default
    // Source: https://huggingface.co/mistralai/Mistral-Large-Instruct-2407/raw/main/config.json (accessed 2026-07-16)
    ModelEntry {
        tokens: &["mistral", "large"],
        entry: catalog_dense! {
            family: "mistral",
            param_count: 123 * B,
            active_param_count: None,
            num_layers: 88,
            hidden_dim: 8192,
            default_weight_dtype: "bf16",
            num_kv_heads: Some(8),
            head_dim: Some(128),
            num_kv_layers: None,
            attn_flops_coeff: None,
        },
    },
    // ── Mixtral MoE ──────────────────────────────────────────────────────────
    // 8x22B: 141B total, ~39B active
    // Source: https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1/raw/main/config.json (accessed 2026-07-16)
    ModelEntry {
        tokens: &["mixtral", "8x22b"],
        entry: catalog_dense! {
            family: "mistral",
            param_count: 141 * B,
            active_param_count: Some(39 * B),
            num_layers: 56,
            hidden_dim: 6144,
            default_weight_dtype: "bf16",
            num_kv_heads: Some(8),
            head_dim: Some(128),
            num_kv_layers: None,
            attn_flops_coeff: None,
        },
    },
    // 8x7B: 47B total, ~13B active
    // Source: https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/raw/main/config.json (accessed 2026-07-16)
    ModelEntry {
        tokens: &["mixtral", "8x7b"],
        entry: catalog_dense! {
            family: "mistral",
            param_count: 47 * B,
            active_param_count: Some(13 * B),
            num_layers: 32,
            hidden_dim: 4096,
            default_weight_dtype: "bf16",
            num_kv_heads: Some(8),
            head_dim: Some(128),
            num_kv_layers: None,
            attn_flops_coeff: None,
        },
    },
    // ── Mistral 7B dense ─────────────────────────────────────────────────────
    // Source: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/raw/main/config.json (accessed 2026-07-16)
    ModelEntry {
        tokens: &["mistral", "7b"],
        entry: catalog_dense! {
            family: "mistral",
            param_count: 7 * B,
            active_param_count: None,
            num_layers: 32,
            hidden_dim: 4096,
            default_weight_dtype: "bf16",
            num_kv_heads: Some(8),
            head_dim: Some(128),
            num_kv_layers: None,
            attn_flops_coeff: None,
        },
    },
    // ── Gemma 2 / 3 / 4 ──────────────────────────────────────────────────────
    // Gemma 4 flagship: Google markets as "27B"; actual param count ~31B.
    // Two token entries cover both HuggingFace name ("gemma-4-27b-it") and
    // quantized GGUFs ("gemma-4-31b-it-bf16.gguf"). Both must appear before
    // the Gemma 2 27B entry, otherwise "gemma-4-27b-it" falls through and
    // matches Gemma 2 27B (wrong arch params).
    // KV architecture non-standard (per-token head quantization): num_kv_heads = None.
    // Source: https://huggingface.co/google/gemma-4-27b-it/raw/main/config.json (accessed 2026-07-16)
    ModelEntry {
        tokens: &["gemma", "4", "27b"],
        entry: catalog_dense! {
            family: "gemma4",
            param_count: 31 * B,
            active_param_count: None,
            num_layers: 60,
            hidden_dim: 5376,
            default_weight_dtype: "bf16",
            num_kv_heads: None,
            head_dim: None,
            num_kv_layers: None,
            attn_flops_coeff: None,
        },
    },
    // Source: https://huggingface.co/google/gemma-4-27b-it/raw/main/config.json (accessed 2026-07-16)
    ModelEntry {
        tokens: &["gemma", "4", "31b"],
        entry: catalog_dense! {
            family: "gemma4",
            param_count: 31 * B,
            active_param_count: None,
            num_layers: 60,
            hidden_dim: 5376,
            default_weight_dtype: "bf16",
            num_kv_heads: None,
            head_dim: None,
            num_kv_layers: None,
            attn_flops_coeff: None,
        },
    },
    // Gemma 4 26B-A4B MoE
    // Source: https://huggingface.co/google/gemma-4-26b-it/raw/main/config.json (accessed 2026-07-16)
    ModelEntry {
        tokens: &["gemma", "4", "26b"],
        entry: catalog_dense! {
            family: "gemma4",
            param_count: 26 * B,
            active_param_count: Some(4 * B),
            num_layers: 30, // Verified from HF config
            hidden_dim: 2816, // Verified from HF config
            default_weight_dtype: "bf16",
            num_kv_heads: None,
            head_dim: None,
            num_kv_layers: None,
            attn_flops_coeff: None,
        },
    },
    // ── Gemma 3 ──────────────────────────────────────────────────────────────
    // After Gemma 4, before Gemma 2. Hyphenated names normalize to "gemma 3 …"
    // (token "gemma 3", not bare "3") so "gemma-2-27b-3bit" cannot false-match.
    // Fused "gemma3-27b" covered by parallel entries. Sources: HF text_config.
    // Source: https://huggingface.co/google/gemma-3-27b-it/raw/main/config.json (accessed 2026-07-16)
    ModelEntry {
        // https://huggingface.co/google/gemma-3-27b-it (text_config)
        tokens: &["gemma 3", "27b"],
        entry: catalog_dense! {
            family: "gemma3",
            param_count: 27 * B,
            active_param_count: None,
            num_layers: 62,
            hidden_dim: 5376,
            default_weight_dtype: "bf16",
            num_kv_heads: Some(16),
            head_dim: Some(128),
            num_kv_layers: None,
            attn_flops_coeff: None,
        },
    },
    // Source: https://huggingface.co/google/gemma-3-27b-it/raw/main/config.json (accessed 2026-07-16)
    ModelEntry {
        tokens: &["gemma3", "27b"],
        entry: catalog_dense! {
            family: "gemma3",
            param_count: 27 * B,
            active_param_count: None,
            num_layers: 62,
            hidden_dim: 5376,
            default_weight_dtype: "bf16",
            num_kv_heads: Some(16),
            head_dim: Some(128),
            num_kv_layers: None,
            attn_flops_coeff: None,
        },
    },
    // Source: https://huggingface.co/google/gemma-3-12b-it/raw/main/config.json (accessed 2026-07-16)
    ModelEntry {
        // https://huggingface.co/google/gemma-3-12b-it (text_config); head_dim=256
        // from Gemma 3 reference config (gm.nn.Gemma3_12B).
        tokens: &["gemma 3", "12b"],
        entry: catalog_dense! {
            family: "gemma3",
            param_count: 12 * B,
            active_param_count: None,
            num_layers: 48,
            hidden_dim: 3840,
            default_weight_dtype: "bf16",
            num_kv_heads: Some(8),
            head_dim: Some(256),
            num_kv_layers: None,
            attn_flops_coeff: None,
        },
    },
    // Source: https://huggingface.co/google/gemma-3-12b-it/raw/main/config.json (accessed 2026-07-16)
    ModelEntry {
        tokens: &["gemma3", "12b"],
        entry: catalog_dense! {
            family: "gemma3",
            param_count: 12 * B,
            active_param_count: None,
            num_layers: 48,
            hidden_dim: 3840,
            default_weight_dtype: "bf16",
            num_kv_heads: Some(8),
            head_dim: Some(256),
            num_kv_layers: None,
            attn_flops_coeff: None,
        },
    },
    // Source: https://huggingface.co/google/gemma-3-4b-it/raw/main/config.json (accessed 2026-07-16)
    ModelEntry {
        // https://huggingface.co/google/gemma-3-4b-it text_config (+ head/kv from
        // published Gemma 3 configs: num_attention_heads=8, num_key_value_heads=4,
        // head_dim=256).
        tokens: &["gemma 3", "4b"],
        entry: catalog_dense! {
            family: "gemma3",
            param_count: 4 * B,
            active_param_count: None,
            num_layers: 34,
            hidden_dim: 2560,
            default_weight_dtype: "bf16",
            num_kv_heads: Some(4),
            head_dim: Some(256),
            num_kv_layers: None,
            attn_flops_coeff: None,
        },
    },
    // Source: https://huggingface.co/google/gemma-3-4b-it/raw/main/config.json (accessed 2026-07-16)
    ModelEntry {
        tokens: &["gemma3", "4b"],
        entry: catalog_dense! {
            family: "gemma3",
            param_count: 4 * B,
            active_param_count: None,
            num_layers: 34,
            hidden_dim: 2560,
            default_weight_dtype: "bf16",
            num_kv_heads: Some(4),
            head_dim: Some(256),
            num_kv_layers: None,
            attn_flops_coeff: None,
        },
    },
    // Source: https://huggingface.co/google/gemma-3-1b-it/raw/main/config.json (accessed 2026-07-16)
    ModelEntry {
        // https://huggingface.co/google/gemma-3-1b-it/raw/main/config.json
        tokens: &["gemma 3", "1b"],
        entry: catalog_dense! {
            family: "gemma3",
            param_count: B,
            active_param_count: None,
            num_layers: 26,
            hidden_dim: 1152,
            default_weight_dtype: "bf16",
            num_kv_heads: Some(1),
            head_dim: Some(256),
            num_kv_layers: None,
            attn_flops_coeff: None,
        },
    },
    // Source: https://huggingface.co/google/gemma-3-1b-it/raw/main/config.json (accessed 2026-07-16)
    ModelEntry {
        tokens: &["gemma3", "1b"],
        entry: catalog_dense! {
            family: "gemma3",
            param_count: B,
            active_param_count: None,
            num_layers: 26,
            hidden_dim: 1152,
            default_weight_dtype: "bf16",
            num_kv_heads: Some(1),
            head_dim: Some(256),
            num_kv_layers: None,
            attn_flops_coeff: None,
        },
    },
    // Gemma 2 27B: head_dim=128, 32 attn heads, 16 KV heads.
    // Source: https://huggingface.co/google/gemma-2-27b-it/raw/main/config.json (accessed 2026-07-16)
    ModelEntry {
        tokens: &["gemma", "27b"],
        entry: catalog_dense! {
            family: "gemma",
            param_count: 27 * B,
            active_param_count: None,
            num_layers: 46,
            hidden_dim: 4608,
            default_weight_dtype: "bf16",
            num_kv_heads: Some(16),
            head_dim: Some(128),
            num_kv_layers: None,
            attn_flops_coeff: None,
        },
    },
    // Gemma 2 9B: head_dim=256 (larger than typical), 16 attn heads, 8 KV heads.
    // Source: https://huggingface.co/google/gemma-2-9b-it/raw/main/config.json (accessed 2026-07-16)
    ModelEntry {
        tokens: &["gemma", "9b"],
        entry: catalog_dense! {
            family: "gemma",
            param_count: 9 * B,
            active_param_count: None,
            num_layers: 42,
            hidden_dim: 3584,
            default_weight_dtype: "bf16",
            num_kv_heads: Some(8),
            head_dim: Some(256),
            num_kv_layers: None,
            attn_flops_coeff: None,
        },
    },
    // ── Kimi K2 ──────────────────────────────────────────────────────────────
    // MLA (kv_lora_rank in config): standard KV formula doesn't apply.
    // Source: https://huggingface.co/moonshotai/Kimi-K2-Instruct/raw/main/config.json (accessed 2026-07-16)
    ModelEntry {
        tokens: &["kimi", "k2"],
        entry: catalog_dense! {
            family: "kimi",
            param_count: 1_000 * B,
            active_param_count: Some(32 * B),
            num_layers: 61,
            hidden_dim: 7168,
            default_weight_dtype: "bf16",
            num_kv_heads: None,
            head_dim: None,
            num_kv_layers: None,
            attn_flops_coeff: None,
        },
    },
    // ── GLM-5.2 MoE ───────────────────────────────────────────────────────────
    // glm_moe_dsa + sparse/index attention (kv_lora_rank in config): standard KV formula doesn't apply.
    // Total/active params from https://huggingface.co/zai-org/GLM-5.2 model card (753B / 40B).
    // Source: https://huggingface.co/zai-org/GLM-5.2/raw/main/config.json (accessed 2026-07-16)
    ModelEntry {
        tokens: &["glm", "5.2"],
        entry: catalog_dense! {
            family: "glm",
            param_count: 753 * B,
            active_param_count: Some(40 * B),
            num_layers: 78,
            hidden_dim: 6144,
            default_weight_dtype: "bf16",
            num_kv_heads: None,
            head_dim: None,
            num_kv_layers: None,
            attn_flops_coeff: None,
        },
    },
    // Source: https://huggingface.co/zai-org/GLM-5.2/raw/main/config.json (accessed 2026-07-16)
    ModelEntry {
        tokens: &["glm", "744b"],
        entry: catalog_dense! {
            family: "glm",
            param_count: 753 * B,
            active_param_count: Some(40 * B),
            num_layers: 78,
            hidden_dim: 6144,
            default_weight_dtype: "bf16",
            num_kv_heads: None,
            head_dim: None,
            num_kv_layers: None,
            attn_flops_coeff: None,
        },
    },
    // Source: https://huggingface.co/zai-org/GLM-4-32B-0414/raw/main/config.json (accessed 2026-07-16)
    ModelEntry {
        tokens: &["glm", "32b"],
        entry: catalog_dense! {
            family: "glm",
            param_count: 32 * B,
            active_param_count: None,
            num_layers: 61,
            hidden_dim: 6144,
            default_weight_dtype: "bf16",
            num_kv_heads: Some(2),
            head_dim: Some(128),
            num_kv_layers: None,
            attn_flops_coeff: None,
        },
    },
    // ── Inkling (Thinking Machines) ───────────────────────────────────────────
    // Hybrid sliding-window + global attention; separate SWA/GQA KV fields in text_config.
    // Total/active params from https://huggingface.co/thinkingmachines/Inkling model card (975B / 41B).
    // Source: https://huggingface.co/thinkingmachines/Inkling/raw/main/config.json (accessed 2026-07-16)
    ModelEntry {
        tokens: &["inkling"],
        entry: catalog_dense! {
            family: "inkling",
            param_count: 975 * B,
            active_param_count: Some(41 * B),
            num_layers: 66,
            hidden_dim: 6144,
            default_weight_dtype: "bf16",
            num_kv_heads: None,
            head_dim: None,
            num_kv_layers: None,
            attn_flops_coeff: None,
        },
    },
    // ── Phi-4 ────────────────────────────────────────────────────────────────
    // (Phi-4 32B phantom entry removed, no official release at this size)
    // Phi-4 14B: 40 attn heads, 10 KV heads (GQA 4:1), head_dim=128.
    // Source: https://huggingface.co/microsoft/phi-4/raw/main/config.json (accessed 2026-07-16)
    ModelEntry {
        tokens: &["phi", "4", "14b"],
        entry: catalog_dense! {
            family: "phi4",
            param_count: 14 * B,
            active_param_count: None,
            num_layers: 40,
            hidden_dim: 5120,
            default_weight_dtype: "bf16",
            num_kv_heads: Some(10),
            head_dim: Some(128),
            num_kv_layers: None,
            attn_flops_coeff: None,
        },
    },
    // Phi-4 without explicit size, 14B default
    // Source: https://huggingface.co/microsoft/phi-4/raw/main/config.json (accessed 2026-07-16)
    ModelEntry {
        tokens: &["phi", "4"],
        entry: catalog_dense! {
            family: "phi4",
            param_count: 14 * B,
            active_param_count: None,
            num_layers: 40,
            hidden_dim: 5120,
            default_weight_dtype: "bf16",
            num_kv_heads: Some(10),
            head_dim: Some(128),
            num_kv_layers: None,
            attn_flops_coeff: None,
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
            assert!(e.active_param_count.is_none());
        }
    }

    #[test]
    fn llama4_maverick() {
        let e = lookup_model("meta-llama/Llama-4-Maverick-17B-128E-Instruct")
            .expect("should match maverick");
        assert_eq!(e.family, "llama4");
        assert_eq!(e.active_param_count, Some(17 * B));
        assert!(e.active_param_count.is_some());
    }

    #[test]
    fn llama4_scout() {
        let e =
            lookup_model("meta-llama/Llama-4-Scout-17B-16E-Instruct").expect("should match scout");
        assert_eq!(e.family, "llama4");
        assert!(e.active_param_count.is_some());
    }

    #[test]
    fn qwen3_moe_235b() {
        let e = lookup_model("Qwen/Qwen3-235B-A22B").expect("should match qwen3 235b moe");
        assert_eq!(e.family, "qwen3");
        assert_eq!(e.param_count, 235 * B);
        assert_eq!(e.active_param_count, Some(22 * B));
        assert!(e.active_param_count.is_some());
    }

    #[test]
    fn qwen25_72b() {
        let e = lookup_model("Qwen/Qwen2.5-72B-Instruct").expect("no match");
        assert_eq!(e.family, "qwen2.5");
        assert_eq!(e.param_count, 72 * B);
        assert!(e.active_param_count.is_none());
    }

    #[test]
    fn deepseek_r1_671b() {
        let e = lookup_model("deepseek-ai/DeepSeek-R1").expect("no match");
        assert_eq!(e.family, "deepseek");
        assert_eq!(e.param_count, 671 * B);
        assert_eq!(e.active_param_count, Some(37 * B));
        assert!(e.num_kv_heads.is_none());
        assert!(e.head_dim.is_none());
    }

    #[test]
    fn deepseek_v3_671b() {
        let e = lookup_model("deepseek-ai/DeepSeek-V3").expect("no match");
        assert_eq!(e.family, "deepseek");
        assert_eq!(e.param_count, 671 * B);
        assert_eq!(e.active_param_count, Some(37 * B));
        assert!(e.num_kv_heads.is_none());
        assert!(e.head_dim.is_none());
    }

    #[test]
    fn mixtral_8x7b() {
        let e = lookup_model("mistralai/Mixtral-8x7B-Instruct-v0.1").expect("no match");
        assert_eq!(e.family, "mistral");
        assert_eq!(e.param_count, 47 * B);
        assert!(e.active_param_count.is_some());
    }

    #[test]
    fn mixtral_8x22b() {
        let e = lookup_model("mistralai/Mixtral-8x22B-Instruct-v0.1").expect("no match");
        assert_eq!(e.family, "mistral");
        assert_eq!(e.param_count, 141 * B);
        assert!(e.active_param_count.is_some());
    }

    #[test]
    fn mistral_large_675b() {
        let e = lookup_model("mistralai/Mistral-Large-3-675B-Instruct-2512").expect("no match");
        assert_eq!(e.family, "mistral");
        assert_eq!(e.param_count, 675 * B);
        assert_eq!(e.active_param_count, Some(41 * B));
        assert_eq!(e.num_layers, 61);
        assert_eq!(e.hidden_dim, 7168);
        assert!(e.num_kv_heads.is_none());
        assert!(e.head_dim.is_none());
    }

    #[test]
    fn mistral_large_123b() {
        let e = lookup_model("mistralai/Mistral-Large-Instruct-2411").expect("no match");
        assert_eq!(e.family, "mistral");
        assert_eq!(e.param_count, 123 * B);
        assert!(e.active_param_count.is_none());
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
        assert_eq!(e.active_param_count, Some(32 * B));
        assert_eq!(e.num_layers, 61);
        assert_eq!(e.hidden_dim, 7168);
        assert!(e.num_kv_heads.is_none());
        assert!(e.head_dim.is_none());
    }

    #[test]
    fn phi4_14b() {
        let e = lookup_model("microsoft/phi-4").expect("no match");
        assert_eq!(e.family, "phi4");
        assert_eq!(e.param_count, 14 * B);
    }

    #[test]
    fn glm_52_token_matches() {
        let e = lookup_model("zai-org/GLM-5.2").expect("no match");
        assert_eq!(e.family, "glm");
        assert_eq!(e.param_count, 753 * B);
        assert_eq!(e.active_param_count, Some(40 * B));
        assert_eq!(e.num_layers, 78);
        assert_eq!(e.hidden_dim, 6144);
        assert!(e.num_kv_heads.is_none());
    }

    #[test]
    fn glm_32b() {
        let e = lookup_model("zai-org/GLM-4-32B-0414").expect("no match");
        assert_eq!(e.family, "glm");
        assert_eq!(e.param_count, 32 * B);
        assert_eq!(e.num_layers, 61);
        assert_eq!(e.hidden_dim, 6144);
        assert_eq!(e.num_kv_heads, Some(2));
        assert_eq!(e.head_dim, Some(128));
    }

    #[test]
    fn inkling_hf_name() {
        let e = lookup_model("thinkingmachines/Inkling").expect("no match");
        assert_eq!(e.family, "inkling");
        assert_eq!(e.param_count, 975 * B);
        assert_eq!(e.active_param_count, Some(41 * B));
        assert_eq!(e.num_layers, 66);
        assert_eq!(e.hidden_dim, 6144);
        assert!(e.num_kv_heads.is_none());
    }

    /// Ground truth from vllm.log (H100 80GB, gpu-mem-util 0.9, max-model-len 32768, 2026-07-16):
    /// KV pool 201,874 tokens; mamba block cap 345; concurrency 24.64x at 8,192.
    /// Catalog-only assertions here; derivation math is a follow-up work package.
    #[test]
    fn qwen36_27b_h100_boot_log_catalog_facts() {
        let e = lookup_model("Qwen/Qwen3.6-27B").expect("no match");
        assert_eq!(e.num_kv_layers, Some(16));
        assert_eq!(e.num_kv_heads, Some(4));
        assert_eq!(e.head_dim, Some(256));
        assert_eq!(e.linear_num_layers, Some(48));
        assert_eq!(e.linear_key_heads, Some(16));
        assert_eq!(e.linear_value_heads, Some(48));
        assert_eq!(e.linear_key_head_dim, Some(128));
        assert_eq!(e.linear_value_head_dim, Some(128));
        assert_eq!(e.linear_conv_kernel_dim, Some(4));
        assert_eq!(e.state_dtype, Some("fp32"));
    }

    #[test]
    fn qwen36_27b() {
        let e = lookup_model("Qwen/Qwen3.6-27B").expect("no match");
        assert_eq!(e.family, "qwen3.6");
        assert_eq!(e.param_count, 27 * B);
        assert!(e.active_param_count.is_none());
        assert_eq!(e.num_layers, 64);
        assert_eq!(e.hidden_dim, 5120);
        assert_eq!(e.num_kv_heads, Some(4));
        assert_eq!(e.head_dim, Some(256));
        assert_eq!(e.num_kv_layers, Some(16));
        assert_eq!(e.linear_num_layers, Some(48));
        assert_eq!(e.linear_key_heads, Some(16));
        assert_eq!(e.linear_value_heads, Some(48));
        assert_eq!(e.linear_key_head_dim, Some(128));
        assert_eq!(e.linear_value_head_dim, Some(128));
        assert_eq!(e.linear_conv_kernel_dim, Some(4));
        assert_eq!(e.state_dtype, Some("fp32"));
    }

    #[test]
    fn gemma_4_27b_hf_name() {
        // Google's HuggingFace name, must NOT fall through to Gemma 2 27B.
        let e = lookup_model("google/gemma-4-pt-27b-it").expect("no match");
        assert_eq!(e.family, "gemma4");
        assert_eq!(e.param_count, 31 * B);
        assert_eq!(e.num_layers, 60);
    }

    #[test]
    fn gemma_4_31b() {
        let e = lookup_model("gemma-4-31b-it-bf16.gguf").expect("no match");
        assert_eq!(e.family, "gemma4");
        assert_eq!(e.param_count, 31 * B);
        assert!(e.active_param_count.is_none());
    }

    #[test]
    fn gemma_4_26b() {
        let e = lookup_model("gemma-4-26b-it-bf16.gguf").expect("no match");
        assert_eq!(e.family, "gemma4");
        assert_eq!(e.param_count, 26 * B);
        assert_eq!(e.active_param_count, Some(4 * B));
        assert!(e.active_param_count.is_some());
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

    #[test]
    fn standard_models_have_no_attn_flops_coeff() {
        for name in [
            "meta-llama/Llama-3-8B",
            "Qwen/Qwen2.5-72B",
            "mistralai/Mistral-7B",
        ] {
            let e = lookup_model(name).expect("catalog hit");
            assert!(
                e.attn_flops_coeff.is_none(),
                "{name} should use standard hidden_dim path"
            );
        }
    }

    #[test]
    fn deepseek_v3_has_mla_coeff() {
        let e = lookup_model("deepseek-ai/DeepSeek-V3").expect("catalog hit");
        assert_eq!(e.attn_flops_coeff, Some(139_264));
    }

    #[test]
    fn deepseek_r1_has_mla_coeff() {
        let e = lookup_model("deepseek-ai/DeepSeek-R1").expect("catalog hit");
        assert_eq!(e.attn_flops_coeff, Some(139_264));
    }

    #[test]
    fn r1_distill_llama_8b_not_671b_mla() {
        let e = lookup_model("deepseek-ai/DeepSeek-R1-Distill-Llama-8B").expect("no match");
        assert_eq!(e.param_count, 8 * B);
        assert!(e.active_param_count.is_none());
        assert_eq!(e.num_kv_heads, Some(8));
        assert_eq!(e.head_dim, Some(128));
        assert!(e.attn_flops_coeff.is_none());
    }

    #[test]
    fn r1_distill_qwen_32b() {
        let e = lookup_model("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B").expect("no match");
        assert_eq!(e.param_count, 32 * B);
        assert_eq!(e.num_layers, 64);
        assert_eq!(e.num_kv_heads, Some(8));
    }

    #[test]
    fn r1_distill_llama_70b() {
        let e = lookup_model("deepseek-ai/DeepSeek-R1-Distill-Llama-70B").expect("no match");
        assert_eq!(e.param_count, 70 * B);
        assert_eq!(e.num_layers, 80);
        assert_eq!(e.num_kv_heads, Some(8));
    }

    #[test]
    fn r1_distill_qwen_1_5b() {
        let e = lookup_model("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B").expect("no match");
        assert_eq!(e.param_count, 1_500_000_000);
        assert_eq!(e.num_kv_heads, Some(2));
        assert_eq!(e.head_dim, Some(128));
    }

    #[test]
    fn gemma3_27b_not_gemma2() {
        let e = lookup_model("google/gemma-3-27b-it").expect("no match");
        assert_eq!(e.family, "gemma3");
        assert_eq!(e.param_count, 27 * B);
        assert_eq!(e.num_layers, 62);
        assert_eq!(e.hidden_dim, 5376);
        assert_eq!(e.num_kv_heads, Some(16));
        assert_eq!(e.head_dim, Some(128));
    }

    #[test]
    fn gemma3_compact_name() {
        let e = lookup_model("gemma3-27b").expect("no match");
        assert_eq!(e.family, "gemma3");
        assert_eq!(e.param_count, 27 * B);
        assert_eq!(e.num_layers, 62);
    }

    #[test]
    fn gemma2_quant_with_3bit_does_not_match_gemma3() {
        // Bare "3" token would false-match "3bit"; spaced "gemma 3" must not.
        let e = lookup_model("google/gemma-2-27b-3bit").expect("gemma2");
        assert_eq!(e.family, "gemma");
        assert_eq!(e.num_layers, 46);
        assert_eq!(e.hidden_dim, 4608);
    }

    #[test]
    fn gemma_27b_v3_does_not_match_gemma3() {
        let e = lookup_model("vendor/gemma-27b-v3").expect("gemma2");
        assert_eq!(e.family, "gemma");
    }

    #[test]
    fn gemma3_4b_and_1b() {
        let e4 = lookup_model("google/gemma-3-4b-it").expect("4b");
        assert_eq!(e4.family, "gemma3");
        assert_eq!(e4.num_layers, 34);
        assert_eq!(e4.num_kv_heads, Some(4));
        let e1 = lookup_model("google/gemma-3-1b-it").expect("1b");
        assert_eq!(e1.family, "gemma3");
        assert_eq!(e1.num_layers, 26);
        assert_eq!(e1.num_kv_heads, Some(1));
    }

    #[test]
    fn gemma4_still_guarded_after_gemma3() {
        let e = lookup_model("google/gemma-4-pt-27b-it").expect("gemma4");
        assert_eq!(e.family, "gemma4");
        assert_eq!(e.param_count, 31 * B);
    }

    #[test]
    fn gemma2_27b_still_matches() {
        let e = lookup_model("google/gemma-2-27b-it").expect("gemma2");
        assert_eq!(e.family, "gemma");
        assert_eq!(e.num_layers, 46);
        assert_eq!(e.hidden_dim, 4608);
    }

    #[test]
    fn qwen3_phantoms_7b_72b_gone() {
        assert!(lookup_model("Qwen/Qwen3-7B").is_none());
        assert!(lookup_model("Qwen/Qwen3-72B").is_none());
        // Explicit Qwen2.5 still works.
        let e = lookup_model("Qwen/Qwen2.5-7B").expect("qwen2.5");
        assert_eq!(e.family, "qwen2.5");
        assert_eq!(e.param_count, 7 * B);
    }

    #[test]
    fn qwen3_dense_new_skus() {
        let e06 = lookup_model("Qwen/Qwen3-0.6B").expect("0.6b");
        assert_eq!(e06.param_count, 600_000_000);
        assert_eq!(e06.hidden_dim, 1024);
        let e17 = lookup_model("Qwen/Qwen3-1.7B").expect("1.7b");
        assert_eq!(e17.param_count, 1_700_000_000);
        let e4 = lookup_model("Qwen/Qwen3-4B").expect("4b");
        assert_eq!(e4.num_layers, 36);
        assert_eq!(e4.hidden_dim, 2560);
        let e8 = lookup_model("Qwen/Qwen3-8B").expect("8b");
        assert_eq!(e8.num_layers, 36);
        assert_eq!(e8.hidden_dim, 4096);
        assert_eq!(e8.num_kv_heads, Some(8));
    }

    #[test]
    fn qwen3_30b_a3b_kv_fields_enable_kv_math() {
        let e = lookup_model("Qwen/Qwen3-30B-A3B").expect("a3b");
        assert_eq!(e.num_kv_heads, Some(4));
        assert_eq!(e.head_dim, Some(128));
        // Catalog fields alone are the gate for kv_max_concurrent_seqs (engine).
        // Previously both were None and KV math short-circuited.
        assert!(e.num_kv_heads.is_some() && e.head_dim.is_some());
    }

    #[test]
    fn llama4_has_zero_coeff() {
        let e = lookup_model("meta-llama/Llama-4-Maverick-17B-128E-Instruct").expect("catalog hit");
        assert_eq!(e.attn_flops_coeff, Some(0));
    }
}
