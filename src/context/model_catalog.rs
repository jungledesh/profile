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
    /// Total parameter count.
    pub param_count: u64,
    /// Active parameter count for MoE (experts on path) or multimodal LM-only
    /// decode (e.g. Muse text stack). None for dense LM-only models.
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
    /// Sliding-window size in tokens. `None` means no windowed layers.
    pub swa_window: Option<u32>,
    /// Number of layers whose transcript is capped by `swa_window`.
    pub num_swa_layers: Option<u32>,
}

/// Pure-attention catalog row. Named fields are transposition-proof; hybrid/linear
/// state fields default to `None` inside the expansion.
macro_rules! catalog_dense {
    (
        param_count: $param_count:expr,
        active_param_count: $active_param_count:expr,
        num_layers: $num_layers:expr,
        hidden_dim: $hidden_dim:expr,
        default_weight_dtype: $default_weight_dtype:expr,
        num_kv_heads: $num_kv_heads:expr,
        head_dim: $head_dim:expr,
        num_kv_layers: $num_kv_layers:expr,
        attn_flops_coeff: $attn_flops_coeff:expr
        $(, swa_window: $swa_window:expr, num_swa_layers: $num_swa_layers:expr)?
        $(,)?
    ) => {
        CatalogEntry {
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
            swa_window: catalog_dense!(@optional $($swa_window)?),
            num_swa_layers: catalog_dense!(@optional $($num_swa_layers)?),
        }
    };
    (@optional) => {
        None
    };
    (@optional $value:expr) => {
        Some($value)
    };
}

/// Hybrid catalog row (linear_* / state_dtype set by name).
macro_rules! catalog_hybrid {
    (
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
            swa_window: None,
            num_swa_layers: None,
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
    // Omitted only when 4-bit weights cannot fit the largest catalog GPU
    // (~288 GB: B300 / MI355X). bf16 or quantized (fp8 / 4-bit) that fits
    // at least one supported card stays in. Qwen3 has no official 7B/72B dense SKU.
    // ── Llama 4 ──────────────────────────────────────────────────────────────
    // Interleaved attention: standard KV formula does not apply.
    // Maverick: 17B active / 400B total MoE (model card).
    // Source: https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct/raw/main/config.json
    // (gated). Verified 2026-08-13 against Unsloth copy: hidden_size=5120,
    // num_hidden_layers=48, num_key_value_heads=8, head_dim=128,
    // attention_chunk_size=8192. KV fields stay None: iRoPE / chunked
    // attention is not the standard per-head KV formula.
    ModelEntry {
        tokens: &["llama", "4", "maverick"],
        entry: catalog_dense! {
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
    // Scout: 17B active / 109B total MoE (model card).
    // Source: https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct/raw/main/config.json
    // (gated). Verified 2026-08-13 against Unsloth copy: same text geometry
    // as Maverick (5120 / 48); 16 experts vs 128. KV None for the same reason.
    ModelEntry {
        tokens: &["llama", "4", "scout"],
        entry: catalog_dense! {
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
    // Source: https://huggingface.co/nvidia/Llama-3.1-Nemotron-70B-Instruct-HF/raw/main/config.json (accessed 2026-08-13)
    // (nvidia/Llama-3.1-Nemotron-70B-Instruct 404s; -HF is the public config)
    ModelEntry {
        tokens: &["nemotron", "70b"],
        entry: catalog_dense! {
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
    // ── Muse Glimmer ─────────────────────────────────────────────────────────
    // Dense multimodal (text + perception encoder). Text stack: 52 layers,
    // GQA 32/2, sliding_window=2048, layer_types 3×sliding + 1×full repeating
    // → 39 SWA + 13 full. Card total ~29.6B includes ~1.8B ViT; decode streams
    // the text stack only, so active_param_count is LM-only (~27.8B) for the
    // roof while param_count keeps the full footprint for weight/OOM.
    // Source: https://huggingface.co/meta-models/Muse-Glimmer-30B/raw/main/config.json
    // (text_config; accessed 2026-08-10). ViT ~1.8B from model card.
    ModelEntry {
        tokens: &["muse", "glimmer", "30b"],
        entry: catalog_dense! {
            param_count: 29_600_000_000,
            active_param_count: Some(27_800_000_000),
            num_layers: 52,
            hidden_dim: 6656,
            default_weight_dtype: "bf16",
            num_kv_heads: Some(2),
            head_dim: Some(128),
            num_kv_layers: None,
            attn_flops_coeff: None,
            swa_window: 2048,
            num_swa_layers: 39,
        },
    },
    // ── Llama 3.2 (before generic llama 3; "3.2" token, dots kept by normalize) ─
    // Official config.json is gated. Geometry verified against the Unsloth
    // copy of Meta's file (same path the Gemma 3 1B entry uses).
    // 3B: hidden 3072 / 24 attention heads → head_dim=128.
    // Source: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct/raw/main/config.json (gated)
    // Verified: https://huggingface.co/unsloth/Llama-3.2-3B-Instruct/raw/main/config.json (accessed 2026-08-13)
    ModelEntry {
        tokens: &["llama", "3.2", "3b"],
        entry: catalog_dense! {
            param_count: 3 * B,
            active_param_count: None,
            num_layers: 28,
            hidden_dim: 3072,
            default_weight_dtype: "bf16",
            num_kv_heads: Some(8),
            head_dim: Some(128),
            num_kv_layers: None,
            attn_flops_coeff: None,
        },
    },
    // Source: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct/raw/main/config.json (gated)
    // Verified: https://huggingface.co/unsloth/Llama-3.2-1B-Instruct/raw/main/config.json (accessed 2026-08-13)
    ModelEntry {
        tokens: &["llama", "3.2", "1b"],
        entry: catalog_dense! {
            param_count: B,
            active_param_count: None,
            num_layers: 16,
            hidden_dim: 2048,
            default_weight_dtype: "bf16",
            num_kv_heads: Some(8),
            head_dim: Some(64),
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
    // hidden_size=4096, num_attention_heads=64, head_dim=128 (not hidden/heads).
    // Source: https://huggingface.co/Qwen/Qwen3-235B-A22B/raw/main/config.json (accessed 2026-08-13)
    ModelEntry {
        tokens: &["qwen3", "235b"],
        entry: catalog_dense! {
            param_count: 235 * B,
            active_param_count: Some(22 * B),
            num_layers: 94,
            hidden_dim: 4096,
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
    // Open 27B of Qwen3.8 (not the 2.4T Max). Same hybrid as 3.6: 3:1 DeltaNet /
    // full attention, 16 KV layers. Multimodal (vision_config).
    // param_count is the HF safetensors total (ViT included). active_param_count
    // stays None until a published text-stack count exists for the decode roof.
    // Source: https://huggingface.co/api/models/Qwen/Qwen3.8-27B (safetensors.total,
    // accessed 2026-08-15) and text_config in
    // https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/config.json
    ModelEntry {
        tokens: &["qwen3.8", "27b"],
        entry: catalog_hybrid! {
            param_count: 27_781_427_952,
            active_param_count: None,
            num_layers: 64,
            hidden_dim: 5120,
            default_weight_dtype: "bf16",
            num_kv_heads: Some(4),
            head_dim: Some(256),
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
    // Source: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/raw/main/config.json (accessed 2026-08-13)
    // head_dim = hidden_size/num_attention_heads = 2048/16 = 128 (not in config).
    ModelEntry {
        tokens: &["qwen2.5", "3b"],
        entry: catalog_dense! {
            param_count: 3 * B,
            active_param_count: None,
            num_layers: 36,
            hidden_dim: 2048,
            default_weight_dtype: "bf16",
            num_kv_heads: Some(2),
            head_dim: Some(128),
            num_kv_layers: None,
            attn_flops_coeff: None,
        },
    },
    // Source: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/raw/main/config.json (accessed 2026-08-13)
    // head_dim = 1536/12 = 128.
    ModelEntry {
        tokens: &["qwen2.5", "1.5b"],
        entry: catalog_dense! {
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
    // Source: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/raw/main/config.json (accessed 2026-08-13)
    // Card: 0.49B parameters. Family rounding matches Qwen3 0.6B → 600M.
    // head_dim = 896/14 = 64. use_sliding_window is false; do not set SWA.
    ModelEntry {
        tokens: &["qwen2.5", "0.5b"],
        entry: catalog_dense! {
            param_count: 500_000_000,
            active_param_count: None,
            num_layers: 24,
            hidden_dim: 896,
            default_weight_dtype: "bf16",
            num_kv_heads: Some(2),
            head_dim: Some(64),
            num_kv_layers: None,
            attn_flops_coeff: None,
        },
    },
    // ── DeepSeek R1 distills / dense ──────────────────────────────────────────
    // Source: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B/raw/main/config.json (accessed 2026-07-16)
    ModelEntry {
        tokens: &["deepseek", "r1", "distill", "70b"],
        entry: catalog_dense! {
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
    // Source: https://huggingface.co/deepseek-ai/deepseek-llm-70b-chat/raw/main/config.json (accessed 2026-07-16)
    ModelEntry {
        tokens: &["deepseek", "70b"],
        entry: catalog_dense! {
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
    // ── Mistral Large (123B). 675B stays out: 4-bit still exceeds ~288 GB. ──
    // Source: https://huggingface.co/mistralai/Mistral-Large-Instruct-2407/raw/main/config.json (accessed 2026-07-16)
    ModelEntry {
        tokens: &["mistral", "large", "123b"],
        entry: catalog_dense! {
            param_count: 123 * B,
            active_param_count: None,
            num_layers: 88,
            hidden_dim: 12288,
            default_weight_dtype: "bf16",
            num_kv_heads: Some(8),
            head_dim: Some(128),
            num_kv_layers: None,
            attn_flops_coeff: None,
        },
    },
    // Mistral Large without explicit size, 123B default.
    // lookup_model refuses 675B / Large-3 names so they cannot land here.
    // Source: https://huggingface.co/mistralai/Mistral-Large-Instruct-2407/raw/main/config.json (accessed 2026-07-16)
    ModelEntry {
        tokens: &["mistral", "large"],
        entry: catalog_dense! {
            param_count: 123 * B,
            active_param_count: None,
            num_layers: 88,
            hidden_dim: 12288,
            default_weight_dtype: "bf16",
            num_kv_heads: Some(8),
            head_dim: Some(128),
            num_kv_layers: None,
            attn_flops_coeff: None,
        },
    },
    // ── Mixtral MoE ──────────────────────────────────────────────────────────
    // Before Mistral 7B: "8x7b" contains "7b" and would otherwise steal 7B.
    // 8x22B: 141B total, ~39B active
    // Source: https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1/raw/main/config.json (accessed 2026-07-16)
    ModelEntry {
        tokens: &["mixtral", "8x22b"],
        entry: catalog_dense! {
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
    // Local and global layers have different KV geometry. The current model
    // cannot represent per-layer-group heads and dimensions, so KV fields stay
    // None and bytes_per_seq intentionally declines instead of guessing.
    // Source: https://huggingface.co/google/gemma-4-27b-it/raw/main/config.json (accessed 2026-07-16)
    ModelEntry {
        tokens: &["gemma", "4", "27b"],
        entry: catalog_dense! {
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
    // Gemma 4 26B-A4B MoE. Its local/global KV geometry also cannot be
    // represented by the current model, so KV fields stay None.
    // Param counts from model card (25.2B total / 3.8B active), not marketing integers.
    // Official repos: google/gemma-4-26B-A4B and google/gemma-4-26B-A4B-it
    // (not google/gemma-4-26b-it). Tokens gemma+4+26b still match both.
    // Param counts: https://huggingface.co/google/gemma-4-26B-A4B-it
    // (model card, accessed 2026-07-23). Architecture fields:
    // https://huggingface.co/google/gemma-4-26B-A4B-it/raw/main/config.json
    // (accessed 2026-07-23)
    ModelEntry {
        tokens: &["gemma", "4", "26b"],
        entry: catalog_dense! {
            param_count: 25_200_000_000,
            active_param_count: Some(3_800_000_000),
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
            param_count: 27 * B,
            active_param_count: None,
            num_layers: 62,
            hidden_dim: 5376,
            default_weight_dtype: "bf16",
            num_kv_heads: Some(16),
            head_dim: Some(128),
            num_kv_layers: None,
            attn_flops_coeff: None,
            swa_window: 1024,
            num_swa_layers: 52,
        },
    },
    // Source: https://huggingface.co/google/gemma-3-27b-it/raw/main/config.json (accessed 2026-07-16)
    ModelEntry {
        tokens: &["gemma3", "27b"],
        entry: catalog_dense! {
            param_count: 27 * B,
            active_param_count: None,
            num_layers: 62,
            hidden_dim: 5376,
            default_weight_dtype: "bf16",
            num_kv_heads: Some(16),
            head_dim: Some(128),
            num_kv_layers: None,
            attn_flops_coeff: None,
            swa_window: 1024,
            num_swa_layers: 52,
        },
    },
    // Source: https://huggingface.co/google/gemma-3-12b-it/raw/main/config.json (accessed 2026-07-16)
    ModelEntry {
        // https://huggingface.co/google/gemma-3-12b-it (text_config); head_dim=256
        // from Gemma 3 reference config (gm.nn.Gemma3_12B).
        tokens: &["gemma 3", "12b"],
        entry: catalog_dense! {
            param_count: 12 * B,
            active_param_count: None,
            num_layers: 48,
            hidden_dim: 3840,
            default_weight_dtype: "bf16",
            num_kv_heads: Some(8),
            head_dim: Some(256),
            num_kv_layers: None,
            attn_flops_coeff: None,
            swa_window: 1024,
            num_swa_layers: 40,
        },
    },
    // Source: https://huggingface.co/google/gemma-3-12b-it/raw/main/config.json (accessed 2026-07-16)
    ModelEntry {
        tokens: &["gemma3", "12b"],
        entry: catalog_dense! {
            param_count: 12 * B,
            active_param_count: None,
            num_layers: 48,
            hidden_dim: 3840,
            default_weight_dtype: "bf16",
            num_kv_heads: Some(8),
            head_dim: Some(256),
            num_kv_layers: None,
            attn_flops_coeff: None,
            swa_window: 1024,
            num_swa_layers: 40,
        },
    },
    // Source: https://huggingface.co/google/gemma-3-4b-it/raw/main/config.json (accessed 2026-07-16)
    ModelEntry {
        // https://huggingface.co/google/gemma-3-4b-it text_config (+ head/kv from
        // published Gemma 3 configs: num_attention_heads=8, num_key_value_heads=4,
        // head_dim=256).
        tokens: &["gemma 3", "4b"],
        entry: catalog_dense! {
            param_count: 4 * B,
            active_param_count: None,
            num_layers: 34,
            hidden_dim: 2560,
            default_weight_dtype: "bf16",
            num_kv_heads: Some(4),
            head_dim: Some(256),
            num_kv_layers: None,
            attn_flops_coeff: None,
            swa_window: 1024,
            num_swa_layers: 29,
        },
    },
    // Source: https://huggingface.co/google/gemma-3-4b-it/raw/main/config.json (accessed 2026-07-16)
    ModelEntry {
        tokens: &["gemma3", "4b"],
        entry: catalog_dense! {
            param_count: 4 * B,
            active_param_count: None,
            num_layers: 34,
            hidden_dim: 2560,
            default_weight_dtype: "bf16",
            num_kv_heads: Some(4),
            head_dim: Some(256),
            num_kv_layers: None,
            attn_flops_coeff: None,
            swa_window: 1024,
            num_swa_layers: 29,
        },
    },
    // Source: https://huggingface.co/google/gemma-3-1b-it/raw/main/config.json (gated).
    // sliding_window=512 (1B differs from the rest of the family); verified against
    // https://huggingface.co/unsloth/gemma-3-1b-it-GGUF/raw/main/config.json (2026-07-21).
    ModelEntry {
        // https://huggingface.co/google/gemma-3-1b-it/raw/main/config.json
        tokens: &["gemma 3", "1b"],
        entry: catalog_dense! {
            param_count: B,
            active_param_count: None,
            num_layers: 26,
            hidden_dim: 1152,
            default_weight_dtype: "bf16",
            num_kv_heads: Some(1),
            head_dim: Some(256),
            num_kv_layers: None,
            attn_flops_coeff: None,
            swa_window: 512,
            num_swa_layers: 22,
        },
    },
    // Source: https://huggingface.co/google/gemma-3-1b-it/raw/main/config.json (gated).
    // sliding_window=512 (1B differs from the rest of the family); verified against
    // https://huggingface.co/unsloth/gemma-3-1b-it-GGUF/raw/main/config.json (2026-07-21).
    ModelEntry {
        tokens: &["gemma3", "1b"],
        entry: catalog_dense! {
            param_count: B,
            active_param_count: None,
            num_layers: 26,
            hidden_dim: 1152,
            default_weight_dtype: "bf16",
            num_kv_heads: Some(1),
            head_dim: Some(256),
            num_kv_layers: None,
            attn_flops_coeff: None,
            swa_window: 512,
            num_swa_layers: 22,
        },
    },
    // Gemma 2 2B. Tokens "gemma 2"+"2b" (and fused "gemma2"+"2b"): bare
    // "gemma"+"2b" would also match Gemma 3 12B ("12b" contains "2b").
    // Official config.json is gated. Geometry matches HuggingFace
    // Gemma2Config defaults (the 2B layout) and the Unsloth copy.
    // Alternates local/global: 13 of 26 layers are windowed.
    // Source: https://huggingface.co/google/gemma-2-2b-it/raw/main/config.json (gated)
    // Verified: https://huggingface.co/unsloth/gemma-2-2b-it/raw/main/config.json (accessed 2026-08-13)
    // https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma2/configuration_gemma2.py
    ModelEntry {
        tokens: &["gemma 2", "2b"],
        entry: catalog_dense! {
            param_count: 2 * B,
            active_param_count: None,
            num_layers: 26,
            hidden_dim: 2304,
            default_weight_dtype: "bf16",
            num_kv_heads: Some(4),
            head_dim: Some(256),
            num_kv_layers: None,
            attn_flops_coeff: None,
            swa_window: 4096,
            num_swa_layers: 13,
        },
    },
    ModelEntry {
        tokens: &["gemma2", "2b"],
        entry: catalog_dense! {
            param_count: 2 * B,
            active_param_count: None,
            num_layers: 26,
            hidden_dim: 2304,
            default_weight_dtype: "bf16",
            num_kv_heads: Some(4),
            head_dim: Some(256),
            num_kv_layers: None,
            attn_flops_coeff: None,
            swa_window: 4096,
            num_swa_layers: 13,
        },
    },
    // Gemma 2 27B: head_dim=128, 32 attn heads, 16 KV heads.
    // sliding_window=4096 from config.json. Gemma 2 alternates local and global
    // attention, so 23 of 46 layers are windowed.
    // Source: https://huggingface.co/google/gemma-2-27b-it/raw/main/config.json (accessed 2026-07-16)
    // Alternation sources:
    // https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma2/modeling_gemma2.py
    // https://arxiv.org/html/2408.00118
    ModelEntry {
        tokens: &["gemma", "27b"],
        entry: catalog_dense! {
            param_count: 27 * B,
            active_param_count: None,
            num_layers: 46,
            hidden_dim: 4608,
            default_weight_dtype: "bf16",
            num_kv_heads: Some(16),
            head_dim: Some(128),
            num_kv_layers: None,
            attn_flops_coeff: None,
            swa_window: 4096,
            num_swa_layers: 23,
        },
    },
    // Gemma 2 9B: head_dim=256 (larger than typical), 16 attn heads, 8 KV heads.
    // sliding_window=4096 from config.json. Gemma 2 alternates local and global
    // attention, so 21 of 42 layers are windowed.
    // Source: https://huggingface.co/google/gemma-2-9b-it/raw/main/config.json (accessed 2026-07-16)
    // Alternation sources:
    // https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma2/modeling_gemma2.py
    // https://arxiv.org/html/2408.00118
    ModelEntry {
        tokens: &["gemma", "9b"],
        entry: catalog_dense! {
            param_count: 9 * B,
            active_param_count: None,
            num_layers: 42,
            hidden_dim: 3584,
            default_weight_dtype: "bf16",
            num_kv_heads: Some(8),
            head_dim: Some(256),
            num_kv_layers: None,
            attn_flops_coeff: None,
            swa_window: 4096,
            num_swa_layers: 21,
        },
    },
    // ── GLM ───────────────────────────────────────────────────────────────────
    // Source: https://huggingface.co/zai-org/GLM-4-32B-0414/raw/main/config.json (accessed 2026-07-16)
    ModelEntry {
        tokens: &["glm", "32b"],
        entry: catalog_dense! {
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
    // ── Phi-4 ────────────────────────────────────────────────────────────────
    // (Phi-4 32B phantom entry removed, no official release at this size)
    // Phi-4-mini 3.8B: 24 attn heads, 8 KV heads, head_dim=128 (3072/24).
    // Must precede ["phi","4"] so "Phi-4-mini-…" does not inherit 14B geometry.
    // Source: https://huggingface.co/microsoft/Phi-4-mini-instruct/raw/main/config.json
    // (accessed 2026-07-21)
    ModelEntry {
        tokens: &["phi", "4", "mini"],
        entry: catalog_dense! {
            param_count: 3_800_000_000,
            active_param_count: None,
            num_layers: 32,
            hidden_dim: 3072,
            default_weight_dtype: "bf16",
            num_kv_heads: Some(8),
            head_dim: Some(128),
            num_kv_layers: None,
            attn_flops_coeff: None,
        },
    },
    // Phi-4 14B: 40 attn heads, 10 KV heads (GQA 4:1), head_dim=128.
    // Source: https://huggingface.co/microsoft/phi-4/raw/main/config.json (accessed 2026-07-16)
    ModelEntry {
        tokens: &["phi", "4", "14b"],
        entry: catalog_dense! {
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
    // 675B-class weights cannot fit a catalog GPU even at 4-bit (~337 GB).
    // Without this, "Mistral-Large-3-675B" matches the 123B Mistral Large default.
    if norm.contains("675b") || norm.contains("mistral large 3") {
        return None;
    }
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
    use crate::context::ModelArch;
    use crate::engine::baseline::bytes_per_seq;

    fn model_arch(entry: &CatalogEntry) -> ModelArch {
        ModelArch {
            param_count: Some(entry.param_count),
            active_param_count: entry.active_param_count,
            num_layers: Some(entry.num_layers),
            hidden_dim: Some(entry.hidden_dim),
            default_weight_dtype: Some(entry.default_weight_dtype.to_string()),
            num_kv_heads: entry.num_kv_heads,
            head_dim: entry.head_dim,
            num_kv_layers: entry.num_kv_layers,
            attn_flops_coeff: entry.attn_flops_coeff,
            linear_num_layers: entry.linear_num_layers,
            linear_key_heads: entry.linear_key_heads,
            linear_value_heads: entry.linear_value_heads,
            linear_key_head_dim: entry.linear_key_head_dim,
            linear_value_head_dim: entry.linear_value_head_dim,
            linear_conv_kernel_dim: entry.linear_conv_kernel_dim,
            state_dtype: entry.state_dtype.map(str::to_string),
            swa_window: entry.swa_window,
            num_swa_layers: entry.num_swa_layers,
        }
    }

    #[test]
    fn launch_catalog_has_expected_entry_count() {
        assert_eq!(CATALOG.len(), 59);
    }

    #[test]
    fn models_too_large_for_supported_single_gpus_are_absent() {
        for name in [
            "moonshotai/Kimi-K2-Instruct",
            "thinkingmachines/Inkling",
            "zai-org/GLM-5.2",
            "zai-org/GLM-744B",
            "mistralai/Mistral-Large-3-675B-Instruct-2512",
            "deepseek-ai/DeepSeek-671B",
            "deepseek-ai/DeepSeek-R1",
            "deepseek-ai/DeepSeek-V3",
        ] {
            assert!(
                lookup_model(name).is_none(),
                "{name} must stay outside the catalog (4-bit weights exceed ~288 GB)"
            );
        }
    }

    #[test]
    fn phi4_14b() {
        let e = lookup_model("microsoft/phi-4").expect("no match");
        assert_eq!(e.param_count, 14 * B);
    }

    #[test]
    fn phi4_mini_instruct_resolves_to_3_8b() {
        let e = lookup_model("microsoft/Phi-4-mini-instruct").expect("no match");
        assert_eq!(e.param_count, 3_800_000_000);
        assert_eq!(e.num_layers, 32);
        assert_eq!(e.hidden_dim, 3072);
        assert_eq!(e.num_kv_heads, Some(8));
        assert_eq!(e.head_dim, Some(128));
        assert_eq!(e.default_weight_dtype, "bf16");
    }

    #[test]
    fn glm_32b() {
        let e = lookup_model("zai-org/GLM-4-32B-0414").expect("no match");
        assert_eq!(e.param_count, 32 * B);
        assert_eq!(e.num_layers, 61);
        assert_eq!(e.hidden_dim, 6144);
        assert_eq!(e.num_kv_heads, Some(2));
        assert_eq!(e.head_dim, Some(128));
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
    fn qwen38_27b_from_hf_config() {
        // Source: Qwen/Qwen3.8-27B text_config (2026-08-15). H100 demo serves BF16.
        let e = lookup_model("Qwen/Qwen3.8-27B").expect("no match");
        assert_eq!(e.param_count, 27_781_427_952);
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
        assert!(bytes_per_seq(&model_arch(e), 8192, 2).is_some());
        let fp8 = lookup_model("Qwen/Qwen3.8-27B-FP8").expect("fp8");
        assert_eq!(fp8.num_kv_layers, Some(16));
        let nvfp4 = lookup_model("Inferact/Qwen3.8-27B-NVFP4").expect("nvfp4");
        assert_eq!(nvfp4.param_count, 27_781_427_952);
        let qwen38 = lookup_model("Qwen/Qwen3.8-27B").expect("3.8");
        let qwen36 = lookup_model("Qwen/Qwen3.6-27B").expect("3.6");
        assert_eq!(qwen38.num_kv_layers, qwen36.num_kv_layers);
        let qwen3_8b = lookup_model("Qwen/Qwen3-8B").expect("qwen3 8b");
        assert_eq!(qwen3_8b.param_count, 8 * B);
        assert_eq!(qwen3_8b.num_layers, 36);
    }

    #[test]
    fn qwen36_27b() {
        let e = lookup_model("Qwen/Qwen3.6-27B").expect("no match");
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
        assert_eq!(e.param_count, 31 * B);
        assert_eq!(e.num_layers, 60);
    }

    #[test]
    fn gemma_4_31b() {
        let e = lookup_model("gemma-4-31b-it-bf16.gguf").expect("no match");
        assert_eq!(e.param_count, 31 * B);
        assert!(e.active_param_count.is_none());
        assert_eq!(bytes_per_seq(&model_arch(e), 8192, 2), None);
    }

    #[test]
    fn gemma_4_26b() {
        let e = lookup_model("gemma-4-26b-it-bf16.gguf").expect("no match");
        assert_eq!(e.param_count, 25_200_000_000);
        assert_eq!(e.active_param_count, Some(3_800_000_000));
    }

    #[test]
    fn gemma_4_26b_a4b_official_name() {
        let e = lookup_model("google/gemma-4-26B-A4B-it").expect("no match");
        assert_eq!(e.param_count, 25_200_000_000);
        assert_eq!(e.active_param_count, Some(3_800_000_000));
    }

    #[test]
    fn muse_glimmer_30b_from_hf_config() {
        // Source: meta-models/Muse-Glimmer-30B text_config (2026-08-10).
        let e = lookup_model("meta-models/Muse-Glimmer-30B").expect("no match");
        assert_eq!(e.param_count, 29_600_000_000);
        assert_eq!(e.active_param_count, Some(27_800_000_000)); // LM-only; ViT out of decode roof
        assert_eq!(e.num_layers, 52);
        assert_eq!(e.hidden_dim, 6656);
        assert_eq!(e.num_kv_heads, Some(2));
        assert_eq!(e.head_dim, Some(128));
        assert_eq!(e.swa_window, Some(2048));
        assert_eq!(e.num_swa_layers, Some(39)); // 3/4 of 52 (layer_types 3×SWA + 1×full)
        assert_eq!(e.default_weight_dtype, "bf16");
        // SWA + full layers must price; full_layers = 52 - 39 = 13.
        assert!(bytes_per_seq(&model_arch(e), 8192, 2).is_some());
        assert!(lookup_model("Muse-Glimmer-30B-GGUF").is_some());
        assert!(lookup_model("somevendor/muse-only-99B").is_none()); // needs all three tokens
    }

    #[test]
    fn unknown_model_returns_none() {
        assert!(lookup_model("somevendor/mystery-model-99B").is_none());
    }

    #[test]
    fn llama32_3b_and_1b() {
        let e3 = lookup_model("meta-llama/Llama-3.2-3B-Instruct").expect("3.2 3b");
        assert_eq!(e3.param_count, 3 * B);
        assert_eq!(e3.num_layers, 28);
        assert_eq!(e3.hidden_dim, 3072);
        assert_eq!(e3.num_kv_heads, Some(8));
        assert_eq!(e3.head_dim, Some(128));
        let e1 = lookup_model("meta-llama/Llama-3.2-1B-Instruct").expect("3.2 1b");
        assert_eq!(e1.param_count, B);
        assert_eq!(e1.num_layers, 16);
        assert_eq!(e1.hidden_dim, 2048);
        assert_eq!(e1.head_dim, Some(64));
        // 3.1 8B must not land on 3.2.
        let e8 = lookup_model("meta-llama/Llama-3.1-8B-Instruct").expect("3.1 8b");
        assert_eq!(e8.param_count, 8 * B);
        assert_eq!(e8.num_layers, 32);
    }

    #[test]
    fn qwen25_small_sizes() {
        let e05 = lookup_model("Qwen/Qwen2.5-0.5B-Instruct").expect("0.5b");
        assert_eq!(e05.param_count, 500_000_000);
        assert_eq!(e05.hidden_dim, 896);
        assert_eq!(e05.num_kv_heads, Some(2));
        assert_eq!(e05.head_dim, Some(64));
        assert!(e05.swa_window.is_none());
        let e15 = lookup_model("Qwen/Qwen2.5-1.5B-Instruct").expect("1.5b");
        assert_eq!(e15.param_count, 1_500_000_000);
        assert_eq!(e15.hidden_dim, 1536);
        let e3 = lookup_model("Qwen/Qwen2.5-3B-Instruct").expect("3b");
        assert_eq!(e3.param_count, 3 * B);
        assert_eq!(e3.num_layers, 36);
        assert_eq!(e3.hidden_dim, 2048);
        // 32B must not land on 3B.
        let e32 = lookup_model("Qwen/Qwen2.5-32B-Instruct").expect("32b");
        assert_eq!(e32.param_count, 32 * B);
    }

    #[test]
    fn gemma2_2b_not_12b_or_27b() {
        let e = lookup_model("google/gemma-2-2b-it").expect("gemma2 2b");
        assert_eq!(e.param_count, 2 * B);
        assert_eq!(e.num_layers, 26);
        assert_eq!(e.hidden_dim, 2304);
        assert_eq!(e.num_kv_heads, Some(4));
        assert_eq!(e.head_dim, Some(256));
        assert_eq!(e.swa_window, Some(4096));
        assert_eq!(e.num_swa_layers, Some(13));
        let fused = lookup_model("gemma2-2b").expect("fused");
        assert_eq!(fused.param_count, 2 * B);
        let g12 = lookup_model("google/gemma-3-12b-it").expect("gemma3 12b");
        assert_eq!(g12.param_count, 12 * B);
        let g27 = lookup_model("google/gemma-2-27b-it").expect("gemma2 27b");
        assert_eq!(g27.param_count, 27 * B);
    }

    #[test]
    fn llama4_untagged_70b_returns_none() {
        // A hypothetical "Llama-4-70B" (no Scout/Maverick) must NOT match as
        // llama3. Before the "3" token guard this would have returned llama3.
        assert!(lookup_model("meta-llama/Llama-4-70B-Instruct").is_none());
    }

    #[test]
    fn llama3_70b_variants() {
        for name in [
            "meta-llama/Llama-3.1-70B-Instruct",
            "meta-llama/Meta-Llama-3-70B",
            "llama-3.3-70b-instruct",
        ] {
            let e = lookup_model(name).unwrap_or_else(|| panic!("no match for {name}"));
            assert_eq!(e.param_count, 70 * B);
            assert!(e.active_param_count.is_none());
        }
    }

    #[test]
    fn llama4_maverick() {
        let e = lookup_model("meta-llama/Llama-4-Maverick-17B-128E-Instruct")
            .expect("should match maverick");
        assert_eq!(e.param_count, 400 * B);
        assert_eq!(e.active_param_count, Some(17 * B));
        assert_eq!(e.attn_flops_coeff, Some(0));
        assert!(e.num_kv_heads.is_none());
    }

    #[test]
    fn llama4_scout() {
        let e =
            lookup_model("meta-llama/Llama-4-Scout-17B-16E-Instruct").expect("should match scout");
        assert_eq!(e.param_count, 109 * B);
        assert_eq!(e.active_param_count, Some(17 * B));
    }

    #[test]
    fn llama3_405b() {
        let e = lookup_model("meta-llama/Meta-Llama-3.1-405B-Instruct").expect("no match");
        assert_eq!(e.param_count, 405 * B);
        assert_eq!(e.num_layers, 126);
    }

    #[test]
    fn qwen3_moe_235b() {
        let e = lookup_model("Qwen/Qwen3-235B-A22B").expect("should match qwen3 235b moe");
        assert_eq!(e.param_count, 235 * B);
        assert_eq!(e.active_param_count, Some(22 * B));
        assert_eq!(e.hidden_dim, 4096);
        assert_eq!(e.num_layers, 94);
        assert_eq!(e.num_kv_heads, Some(4));
        assert_eq!(e.head_dim, Some(128));
    }

    #[test]
    fn qwen25_72b() {
        let e = lookup_model("Qwen/Qwen2.5-72B-Instruct").expect("no match");
        assert_eq!(e.param_count, 72 * B);
        assert!(e.active_param_count.is_none());
    }

    #[test]
    fn mixtral_8x7b_not_mistral_7b() {
        let e = lookup_model("mistralai/Mixtral-8x7B-Instruct-v0.1").expect("no match");
        assert_eq!(e.param_count, 47 * B);
        assert_eq!(e.active_param_count, Some(13 * B));
    }

    #[test]
    fn mixtral_8x22b() {
        let e = lookup_model("mistralai/Mixtral-8x22B-Instruct-v0.1").expect("no match");
        assert_eq!(e.param_count, 141 * B);
        assert_eq!(e.active_param_count, Some(39 * B));
    }

    #[test]
    fn mistral_large_123b() {
        let e = lookup_model("mistralai/Mistral-Large-Instruct-2411").expect("no match");
        assert_eq!(e.param_count, 123 * B);
        assert_eq!(e.hidden_dim, 12288);
        assert!(e.active_param_count.is_none());
    }

    #[test]
    fn mistral_large_675b_does_not_match_123b() {
        assert!(lookup_model("mistralai/Mistral-Large-3-675B-Instruct-2512").is_none());
    }

    #[test]
    fn nemotron_70b() {
        for name in [
            "nvidia/Llama-3.1-Nemotron-70B-Instruct",
            "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
        ] {
            let e = lookup_model(name).unwrap_or_else(|| panic!("no match for {name}"));
            assert_eq!(e.param_count, 70 * B);
            assert_eq!(e.num_layers, 80);
            assert_eq!(e.hidden_dim, 8192);
            assert_eq!(e.num_kv_heads, Some(8));
            assert_eq!(e.head_dim, Some(128));
        }
    }

    #[test]
    fn r1_distill_llama_70b() {
        let e = lookup_model("deepseek-ai/DeepSeek-R1-Distill-Llama-70B").expect("no match");
        assert_eq!(e.param_count, 70 * B);
        assert_eq!(e.num_layers, 80);
        assert_eq!(e.num_kv_heads, Some(8));
    }

    #[test]
    fn deepseek_70b() {
        let e = lookup_model("deepseek-ai/deepseek-llm-70b-chat").expect("no match");
        assert_eq!(e.param_count, 70 * B);
        assert_eq!(e.attn_flops_coeff, Some(69_632));
    }

    #[test]
    fn llama_70b_does_not_match_llama4() {
        let e = lookup_model("meta-llama/Llama-3.1-70B-Instruct").expect("no match");
        assert_eq!(e.param_count, 70 * B);
        assert!(e.active_param_count.is_none());
        assert_eq!(e.attn_flops_coeff, None);
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
    fn r1_distill_qwen_1_5b() {
        let e = lookup_model("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B").expect("no match");
        assert_eq!(e.param_count, 1_500_000_000);
        assert_eq!(e.num_kv_heads, Some(2));
        assert_eq!(e.head_dim, Some(128));
    }

    #[test]
    fn gemma3_27b_not_gemma2() {
        let e = lookup_model("google/gemma-3-27b-it").expect("no match");
        assert_eq!(e.param_count, 27 * B);
        assert_eq!(e.num_layers, 62);
        assert_eq!(e.hidden_dim, 5376);
        assert_eq!(e.num_kv_heads, Some(16));
        assert_eq!(e.head_dim, Some(128));
        assert_eq!(e.swa_window, Some(1024));
        assert_eq!(e.num_swa_layers, Some(52));
    }

    #[test]
    fn gemma3_compact_name() {
        let e = lookup_model("gemma3-27b").expect("no match");
        assert_eq!(e.param_count, 27 * B);
        assert_eq!(e.num_layers, 62);
    }

    #[test]
    fn gemma2_quant_with_3bit_does_not_match_gemma3() {
        // Bare "3" token would false-match "3bit"; spaced "gemma 3" must not.
        let e = lookup_model("google/gemma-2-27b-3bit").expect("gemma2");
        assert_eq!(e.num_layers, 46);
        assert_eq!(e.hidden_dim, 4608);
    }

    #[test]
    fn gemma_27b_v3_does_not_match_gemma3() {
        // Must resolve to Gemma 2 27B (46 layers), not Gemma 3.
        let e = lookup_model("vendor/gemma-27b-v3").expect("gemma2");
        assert_eq!(e.num_layers, 46);
        assert_eq!(e.hidden_dim, 4608);
    }

    #[test]
    fn gemma3_4b_and_1b() {
        let e4 = lookup_model("google/gemma-3-4b-it").expect("4b");
        assert_eq!(e4.num_layers, 34);
        assert_eq!(e4.num_kv_heads, Some(4));
        assert_eq!(e4.swa_window, Some(1024));
        assert_eq!(e4.num_swa_layers, Some(29));
        let e1 = lookup_model("google/gemma-3-1b-it").expect("1b");
        assert_eq!(e1.num_layers, 26);
        assert_eq!(e1.num_kv_heads, Some(1));
        assert_eq!(e1.swa_window, Some(512));
        assert_eq!(e1.num_swa_layers, Some(22));
    }

    #[test]
    fn gemma3_window_split_is_pinned() {
        let e1 = lookup_model("google/gemma-3-1b-it").expect("1b");
        let e27 = lookup_model("google/gemma-3-27b-it").expect("27b");
        assert_eq!(e1.swa_window, Some(512));
        assert_eq!(e27.swa_window, Some(1024));
    }

    #[test]
    fn gemma4_still_guarded_after_gemma3() {
        let e = lookup_model("google/gemma-4-pt-27b-it").expect("gemma4");
        assert_eq!(e.param_count, 31 * B);
    }

    #[test]
    fn gemma2_27b_still_matches() {
        let e = lookup_model("google/gemma-2-27b-it").expect("gemma2");
        assert_eq!(e.num_layers, 46);
        assert_eq!(e.hidden_dim, 4608);
        assert_eq!(e.swa_window, Some(4096));
        assert_eq!(e.num_swa_layers, Some(23));
    }

    #[test]
    fn gemma2_9b_window_split_is_pinned() {
        let e = lookup_model("google/gemma-2-9b-it").expect("gemma2");
        assert_eq!(e.num_layers, 42);
        assert_eq!(e.swa_window, Some(4096));
        assert_eq!(e.num_swa_layers, Some(21));
    }

    #[test]
    fn qwen3_phantoms_7b_72b_gone() {
        assert!(lookup_model("Qwen/Qwen3-7B").is_none());
        assert!(lookup_model("Qwen/Qwen3-72B").is_none());
        // Explicit Qwen2.5 still works.
        let e = lookup_model("Qwen/Qwen2.5-7B").expect("qwen2.5");
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
        // Catalog fields alone are the gate for per-request memory pricing.
        // Previously both were None and KV math short-circuited.
        assert!(e.num_kv_heads.is_some() && e.head_dim.is_some());
    }
}
