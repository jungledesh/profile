use std::collections::HashMap;
use std::sync::OnceLock;

use super::gpu_catalog;

const GPU_PRICES_JSON: &str = include_str!("gpu_prices.json");

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuPriceEntry {
    pub on_demand_per_hr: f64,
    pub spot_per_hr: f64,
}

static PRICES: OnceLock<HashMap<String, GpuPriceEntry>> = OnceLock::new();

fn load_prices() -> &'static HashMap<String, GpuPriceEntry> {
    PRICES.get_or_init(|| {
        let root: serde_json::Value =
            serde_json::from_str(GPU_PRICES_JSON).expect("gpu_prices.json must be valid JSON");
        let gpus = root
            .get("gpus")
            .and_then(|v| v.as_object())
            .expect("gpu_prices.json must have a gpus object");
        let mut out = HashMap::new();
        for (key, entry) in gpus {
            let on = entry
                .get("on_demand_per_hr")
                .and_then(|v| v.as_f64())
                .filter(|v| v.is_finite() && *v > 0.0)
                .expect("gpu_prices entry missing on_demand_per_hr");
            let spot = entry
                .get("spot_per_hr")
                .and_then(|v| v.as_f64())
                .filter(|v| v.is_finite() && *v > 0.0)
                .expect("gpu_prices entry missing spot_per_hr");
            out.insert(
                key.clone(),
                GpuPriceEntry {
                    on_demand_per_hr: on,
                    spot_per_hr: spot,
                },
            );
        }
        out
    })
}

/// Looks up price entry by GPU name tokens (loose substring match on catalog keys).
/// Returns None if GPU not in catalog — caller must handle gracefully.
pub fn lookup_gpu_price(gpu_name: &str) -> Option<GpuPriceEntry> {
    let norm = gpu_catalog::normalize_gpu_name(gpu_name);
    let prices = load_prices();
    let mut keys: Vec<&String> = prices.keys().collect();
    keys.sort_by_key(|k| std::cmp::Reverse(k.len()));
    keys.into_iter().find_map(|key| {
        if norm.contains(key.as_str()) {
            prices.get(key.as_str()).copied()
        } else {
            None
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn h200_matches_sxm_name() {
        let p = lookup_gpu_price("NVIDIA H200 SXM5").expect("h200 price");
        assert!((p.on_demand_per_hr - 3.50).abs() < 1e-9);
    }

    #[test]
    fn gb200_does_not_match_b200_price() {
        let p = lookup_gpu_price("NVIDIA GB200").expect("gb200 price");
        assert!((p.on_demand_per_hr - 8.50).abs() < 1e-9);
    }

    #[test]
    fn unknown_gpu_returns_none() {
        assert!(lookup_gpu_price("NVIDIA Tesla V100").is_none());
    }
}
