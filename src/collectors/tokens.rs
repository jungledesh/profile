//! Token stats collector.
//!
//! Will be backed by vLLM (or remote profile-agent) later.

/// Token throughput stats.
#[derive(Debug, Clone)]
pub struct TokenStats {
    pub tokens_per_sec: f32,
}

/// Read current token stats. Returns None if unavailable.
pub fn token_stats() -> Option<TokenStats> {
    // Stub: no vLLM integration yet.
    None
}
