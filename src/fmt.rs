/// Milliseconds for operator-facing latency lines (uses `ms` below 1s, `s` at or above).
pub fn fmt_seconds_from_ms(ms: f64) -> String {
    if ms >= 1000.0 {
        format!("{:.1}s", ms / 1000.0)
    } else {
        format!("{:.0}ms", ms)
    }
}

/// Like [`fmt_seconds_from_ms`], but prefixes `>= ` when the quantile was clamped to a floor.
pub fn fmt_seconds_from_ms_maybe_floor(ms: f64, clamped: bool) -> String {
    let body = fmt_seconds_from_ms(ms);
    if clamped { format!(">= {body}") } else { body }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fmt_seconds_from_ms_prefers_seconds_when_large() {
        assert_eq!(fmt_seconds_from_ms(1200.0), "1.2s");
        assert_eq!(fmt_seconds_from_ms(50.0), "50ms");
    }

    #[test]
    fn fmt_seconds_from_ms_maybe_floor_marks_clamped() {
        assert_eq!(fmt_seconds_from_ms_maybe_floor(40_000.0, true), ">= 40.0s");
        assert_eq!(fmt_seconds_from_ms_maybe_floor(40_000.0, false), "40.0s");
        assert_eq!(fmt_seconds_from_ms_maybe_floor(50.0, true), ">= 50ms");
        assert_eq!(fmt_seconds_from_ms_maybe_floor(50.0, false), "50ms");
    }
}
