//! `live-execution-server` — Sub-Millisecond Axum 0.8 Live Execution Server,
//! `POSITIONS OPENED` Active Monitor, `closed_trade_ledger` Archiver, and
//! `.CSV` / Multi-Sheet `.XLSX` Workbook Exporter.
//!
//! Full server routes, ledger state engine, and workbook exporter are expanded in Phase 3.

use raw_fingerprint_core::{BrainHorizonId, RawFingerprintEngine};

/// Returns the active server version and linked `RawFingerprintEngine` readiness.
pub fn server_engine_health() -> (&'static str, usize) {
    let _engine = RawFingerprintEngine::new();
    let brains = [
        BrainHorizonId::Brain01PennyAnomaly,
        BrainHorizonId::Brain02IntradayDayTrade,
        BrainHorizonId::Brain03ShortTermSwing,
        BrainHorizonId::Brain04CoreSwing,
        BrainHorizonId::Brain05LongTermWealth,
    ];
    ("argon-live-execution-server-v0.1.0", brains.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_server_engine_link() {
        let (version, brain_count) = server_engine_health();
        assert_eq!(version, "argon-live-execution-server-v0.1.0");
        assert_eq!(brain_count, 5);
    }
}
