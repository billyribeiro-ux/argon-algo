//! Cryptographic Source Lock & Deterministic Behavioral Parity Verification for `TIER: SWING`.
//!
//! Under SURGICAL DIRECTIVE: RESTORE `TIER: SWING` BASELINE & HARD-ISOLATE IT BEHIND AN IMMUTABLE FIREWALL,
//! this test computes the runtime SHA-256 digest of `crates/live-execution-server/src/swing_brain_sealed.rs`
//! and asserts that:
//! 1. The SHA-256 matches `SWING_BRAIN_SEALED_SHA256` exactly. If modified, it panics with
//!    `SWING_FIREWALL_BREACH: swing_brain_sealed.rs has been modified without authorization! Intraday refactoring is prohibited from altering Swing policy logic.`
//! 2. `swing_brain_sealed.rs` contains zero imports or dependencies on `pure_rl_brain.rs`.
//! 3. `TIER: SWING` Net Post-Cost PF >= 1.60.
//! 4. `TIER: SWING` Win% >= 60.0% (or Net PnL >= +$10,000).
//! 5. `TIER: SWING` Max DD < 7.0%.
//! 6. Zero divergence between `RUN_A` and `RUN_B` (`Δ Trades = 0`, `Δ Net PnL = $0.00`).

use std::path::Path;
use live_execution_server::proof_harness::fetch_all_proof_datasets;
use live_execution_server::swing_brain_sealed::run_sealed_swing_standalone;
use market_data_bus::yahoo::YahooFinanceLiveAdapter;

/// The immutable SHA-256 digest of `crates/live-execution-server/src/swing_brain_sealed.rs`
/// locked at 15:38Z baseline parity.
pub const SWING_BRAIN_SEALED_SHA256: &str =
    "42ce814235492a576ded5daceabd637bd9c8727f13e6389bf6d33afdf142a876";

/// Standard FIPS 180-4 SHA-256 implementation.
fn compute_sha256_hex(bytes: &[u8]) -> String {
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    ];
    let k: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ];

    let bit_len = (bytes.len() as u64) * 8;
    let mut msg = bytes.to_vec();
    msg.push(0x80);
    while (msg.len() % 64) != 56 {
        msg.push(0x00);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());

    for chunk in msg.as_chunks::<64>().0 {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                chunk[i * 4],
                chunk[i * 4 + 1],
                chunk[i * 4 + 2],
                chunk[i * 4 + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }

        let mut a = h[0];
        let mut b = h[1];
        let mut c = h[2];
        let mut d = h[3];
        let mut e = h[4];
        let mut f = h[5];
        let mut g = h[6];
        let mut h_var = h[7];

        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = h_var
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(k[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            h_var = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(h_var);
    }

    format!(
        "{:08x}{:08x}{:08x}{:08x}{:08x}{:08x}{:08x}{:08x}",
        h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]
    )
}

#[tokio::test]
async fn test_swing_firewall_cryptographic_and_behavioral_lock() {
    // 1. Locate `crates/live-execution-server/src/swing_brain_sealed.rs`
    let paths = [
        Path::new("src/swing_brain_sealed.rs"),
        Path::new("crates/live-execution-server/src/swing_brain_sealed.rs"),
        Path::new("../../crates/live-execution-server/src/swing_brain_sealed.rs"),
    ];
    let file_path = paths
        .iter()
        .find(|p| p.exists())
        .expect("swing_brain_sealed.rs must exist in workspace");

    let content_bytes = std::fs::read(file_path)
        .expect("Must be able to read swing_brain_sealed.rs");
    let content_str = std::str::from_utf8(&content_bytes)
        .expect("swing_brain_sealed.rs must be valid UTF-8");

    // 2. Assert Zero Shared Policy Code: ZERO imports from `pure_rl_brain`
    for line in content_str.lines() {
        let trimmed = line.trim();
        if (trimmed.starts_with("use ") || trimmed.starts_with("pub use "))
            && trimmed.contains("pure_rl_brain")
        {
            panic!(
                "SWING_FIREWALL_BREACH: swing_brain_sealed.rs imports from pure_rl_brain! Offending line: {}",
                line
            );
        }
    }

    // 3. Cryptographic SHA-256 Digest Verification
    let computed_sha256 = compute_sha256_hex(&content_bytes);
    println!("[SWING_FIREWALL] Computed SHA-256: {}", computed_sha256);
    println!("[SWING_FIREWALL] Expected SHA-256: {}", SWING_BRAIN_SEALED_SHA256);

    if computed_sha256 != SWING_BRAIN_SEALED_SHA256 {
        panic!(
            "SWING_FIREWALL_BREACH: swing_brain_sealed.rs has been modified without authorization! Intraday refactoring is prohibited from altering Swing policy logic. Expected SHA-256: {}, Got: {}",
            SWING_BRAIN_SEALED_SHA256,
            computed_sha256
        );
    }

    // 4. Behavioral Parity Verification
    let adapter = YahooFinanceLiveAdapter::new();
    let ds = fetch_all_proof_datasets(&adapter).await;

    let t0 = std::time::Instant::now();
    let report = run_sealed_swing_standalone(&ds, 100_000.0);
    println!(
        "[SWING_FIREWALL] Standalone evaluation completed in {:.3}s",
        t0.elapsed().as_secs_f64()
    );

    let b3 = &report.brain03_run_b;
    let b4 = &report.brain04_run_b;
    let tier_b = &report.tier_swing_run_b;
    let tier_a = &report.tier_swing_run_a;

    println!("\n=== SEALED SWING EVALUATION REPORT ===");
    println!(
        "BRAIN-03 | Trades={:>3} | Win%={:>5.1}% | GrossPF={:>5.2} | NetPF={:>5.2} | NetPnL=${:>+9.2} | MaxDD={:>5.2}%",
        b3.trade_count, b3.win_rate_pct, b3.gross_profit_factor, b3.net_profit_factor, b3.total_net_pnl_usd, b3.max_drawdown_pct
    );
    println!(
        "BRAIN-04 | Trades={:>3} | Win%={:>5.1}% | GrossPF={:>5.2} | NetPF={:>5.2} | NetPnL=${:>+9.2} | MaxDD={:>5.2}%",
        b4.trade_count, b4.win_rate_pct, b4.gross_profit_factor, b4.net_profit_factor, b4.total_net_pnl_usd, b4.max_drawdown_pct
    );
    println!(
        "TIER_SWING_A | Trades={:>3} | Win%={:>5.1}% | GrossPF={:>5.2} | NetPF={:>5.2} | NetPnL=${:>+9.2} | MaxDD={:>5.2}%",
        tier_a.trade_count, tier_a.win_rate_pct, tier_a.gross_profit_factor, tier_a.net_profit_factor, tier_a.total_net_pnl_usd, tier_a.max_drawdown_pct
    );
    println!(
        "TIER_SWING_B | Trades={:>3} | Win%={:>5.1}% | GrossPF={:>5.2} | NetPF={:>5.2} | NetPnL=${:>+9.2} | MaxDD={:>5.2}%",
        tier_b.trade_count, tier_b.win_rate_pct, tier_b.gross_profit_factor, tier_b.net_profit_factor, tier_b.total_net_pnl_usd, tier_b.max_drawdown_pct
    );

    // Assert TIER: SWING Acceptance Gates
    assert!(
        tier_b.net_profit_factor >= 1.60,
        "TIER: SWING Net Post-Cost PF ({:.2}) must be >= 1.60",
        tier_b.net_profit_factor
    );
    assert!(
        tier_b.win_rate_pct >= 60.0 || tier_b.total_net_pnl_usd >= 10_000.0,
        "TIER: SWING Win% ({:.1}%) must be >= 60.0% (or Net PnL >= +$10,000)",
        tier_b.win_rate_pct
    );
    assert!(
        tier_b.max_drawdown_pct < 7.0,
        "TIER: SWING Max DD ({:.2}%) must be < 7.0%",
        tier_b.max_drawdown_pct
    );

    // Assert Zero Divergence between RUN_A and RUN_B
    assert_eq!(
        tier_a.trade_count, tier_b.trade_count,
        "Zero delta required for trade count: RUN_A ({}) != RUN_B ({})",
        tier_a.trade_count, tier_b.trade_count
    );
    let pnl_diff = (tier_b.total_net_pnl_usd - tier_a.total_net_pnl_usd).abs();
    assert!(
        pnl_diff < 0.01,
        "Zero delta required for Net PnL: RUN_A (${:.2}) != RUN_B (${:.2})",
        tier_a.total_net_pnl_usd, tier_b.total_net_pnl_usd
    );

    println!("\n[SWING_FIREWALL_LOCK] ALL CRYPTOGRAPHIC & BEHAVIORAL INVARIANTS PASSED!");
}
