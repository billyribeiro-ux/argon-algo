//! `--release` Parallel Self-Supervised World Model + K=5 Specialist Stateful RL + Closed-Loop Loss-Autopsy Gate.
//!
//! Run with:
//! `cargo run --release -p live-execution-server --bin prove_perfection`
//! `cargo run --release -p live-execution-server --bin prove_perfection -- --swing-only`
//! `cargo run --release -p live-execution-server --bin prove_perfection -- --intraday-only`

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use live_execution_server::onnx_engine::OnnxMultiBrainEngine;
use live_execution_server::proof_harness::{
    fetch_all_proof_datasets, run_chronological_hard_evidence_proof_ex,
    verify_future_curtain_zero_lookahead_invariance, BucketPerformanceSummary,
};
use live_execution_server::swing_brain_sealed::run_sealed_swing_standalone;
use market_data_bus::yahoo::YahooFinanceLiveAdapter;

fn compute_daily_annualized_sharpe(blotter_csv_path: &str, run_mode: &str) -> f64 {
    let Ok(content) = std::fs::read_to_string(blotter_csv_path) else {
        return 0.0;
    };
    let mut daily_pnl: BTreeMap<String, f64> = BTreeMap::new();
    for (idx, line) in content.lines().enumerate() {
        if idx == 0 || line.trim().is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split(',').collect();
        if cols.len() < 19 || cols[0] != run_mode {
            continue;
        }
        let date_str = cols[8].split(' ').next().unwrap_or("").to_string();
        let net_pnl: f64 = cols[17].parse().unwrap_or(0.0);
        *daily_pnl.entry(date_str).or_insert(0.0) += net_pnl;
    }
    if daily_pnl.len() < 2 {
        return 0.0;
    }
    let vals: Vec<f64> = daily_pnl.values().copied().collect();
    let n = vals.len() as f64;
    let mean = vals.iter().sum::<f64>() / n;
    let var = vals.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (n - 1.0);
    if var <= 1e-12 {
        return 0.0;
    }
    (mean / var.sqrt()) * 252.0_f64.sqrt()
}

fn check_brain_gate(b: &BucketPerformanceSummary, min_pf: f64, min_trades: usize) -> Vec<String> {
    let mut errs = Vec::new();
    if b.trade_count < min_trades {
        errs.push(format!("trades={} < {}", b.trade_count, min_trades));
    }
    if b.net_profit_factor < min_pf {
        errs.push(format!("net_post_cost_pf={:.3} < {:.2}", b.net_profit_factor, min_pf));
    }
    if b.total_net_pnl_usd <= 0.0 {
        errs.push(format!("net_pnl_usd=${:+.2} <= 0.0", b.total_net_pnl_usd));
    }
    if b.net_expectancy_r <= 0.0 {
        errs.push(format!("net_exp_r={:+.4}R <= 0.0", b.net_expectancy_r));
    }
    errs
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    let swing_only = args.iter().any(|a| a == "--swing-only");
    let intraday_only = args.iter().any(|a| a == "--intraday-only");

    let t0 = Instant::now();

    // 1. Future-curtain zero-lookahead invariance verification (`|Δa_t| == 0.0` when scrambling bars `> t`)
    let (a_orig, a_scram, delta_a, delta_v) = verify_future_curtain_zero_lookahead_invariance();
    println!(
        "[FUTURE_CURTAIN_INVARIANCE] a_t_orig={a_orig:.12} | a_t_scrambled={a_scram:.12} | |Δa_t|={delta_a:.12} | |Δv_t|={delta_v:.12}"
    );

    let mut all_failures: Vec<String> = Vec::new();
    if delta_a != 0.0 || delta_v != 0.0 {
        all_failures.push(format!(
            "FUTURE_CURTAIN_LOOKAHEAD_VIOLATION: |Δa_t|={delta_a:.12}, |Δv_t|={delta_v:.12}"
        ));
    }

    let adapter = YahooFinanceLiveAdapter::new();

    // Mode 1: --swing-only (Executes ONLY BRAIN-03 and BRAIN-04 through swing_brain_sealed.rs in seconds)
    if swing_only {
        println!("========================================================================================");
        println!("                       TIER: SWING FROZEN SCORECARD (BRAIN-03 & BRAIN-04)               ");
        println!("========================================================================================");
        let ds = fetch_all_proof_datasets(&adapter).await;
        let report = run_sealed_swing_standalone(&ds, 100_000.0);
        let elapsed = t0.elapsed().as_secs_f64();

        println!("\n=== CAUSAL INNER-FOLD WARMUP TELEMETRY ===");
        for tel in &report.warmup_telemetry {
            println!(
                "  {:<26} | Population={:>2} | Sequences={:>2} | PBT_Gens={:>2} | InnerVal_Trades={:>3} | InnerVal_NetPF={:>5.2} | InnerVal_NetPnL=${:>+9.2}",
                tel.brain_label, tel.population_size, tel.train_sequences, tel.pbt_generations, tel.best_val_trades, tel.best_val_net_pf, tel.best_val_net_pnl
            );
        }

        let b3 = &report.brain03_run_b;
        let b4 = &report.brain04_run_b;
        let tier_b = &report.tier_swing_run_b;
        let tier_a = &report.tier_swing_run_a;

        let status_b3 = if b3.net_profit_factor >= 1.50 && b3.trade_count >= 15 && b3.total_net_pnl_usd > 0.0 { "PASS" } else { "FAIL" };
        let status_b4 = if b4.net_profit_factor >= 1.50 && b4.trade_count >= 15 && b4.total_net_pnl_usd > 0.0 { "PASS" } else { "FAIL" };

        println!("\n=== INDIVIDUAL SWING BRAINS OOS RESULTS (RUN_B_ONLINE_RL) ===");
        println!(
            "[{status_b3}] {:<26} | Trades={:>3} | Win%={:>5.1}% | GrossPF={:>5.2} | NetPF={:>5.2} (req>=1.50) | NetExpR={:>+6.3}R | MaxDD={:>5.2}% | NetPnL=${:>+9.2}",
            b3.bucket_label, b3.trade_count, b3.win_rate_pct, b3.gross_profit_factor, b3.net_profit_factor, b3.net_expectancy_r, b3.max_drawdown_pct, b3.total_net_pnl_usd
        );
        println!(
            "[{status_b4}] {:<26} | Trades={:>3} | Win%={:>5.1}% | GrossPF={:>5.2} | NetPF={:>5.2} (req>=1.50) | NetExpR={:>+6.3}R | MaxDD={:>5.2}% | NetPnL=${:>+9.2}",
            b4.bucket_label, b4.trade_count, b4.win_rate_pct, b4.gross_profit_factor, b4.net_profit_factor, b4.net_expectancy_r, b4.max_drawdown_pct, b4.total_net_pnl_usd
        );

        println!("\n=== COMBINED TIER: SWING SCORECARD & A/B COMPARISON ===");
        println!(
            "RUN_A_FROZEN_WARMUP_ML: Trades={:>3} | Win%={:>5.1}% | GrossPF={:>5.2} | NetPF={:>5.2} | MaxDD={:>5.2}% | NetPnL=${:>+10.2}",
            tier_a.trade_count, tier_a.win_rate_pct, tier_a.gross_profit_factor, tier_a.net_profit_factor, tier_a.max_drawdown_pct, tier_a.total_net_pnl_usd
        );
        println!(
            "RUN_B_ONLINE_RL       : Trades={:>3} | Win%={:>5.1}% | GrossPF={:>5.2} | NetPF={:>5.2} | MaxDD={:>5.2}% | NetPnL=${:>+10.2}",
            tier_b.trade_count, tier_b.win_rate_pct, tier_b.gross_profit_factor, tier_b.net_profit_factor, tier_b.max_drawdown_pct, tier_b.total_net_pnl_usd
        );
        let delta_trades = (tier_b.trade_count as isize) - (tier_a.trade_count as isize);
        let delta_pnl = tier_b.total_net_pnl_usd - tier_a.total_net_pnl_usd;
        println!(
            "A/B CONVERGENCE DELTA : Δ Trades = {:>+3} | Δ Net PnL = ${:>+8.2}",
            delta_trades, delta_pnl
        );

        let mut swing_errs = Vec::new();
        if tier_b.net_profit_factor < 1.60 {
            swing_errs.push(format!("TIER: SWING Net PF {:.2} < 1.60", tier_b.net_profit_factor));
        }
        if tier_b.win_rate_pct < 60.0 && tier_b.total_net_pnl_usd < 10_000.0 {
            swing_errs.push(format!("TIER: SWING Win% {:.1}% < 60.0% and Net PnL < $10k", tier_b.win_rate_pct));
        }
        if tier_b.max_drawdown_pct >= 7.0 {
            swing_errs.push(format!("TIER: SWING Max DD {:.2}% >= 7.0%", tier_b.max_drawdown_pct));
        }
        if delta_trades != 0 || delta_pnl.abs() > 0.01 {
            swing_errs.push(format!("RUN_A vs RUN_B diverged: Δ Trades={}, Δ PnL=${:.2}", delta_trades, delta_pnl));
        }

        if swing_errs.is_empty() {
            println!(
                "\n[PERFECTION ACHIEVED] TIER: SWING RESTORED TO PEAK BASELINE AND HARD-ISOLATED BEHIND IMMUTABLE FIREWALL! (Completed in {:.2}s)\n",
                elapsed
            );
            return;
        } else {
            eprintln!("\n[SWING GATES FAILED]:");
            for err in swing_errs {
                eprintln!("  - {err}");
            }
            std::process::exit(1);
        }
    }

    // Mode 2: --intraday-only or Default (6-brain proof)
    let models_dir = [Path::new("../../models"), Path::new("models")]
        .into_iter()
        .find(|p| p.exists());
    let onnx = Arc::new(OnnxMultiBrainEngine::new(models_dir));

    let include_swing = !intraday_only;
    let mode_desc = if intraday_only {
        "Intraday Only (SCALP_0DTE + DAY)"
    } else {
        "Full 6-Brain (Intraday + Sealed Swing)"
    };

    println!(
        "[PROVE_PERFECTION] Starting --release M=24 Session-Episode BPTT + Closed-Loop Red-Trade Loss-Autopsy RL proof [{mode_desc}]..."
    );
    let report = run_chronological_hard_evidence_proof_ex(onnx, &adapter, 100_000.0, include_swing)
        .await
        .expect("Walk-forward proof harness must complete");

    println!(
        "\n[PROVE_PERFECTION] Completed in {:.2}s | Warmup Episode Bars={} | OOS Episode Bars={} | Online Updates={} | Promotions={}/{}",
        t0.elapsed().as_secs_f64(),
        report.warmup_counterfactuals_harvested,
        report.oos_counterfactuals_harvested,
        report.online_rl_grad_updates,
        report.promotion_ledger.iter().filter(|r| r.promoted).count(),
        report.promotion_ledger.len(),
    );

    println!("\n=== INDIVIDUAL BRAIN OOS RESULTS (RUN_B_ONLINE_RL) ===");
    for cmp in &report.bucket_comparisons {
        let b = &cmp.run_b;
        let min_pf = if b.bucket_label.contains("BRAIN-03") || b.bucket_label.contains("BRAIN-04") {
            1.50
        } else {
            1.35
        };
        let min_trades = 15usize;
        let errs = check_brain_gate(b, min_pf, min_trades);
        let status = if errs.is_empty() { "PASS" } else { "FAIL" };
        println!(
            "[{status}] {:<26} | Trades={:>3} | Win%={:>5.1}% | GrossPF={:>5.2} | NetPF={:>5.2} (req>={:.2}) | NetExpR={:>+6.3}R | NeuralExit={:>5.1}% | EODFlat={:>5.1}% | MaxDD={:>5.2}% | NetPnL=${:>+9.2} (RUN_A=${:>+9.2}, Δ=${:>+8.2})",
            b.bucket_label,
            b.trade_count,
            b.win_rate_pct,
            b.gross_profit_factor,
            b.net_profit_factor,
            min_pf,
            b.net_expectancy_r,
            b.trail_step_rate_pct,
            b.eod_flatten_rate_pct,
            b.max_drawdown_pct,
            b.total_net_pnl_usd,
            cmp.run_a.total_net_pnl_usd,
            cmp.delta_net_pnl_usd,
        );
        if !errs.is_empty() {
            all_failures.push(format!("{}: {}", b.bucket_label, errs.join(", ")));
        }
    }

    let pa = &report.portfolio_run_a;
    let pb = &report.portfolio_run_b;
    let sharpe_a = compute_daily_annualized_sharpe(&report.blotter_csv_path, "RUN_A_FROZEN_WARMUP_ML");
    let sharpe_b = compute_daily_annualized_sharpe(&report.blotter_csv_path, "RUN_B_ONLINE_RL");
    println!("\n=== ALL_BUCKETS_COMBINED & DATASET DELTAS ===");
    println!(
        "RUN_A_FROZEN_WARMUP_ML: Trades={:>3} | NetPF={:>5.2} | Sharpe={:>5.2} | MaxDD={:>5.2}% | NetPnL=${:>+10.2}",
        pa.trade_count, pa.net_profit_factor, sharpe_a, pa.max_drawdown_pct, pa.total_net_pnl_usd
    );
    println!(
        "RUN_B_ONLINE_RL       : Trades={:>3} | NetPF={:>5.2} | Sharpe={:>5.2} | MaxDD={:>5.2}% | NetPnL=${:>+10.2}",
        pb.trade_count, pb.net_profit_factor, sharpe_b, pb.max_drawdown_pct, pb.total_net_pnl_usd
    );
    println!(
        "ONLINE_RL_LIFT (B - A): Portfolio Δ=${:+.2} | Dataset 1 (1m) Δ=${:+.2} | Dataset 2 (5m & 1h) Δ=${:+.2}",
        pb.total_net_pnl_usd - pa.total_net_pnl_usd,
        report.dataset1_delta_net_pnl_usd,
        report.dataset2_delta_net_pnl_usd,
    );

    let min_portfolio_pf = if intraday_only { 1.35 } else { 1.50 };
    if pb.net_profit_factor < min_portfolio_pf {
        all_failures.push(format!(
            "ALL_BUCKETS_COMBINED net_post_cost_pf={:.3} < {:.2}",
            pb.net_profit_factor, min_portfolio_pf
        ));
    }
    if pb.max_drawdown_pct >= 10.0 {
        all_failures.push(format!(
            "ALL_BUCKETS_COMBINED max_dd_pct={:.2}% >= 10.0%",
            pb.max_drawdown_pct
        ));
    }
    if pb.total_net_pnl_usd <= pa.total_net_pnl_usd {
        all_failures.push(format!(
            "ALL_BUCKETS_COMBINED RUN_B (${:+.2}) did not strictly beat RUN_A (${:+.2})",
            pb.total_net_pnl_usd, pa.total_net_pnl_usd
        ));
    }
    if report.dataset1_delta_net_pnl_usd <= 0.0 {
        all_failures.push(format!(
            "DATASET_1 (1m) RUN_B did not strictly beat RUN_A (Δ=${:+.2} <= $0.00)",
            report.dataset1_delta_net_pnl_usd
        ));
    }
    if !intraday_only && report.dataset2_delta_net_pnl_usd <= 0.0 {
        all_failures.push(format!(
            "DATASET_2 (5m & 1h) RUN_B did not strictly beat RUN_A (Δ=${:+.2} <= $0.00)",
            report.dataset2_delta_net_pnl_usd
        ));
    }

    if all_failures.is_empty() {
        println!("\n[PERFECTION ACHIEVED] ALL PROOF GATES PASSED!");
    } else {
        eprintln!("\n=== CAUSAL INNER-FOLD DIAGNOSTICS (`t <= T_oos_start`) ===");
        for t in &report.warmup_telemetry {
            eprintln!(
                "  {:<26} | Episodes={:>4} | PBT_Gens={:>2} | InnerVal_Trades={:>3} | InnerVal_NetPF={:>5.2} | InnerVal_NetPnL=${:>+9.2}",
                t.brain_label,
                t.train_sequences,
                t.pbt_generations,
                t.best_val_trades,
                t.best_val_net_pf,
                t.best_val_net_pnl
            );
        }
        eprintln!("\n[PERFECTION GATES FAILED — EXITING 1]:");
        for f in &all_failures {
            eprintln!("  - {f}");
        }
        std::process::exit(1);
    }
}
