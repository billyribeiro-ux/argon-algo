use live_execution_server::proof_harness::fetch_all_proof_datasets;
use live_execution_server::swing_brain_sealed::run_sealed_swing_standalone;
use market_data_bus::yahoo::YahooFinanceLiveAdapter;

#[tokio::test]
async fn test_swing_eval_baseline() {
    let adapter = YahooFinanceLiveAdapter::new();
    let ds = fetch_all_proof_datasets(&adapter).await;
    println!("Loaded {} h1 symbols", ds.h1_bars.len());
    let t0 = std::time::Instant::now();
    let report = run_sealed_swing_standalone(&ds, 100_000.0);
    println!("Sealed swing executed in {:.2}s", t0.elapsed().as_secs_f64());
    
    let b3 = &report.brain03_run_b;
    println!(
        "BRAIN-03 | Trades={} | Win%={:.1}% | Gross PF={:.2} | Net PF={:.2} | Net PnL=${:+.2} | Max DD={:.2}%",
        b3.trade_count, b3.win_rate_pct, b3.gross_profit_factor, b3.net_profit_factor, b3.total_net_pnl_usd, b3.max_drawdown_pct
    );

    let b4 = &report.brain04_run_b;
    println!(
        "BRAIN-04 | Trades={} | Win%={:.1}% | Gross PF={:.2} | Net PF={:.2} | Net PnL=${:+.2} | Max DD={:.2}%",
        b4.trade_count, b4.win_rate_pct, b4.gross_profit_factor, b4.net_profit_factor, b4.total_net_pnl_usd, b4.max_drawdown_pct
    );

    let tier = &report.tier_swing_run_b;
    println!(
        "TIER: SWING | Trades={} | Win%={:.1}% | Gross PF={:.2} | Net PF={:.2} | Net PnL=${:+.2} | Max DD={:.2}%",
        tier.trade_count, tier.win_rate_pct, tier.gross_profit_factor, tier.net_profit_factor, tier.total_net_pnl_usd, tier.max_drawdown_pct
    );
}
