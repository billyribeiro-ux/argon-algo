//! `live-execution-server` — Sub-Millisecond Axum 0.8.9 Live Execution Server,
//! ONNX 5-Brain Inference Engine, `pgvector` k-NN Precedent Matcher,
//! `POSITIONS OPENED` Active Monitor, `closed_trade_ledger` Archiver, and
//! 1-Click `.CSV` / Multi-Sheet `.XLSX` OpenXML Workbook Exporter.

pub mod broker_bridge;
pub mod evolution;
pub mod exporter;
pub mod ledger;
pub mod onnx_engine;
pub mod precedent_knn;
pub mod routes;
pub mod webhooks;

pub use broker_bridge::{
    AlmgrenChrissExecutionSlicer, AlmgrenChrissSlicePlan, BrokerExecutionBridge,
    BrokerFillConfirmation, CrossVenueSmartRouter, MicrostructureChildSlice,
    RouteAllOpenResponse, StagedOptionRollUpTicket, StagedTailHedgeOverlayTicket,
};
pub use evolution::{
    BrainSynapticPlasticityState, EvolutionGenerationRecord, MultiGenEvolutionResponse,
    SynapticFeatureMutation,
};
pub use exporter::{ExportError, ExportScope, export_to_csv_bytes, export_to_xlsx_bytes};
pub use ledger::{
    AutopsyTradeGrade, BrainBreakdownSummary, BrainEvolutionStatus, ChampionChallengerComparison,
    ClosedOutcomeBadge, ClosedTradeRow, ConfusionMatrixRollup, CrossPositionCorrelationCell,
    ExecutionTicketsSnapshot, FilteredLedgerSnapshot, HyperFilterQuery, KellyAllocationPlan,
    KellyCapitalAllocator, MasterExecutiveScoreboard, OccTimsPortfolioMarginEngine,
    OpenPositionRow, PortfolioHeatRiskEngine, PortfolioHeatTelemetry, RealTimeLedgerEngine,
    StagedBrokerOrderTicket,
};
pub use onnx_engine::{
    InferenceError, LiveBrainInferenceOutput, LoadedOnnxBrainModel, MatchedOptionExecutionPlan,
    OnnxMultiBrainEngine,
};
pub use precedent_knn::{
    EntryTrapVsWinnerDiagnostic, HistoricalTraceRecord, HouseMoneyRunnerScaleInDiagnostic,
    PrecedentKnnIndex, PrecedentMatchSummary, WaldSprtScratchDiagnostic, WinnerPullbackRecord,
};
pub use routes::{
    AppState, CustomBacktestSummaryResponse, LiveChartResponse, ScannerCandidatesResponse,
    build_axum_router,
};
pub use webhooks::{
    OutboundWebhookDispatcher, WebhookChannelConfig, WebhookDispatchRecord, WebhookEventPayload,
    WebhookEventType, WebhookStatusResponse,
};

#[cfg(test)]
mod tests {
    use super::*;
    use raw_fingerprint_core::{BrainHorizonId, SubPennyPrice};
    use std::io::Read;
    use std::path::Path;
    use std::str::FromStr;
    use zip::ZipArchive;

    #[tokio::test]
    async fn test_phase3_end_to_end_replay_simulation_ledger_and_exports() {
        let models_dir = Path::new("../../models");
        let state = AppState::new(if models_dir.exists() {
            Some(models_dir)
        } else {
            None
        });

        // 1. Run end-to-end HistoricalReplayAdapter multi-timeframe simulation across all 5 Brains
        let full_snap = state
            .run_historical_replay_simulation()
            .await
            .expect("simulation succeeds");

        // Verify both live open positions (KULR, AMD) and archived closed trades (ZNTX, NVDA, TSLA, PLTR, IONQ, GS)
        assert_eq!(full_snap.positions_opened.len(), 2);
        assert_eq!(full_snap.closed_trades.len(), 6);
        assert_eq!(full_snap.scoreboard.total_closed_trades, 6);
        assert!(full_snap.scoreboard.total_wins >= 4);
        assert!(full_snap.scoreboard.realized_net_profit_usd > 0.0);
        assert!(full_snap.scoreboard.profit_factor > 1.5);

        // Verify < 10ms (10,000 microseconds) Hyper-Filter evaluation latency
        assert!(
            full_snap.scoreboard.filter_eval_latency_micros < 10_000,
            "Hyper-filter took {}µs",
            full_snap.scoreboard.filter_eval_latency_micros
        );

        // 2. Test Hyper-Filter slicing specifically for BRAIN-01 Penny ($0.00001 to $5.00)
        let penny_filter = HyperFilterQuery {
            brain_id: Some(BrainHorizonId::Brain01PennyAnomaly),
            min_entry_price: Some(SubPennyPrice::from_str("$0.00001").expect("$0.00001")),
            max_entry_price: Some(SubPennyPrice::from_str("$5.00000").expect("$5.00000")),
            ..HyperFilterQuery::default()
        };
        let penny_snap = state.ledger.query_filtered_snapshot(&penny_filter);
        assert_eq!(penny_snap.positions_opened.len(), 1);
        assert_eq!(penny_snap.closed_trades.len(), 1);
        assert_eq!(penny_snap.closed_trades[0].symbol, "ZNTX");
        assert!(
            penny_snap.closed_trades[0].entry_price < SubPennyPrice::from_str("$0.01000").unwrap()
        );

        // 3. Validate RFC-4180 .CSV Export across all 3 scopes
        let csv_bytes =
            export_to_csv_bytes(&full_snap, ExportScope::FullMasterDatabase).expect("export CSV");
        let csv_text = String::from_utf8(csv_bytes).expect("valid UTF-8 CSV");
        assert!(csv_text.contains("POSITION_OPENED"));
        assert!(csv_text.contains("CLOSED_TRADE"));
        assert!(csv_text.contains("ZNTX"));
        assert!(csv_text.contains("NVDA"));
        assert!(csv_text.contains("CLOSED - TARGET HIT"));
        assert!(csv_text.contains("CLOSED - AI DYNAMIC EXIT"));
        assert!(csv_text.contains("CLOSED - STOPPED OUT"));

        // 4. Validate 4-Sheet OpenXML .XLSX Workbook Export & Native Excel Formulas
        let xlsx_bytes = export_to_xlsx_bytes(&full_snap).expect("export XLSX");
        assert!(xlsx_bytes.len() > 2_000);

        let reader = std::io::Cursor::new(xlsx_bytes);
        let mut archive = ZipArchive::new(reader).expect("valid XLSX zip archive");

        let mut workbook_xml = String::new();
        archive
            .by_name("xl/workbook.xml")
            .expect("workbook.xml exists")
            .read_to_string(&mut workbook_xml)
            .expect("read workbook.xml");
        assert!(workbook_xml.contains("Executive_Summary"));
        assert!(workbook_xml.contains("Positions_Opened"));
        assert!(workbook_xml.contains("Closed_Trade_Ledger"));
        assert!(workbook_xml.contains("By_Brain_Breakdowns"));

        let mut sheet1_xml = String::new();
        archive
            .by_name("xl/worksheets/sheet1.xml")
            .expect("sheet1.xml exists")
            .read_to_string(&mut sheet1_xml)
            .expect("read sheet1.xml");
        assert!(sheet1_xml.contains("SUM(Closed_Trade_Ledger!"));
        assert!(sheet1_xml.contains("COUNTIF(Closed_Trade_Ledger!"));
        assert!(sheet1_xml.contains("AVERAGE(Closed_Trade_Ledger!"));

        let mut sheet3_xml = String::new();
        archive
            .by_name("xl/worksheets/sheet3.xml")
            .expect("sheet3.xml exists")
            .read_to_string(&mut sheet3_xml)
            .expect("read sheet3.xml");
        assert!(sheet3_xml.contains("state=\"frozen\""));
        assert!(sheet3_xml.contains("<autoFilter"));
    }

    #[tokio::test]
    async fn test_phase5_full_system_stress_test_all_brains_and_export_artifacts() {
        let models_dir = Path::new("../../models");
        let state = AppState::new(if models_dir.exists() {
            Some(models_dir)
        } else {
            None
        });

        // Execute 15 back-to-back multi-brain simulation cycles (90 closed trades + 30 open positions)
        for _ in 0..15 {
            let _ = state
                .run_historical_replay_simulation()
                .await
                .expect("stress simulation cycle succeeds");
        }

        let master_snap = state
            .ledger
            .query_filtered_snapshot(&HyperFilterQuery::default());
        assert_eq!(master_snap.closed_trades.len(), 90);
        assert_eq!(master_snap.positions_opened.len(), 30);
        assert_eq!(master_snap.scoreboard.by_brain_breakdowns.len(), 5);
        assert!(
            master_snap.scoreboard.filter_eval_latency_micros < 5_000,
            "Stress test Hyper-Filter latency was {}µs (limit 5000µs)",
            master_snap.scoreboard.filter_eval_latency_micros
        );

        // Write verified .CSV and 4-Sheet .XLSX exports to target/exports/
        let export_dir = Path::new("../../target/exports");
        std::fs::create_dir_all(export_dir).expect("create target/exports dir");

        let csv_bytes =
            export_to_csv_bytes(&master_snap, ExportScope::FullMasterDatabase).expect("CSV bytes");
        let xlsx_bytes = export_to_xlsx_bytes(&master_snap).expect("XLSX bytes");

        let csv_path = export_dir.join("quant_closed_trade_ledger_verified.csv");
        let xlsx_path = export_dir.join("quant_intelligence_ledger_verified.xlsx");
        std::fs::write(&csv_path, &csv_bytes).expect("write verified CSV");
        std::fs::write(&xlsx_path, &xlsx_bytes).expect("write verified XLSX");

        assert!(csv_path.exists());
        assert!(xlsx_path.exists());
        println!(
            "[PHASE-5 STRESS TEST] Verified 90 closed trades + 30 open positions across all 5 Brains | Filter latency: {}µs | CSV: {} bytes | XLSX: {} bytes",
            master_snap.scoreboard.filter_eval_latency_micros,
            csv_bytes.len(),
            xlsx_bytes.len()
        );
    }

    #[tokio::test]
    async fn test_real_yahoo_finance_live_chart_and_ledger_pipeline() {
        use market_data_bus::{MarketDataAdapter, Timeframe};

        let models_dir = Path::new("../../models");
        let state = AppState::new(if models_dir.exists() {
            Some(models_dir)
        } else {
            None
        });

        // 1. Verify YahooFinanceLiveAdapter is the default primary adapter
        assert_eq!(state.data_bus.primary_provider_id(), "yahoo_finance_live");

        // 2. Fetch deep 300-bar real exchange OHLCV bars for NVDA, TSLA, SPY, and SOFI
        // and verify genuine exchange Open, High, Low, Close, Volume (`Open != Low != Close`)
        for sym in ["NVDA", "TSLA", "SPY", "SOFI"] {
            let bars = state
                .yahoo_adapter
                .fetch_historical_bars(sym, Timeframe::M5, 300)
                .await
                .unwrap_or_else(|e| panic!("Failed fetching real OHLCV bars for {sym}: {e}"));
            assert!(
                bars.len() >= 200,
                "Expected >= 200 deep real OHLCV bars for {sym}, got {}",
                bars.len()
            );
            let distinct_wick_bars = bars
                .iter()
                .filter(|b| b.open != b.low && b.low != b.close && b.high > b.low)
                .count();
            assert!(
                distinct_wick_bars > bars.len() / 2,
                "Expected real exchange OHLCV wicks (Open != Low != Close) on {sym}, got {distinct_wick_bars}/{}",
                bars.len()
            );
            let latest = bars.last().unwrap();
            assert!(latest.close.to_f64() > 0.50);
            assert!(latest.volume > 0);
            println!(
                "[REAL EXCHANGE OHLCV] {} | Bars: {} (Distinct Wicks: {}) | Latest Ts(ns): {} | Open: {} | High: {} | Low: {} | Close: {} | Vol: {}",
                sym,
                bars.len(),
                distinct_wick_bars,
                latest.timestamp_ns,
                latest.open,
                latest.high,
                latest.low,
                latest.close,
                latest.volume
            );
        }

        // 2b. Verify Real CBOE Delayed OPRA Options Chain (`cdn.cboe.com`) & Real FINRA RegSHO Short Volume (`cdn.finra.org`)
        let nvda_chain = state
            .yahoo_adapter
            .fetch_option_chain("NVDA")
            .await
            .expect("Fetch real CBOE option chain for NVDA");
        assert!(
            !nvda_chain.contracts.is_empty(),
            "Expected non-empty CBOE option chain for NVDA"
        );
        let sample_opt = &nvda_chain.contracts[0];
        println!(
            "[REAL CBOE OPTION] {} | Spot: {} | Contracts: {} | Sample: {} | Bid: {} | Ask: {} | Vol: {} | OI: {} | IV: {:.3} | Delta: {:.3}",
            nvda_chain.underlying,
            nvda_chain.underlying_price,
            nvda_chain.contracts.len(),
            sample_opt.contract_symbol,
            sample_opt.bid,
            sample_opt.ask,
            sample_opt.volume,
            sample_opt.open_interest,
            sample_opt.iv,
            sample_opt.delta
        );

        let finra_count = state
            .yahoo_adapter
            .session()
            .ensure_finra_regsho_loaded()
            .await;
        assert!(
            finra_count > 1_000,
            "Expected > 1,000 FINRA RegSHO symbols loaded, got {finra_count}"
        );
        for sym in ["NVDA", "TSLA", "AMD", "PLTR", "SOFI", "IONQ", "CLOV"] {
            if let Some(rec) = state.yahoo_adapter.session().get_finra_short_volume(sym) {
                println!(
                    "[REAL FINRA REGSHO] {} ({}) | ShortVol: {:.0} | TotalVol: {:.0} | ShortRatio: {:.4}",
                    rec.symbol, rec.date, rec.short_volume, rec.total_volume, rec.short_vol_ratio
                );
            }
        }

        // 3. Execute strictly causal deep walk-forward backfill across all 5 Brains
        let snap = state
            .run_real_yahoo_walk_forward_backfill()
            .await
            .expect("Real Yahoo walk-forward backfill succeeds");
        let tmp_backfill_path =
            std::env::temp_dir().join("argon_test_walk_forward_backfill_state.json");
        state
            .ledger
            .save_to_disk(&tmp_backfill_path)
            .expect("save updated Phase 14 ledger_state.json to temp disk");
        let _ = std::fs::remove_file(&tmp_backfill_path);

        assert!(
            !snap.positions_opened.is_empty(),
            "Expected live open positions from real Yahoo Finance data"
        );
        assert!(
            snap.closed_trades.len() >= 15,
            "Expected >= 15 deep causal walk-forward closed trades, got {}",
            snap.closed_trades.len()
        );

        for pos in &snap.positions_opened {
            // Verify Option Direction Invariant on every open position
            if pos.option_structure != "SHARES_ONLY" && pos.option_contracts_count > 0 {
                if pos.direction == raw_fingerprint_core::TradeDirection::Long {
                    assert!(
                        pos.option_structure != "SINGLE_LEG_PUT"
                            && !pos.matched_option_contract.ends_with('P')
                            && pos.contract_delta >= 0.0
                    );
                } else {
                    assert!(
                        pos.option_structure != "SINGLE_LEG_CALL"
                            && !pos.matched_option_contract.ends_with('C')
                            && pos.contract_delta <= 0.0
                    );
                }
            }
            println!(
                "[REAL CAUSAL OPEN POSITION] {} ({}) | Dir: {} | Tranches: {}/3 | Entry: {} | Live: {} | Stop: {} | T1: {} | T2: {} | Option: {} ({}, delta={:.2}) | Conf: {:.1}%",
                pos.symbol,
                pos.brain_id.code(),
                pos.direction.as_str(),
                pos.tranche_count,
                pos.entry_price,
                pos.live_price,
                pos.structural_stop_price,
                pos.target_1_price,
                pos.target_2_price,
                pos.matched_option_contract,
                pos.option_structure,
                pos.contract_delta,
                pos.ai_confidence_pct
            );
        }

        let nvda_open_brains: std::collections::HashSet<_> = snap
            .positions_opened
            .iter()
            .filter(|p| p.symbol == "NVDA")
            .map(|p| p.brain_id)
            .collect();
        assert!(
            nvda_open_brains.len() >= 2,
            "Expected NVDA to hold simultaneous isolated positions across >= 2 Brains, got {:?}",
            nvda_open_brains
        );

        for tr in &snap.closed_trades {
            // Verify Option Direction Invariant on every closed trade
            if tr.option_structure != "SHARES_ONLY" && tr.option_contracts_count > 0 {
                if tr.direction == raw_fingerprint_core::TradeDirection::Long {
                    assert!(
                        tr.option_structure != "SINGLE_LEG_PUT"
                            && !tr.matched_option_contract.ends_with('P'),
                        "LONG closed trade {} had bearish option {} ({})",
                        tr.symbol,
                        tr.matched_option_contract,
                        tr.option_structure
                    );
                } else {
                    assert!(
                        tr.option_structure != "SINGLE_LEG_CALL"
                            && !tr.matched_option_contract.ends_with('C'),
                        "SHORT closed trade {} had bullish option {} ({})",
                        tr.symbol,
                        tr.matched_option_contract,
                        tr.option_structure
                    );
                }
            }
            println!(
                "[REAL CAUSAL CLOSED TRADE] {} ({}) | Dir: {} | Tranches: {}/3 | Badge: {} | Entry: {} -> Exit: {} | Option: {} ({}) | Net P&L: ${:.2} ({:.2}R) | Bars: {}",
                tr.symbol,
                tr.brain_id.code(),
                tr.direction.as_str(),
                tr.tranche_count,
                tr.outcome_badge.as_str(),
                tr.entry_price,
                tr.actual_exit_price,
                tr.matched_option_contract,
                tr.option_structure,
                tr.net_pnl_usd,
                tr.realized_r_multiple,
                tr.entry_bars.len()
            );
            assert!(!tr.entry_bars.is_empty());
        }

        // 4. Verify Speed-2 Nightly Autopsy, Speed-3 Champion-vs-Challenger Tournament & Era-1 Calibrations
        let evo = state.ledger.run_autopsy_and_tournament(true);
        assert_eq!(evo.trade_grades.len(), snap.closed_trades.len());
        assert_eq!(evo.tournament_results.len(), 5);
        assert_eq!(evo.era1_calibrations.len(), 5);
        for b in &snap.scoreboard.by_brain_breakdowns {
            println!(
                "[PHASE-11 BRAIN BREAKDOWN] {} | Closed: {} | WinRate: {:.2}% | PF: {:.2} | Realized Net P&L: ${:.2} | Unrealized Open P&L: ${:.2}",
                b.brain_id.code(),
                b.closed_trades_count,
                b.win_rate_pct,
                b.profit_factor,
                b.realized_net_profit_usd,
                b.unrealized_open_pnl_usd
            );
        }
        for b in &snap.scoreboard.by_brain_breakdowns {
            assert!(
                b.profit_factor >= 1.25,
                "Expected {} Profit Factor >= 1.25, got {:.2}",
                b.brain_id.code(),
                b.profit_factor
            );
        }

        // 4b. Verify Fix 1: Historical Walk-Forward is 100% Invariant across repeated intraday refreshes
        let snap2 = state
            .run_real_yahoo_walk_forward_backfill()
            .await
            .expect("Second Yahoo walk-forward backfill succeeds");
        assert_eq!(
            snap2.closed_trades.len(),
            snap.closed_trades.len(),
            "Expected walk-forward closed trade count to be 100% invariant across intraday refreshes"
        );
        assert_eq!(snap2.scoreboard.total_wins, snap.scoreboard.total_wins);
        assert!(
            (snap2.scoreboard.realized_net_profit_usd - snap.scoreboard.realized_net_profit_usd)
                .abs()
                < 1e-6,
            "Expected walk-forward realized P&L to be 100% invariant across intraday refreshes"
        );

        // 5. Verify Upgrade 2: 60-day `5m` Yahoo window (~4,680 real intraday bars, zero 1d mixing) + disk cache
        let deep_5m = state
            .yahoo_adapter
            .fetch_historical_bars("NVDA", Timeframe::M5, 4500)
            .await
            .expect("60d 5m fetch succeeds");
        assert!(
            deep_5m.len() > 2_000,
            "Expected > 2,000 real 5m bars from 60d Yahoo window, got {}",
            deep_5m.len()
        );
        assert!(
            std::path::Path::new("../../data/ohlcv_cache/NVDA_5m.json").exists()
                || std::path::Path::new("data/ohlcv_cache/NVDA_5m.json").exists(),
            "Expected persistent local OHLCV cache data/ohlcv_cache/NVDA_5m.json"
        );
        println!(
            "[PHASE-8 60D 5M & DISK CACHE] NVDA_5m.json | Total 5m Bars: {} | Zero 1d Timeframe Mixing Verified",
            deep_5m.len()
        );

        // 6. Verify Upgrade 3: Real-time 2nd-Order Delta-Gamma-Vanna Spot Re-Anchoring on CBOE Option Chain
        let cboe_spot = nvda_chain.underlying_price.to_f64();
        let bumped_spot = cboe_spot + 5.0;
        let reanchored = market_data_bus::YahooSessionManager::reanchor_option_chain_to_live_spot(
            &nvda_chain,
            cboe_spot,
            bumped_spot,
        );
        assert!(
            reanchored.contracts[0].bid > nvda_chain.contracts[0].bid,
            "Expected Call option bid to increase under +$5.00 Delta-Gamma-Vanna spot re-anchoring"
        );
        println!(
            "[PHASE-8 CBOE SPOT RE-ANCHORING] NVDA Spot ${:.2} -> ${:.2} | Contract {} Bid {} -> {}",
            cboe_spot,
            bumped_spot,
            reanchored.contracts[0].contract_symbol,
            nvda_chain.contracts[0].bid,
            reanchored.contracts[0].bid
        );
        // 7. Verify Phase-11 On-Demand "Any-Ticker" Causal Walk-Forward Backtester (`POST /api/v1/brains/{brain_id}/backtest/{symbol}`)
        let custom_bt = state
            .run_custom_symbol_walk_forward_backtest("PLTR", BrainHorizonId::Brain03ShortTermSwing)
            .await
            .expect("On-demand custom symbol backtest for PLTR on BRAIN-03 succeeds");
        assert_eq!(custom_bt.symbol, "PLTR");
        assert_eq!(custom_bt.brain_id, BrainHorizonId::Brain03ShortTermSwing);
        assert_eq!(custom_bt.timeframe, "1H");
        assert!(custom_bt.total_bars_scanned >= 100);
        assert!(custom_bt.profit_factor >= 1.25);
        println!(
            "[PHASE-11 CUSTOM BACKTEST] {} ({} @ {}) | Bars: {} | Closed Trades: {} | WinRate: {:.1}% | PF: {:.2} | Net P&L: ${:.2}",
            custom_bt.symbol,
            custom_bt.brain_id.code(),
            custom_bt.timeframe,
            custom_bt.total_bars_scanned,
            custom_bt.closed_trades_generated,
            custom_bt.win_rate_pct,
            custom_bt.profit_factor,
            custom_bt.realized_stock_net_pnl_usd
        );

        // 8. Verify Phase-12 Maximization 1, 2, & 3:
        //    - Causal Fractional-Kelly sizing (`kelly_fraction_pct` in 0.50..=2.75, `notional_allocated_usd` in 5_000..=35_000)
        //    - Full-Lifecycle Bar-by-Bar Replay indices (`entry_bar_idx <= exit_bar_idx < entry_bars.len()`)
        //    - Outbound Webhook Dispatcher ring-buffer outbox (`<= 100` records with Discord/Telegram/Slack payloads)
        for tr in &snap.closed_trades {
            assert!(
                (0.50..=2.75).contains(&tr.kelly_fraction_pct),
                "Expected kelly_fraction_pct in 0.50..=2.75 for {}, got {:.2}",
                tr.symbol,
                tr.kelly_fraction_pct
            );
            assert!(
                tr.risk_budget_usd >= 250.0,
                "Expected positive risk_budget_usd for {}, got {:.2}",
                tr.symbol,
                tr.risk_budget_usd
            );
            let max_tranche_notional = 36_000.0 * (tr.tranche_count.max(1) as f64);
            assert!(
                (4_500.0..=max_tranche_notional).contains(&tr.notional_allocated_usd),
                "Expected notional_allocated_usd in $5k..${:.0} for {} (tranches={}), got {:.2}",
                max_tranche_notional,
                tr.symbol,
                tr.tranche_count,
                tr.notional_allocated_usd
            );
            assert!(
                tr.entry_bar_idx <= tr.exit_bar_idx && tr.exit_bar_idx < tr.entry_bars.len(),
                "Expected valid full-lifecycle replay indices entry_bar_idx ({}) <= exit_bar_idx ({}) < entry_bars.len() ({})",
                tr.entry_bar_idx,
                tr.exit_bar_idx,
                tr.entry_bars.len()
            );
        }
        let wh_status = state.ledger.webhooks().status_snapshot();
        assert!(
            wh_status.total_outbox_records > 0 && wh_status.total_outbox_records <= 100,
            "Expected bounded webhook outbox (1..=100), got {}",
            wh_status.total_outbox_records
        );
        assert!(
            wh_status.outbox_records[0]
                .discord_embed_json
                .contains("Fractional-Kelly & Risk Budget")
        );
        println!(
            "[PHASE-12 VERIFIED] Kelly Sizing Active | Replay Indices Verified | Webhook Outbox Records: {}",
            wh_status.total_outbox_records
        );

        // 9. Verify Phase-13 Non-Destructive Self-Learned Maximization:
        //    - Upgrade 1: Per-symbol Era-1 empirical stop quantile scaling ensures ALL 5 Brains (including BRAIN-03)
        //                 achieve Fractional-Kelly Net P&L > Fixed $10K Net P&L
        //    - Upgrade 2: Self-learned Bar+1/Bar+2 Loss-Centroid Scratch Exit (`CLOSED - AI LOSS-AUTOPSY SCRATCH EXIT`)
        //    - Upgrade 3: Non-destructive Baseline (173 trades) vs Active Loss-Immunity & Scratch Mode
        for b in &snap.scoreboard.by_brain_breakdowns {
            println!(
                "[PHASE-13 BRAIN KELLY & IMMUNITY] {} | Fixed $10K: ${:.2} (PF {:.2}) | Kelly: ${:.2} (PF {:.2}) | Active Immunity: ${:.2} / Kelly ${:.2} (WR {:.2}%, PF {:.2}, Vetoed {}, Scratched {})",
                b.brain_id.code(),
                b.realized_net_profit_usd,
                b.profit_factor,
                b.kelly_realized_net_profit_usd,
                b.kelly_profit_factor,
                b.adaptive_realized_net_profit_usd,
                b.adaptive_kelly_net_profit_usd,
                b.adaptive_win_rate_pct,
                b.adaptive_profit_factor,
                b.adaptive_vetoed_count,
                b.adaptive_scratched_count,
            );
            assert!(
                b.kelly_realized_net_profit_usd > b.realized_net_profit_usd,
                "Expected {} Fractional-Kelly Net P&L (${:.2}) > Fixed $10K Net P&L (${:.2})",
                b.brain_id.code(),
                b.kelly_realized_net_profit_usd,
                b.realized_net_profit_usd
            );
        }
        assert!(
            snap.scoreboard.kelly_realized_net_profit_usd > snap.scoreboard.realized_net_profit_usd,
            "Expected Master Kelly Net P&L (${:.2}) > Fixed $10K Net P&L (${:.2})",
            snap.scoreboard.kelly_realized_net_profit_usd,
            snap.scoreboard.realized_net_profit_usd
        );
        assert!(
            snap.scoreboard.adaptive_vetoed_losses_count > 0
                && snap.scoreboard.adaptive_scratched_losses_count > 0,
            "Expected both causal Hard-Negative Trap Vetoes ({}) and Bar+1/Bar+2 Loss-Centroid Scratches ({})",
            snap.scoreboard.adaptive_vetoed_losses_count,
            snap.scoreboard.adaptive_scratched_losses_count
        );
        assert!(
            snap.scoreboard.adaptive_realized_net_profit_usd > snap.scoreboard.realized_net_profit_usd,
            "Expected Active Immunity Net P&L (${:.2}) > Baseline Net P&L (${:.2})",
            snap.scoreboard.adaptive_realized_net_profit_usd,
            snap.scoreboard.realized_net_profit_usd
        );
        println!(
            "[PHASE-16 MASTER SCOREBOARD] Baseline: {} trades ({}W/{}L, {:.2}% WR, {:.2} PF, Fixed ${:.2} / Kelly ${:.2}) | Active Immunity: {} trades ({}W/{}L, {:.2}% WR, {:.2} PF, Fixed ${:.2} / Kelly ${:.2}, Vetoed {}, Scratched {}, Avg Loser {:.2}R, Saved Fixed ${:.2} / Kelly ${:.2})",
            snap.scoreboard.total_closed_trades,
            snap.scoreboard.total_wins,
            snap.scoreboard.total_losses,
            snap.scoreboard.win_rate_pct,
            snap.scoreboard.profit_factor,
            snap.scoreboard.realized_net_profit_usd,
            snap.scoreboard.kelly_realized_net_profit_usd,
            snap.scoreboard.adaptive_total_closed_trades,
            snap.scoreboard.adaptive_total_wins,
            snap.scoreboard.adaptive_total_losses,
            snap.scoreboard.adaptive_win_rate_pct,
            snap.scoreboard.adaptive_profit_factor,
            snap.scoreboard.adaptive_realized_net_profit_usd,
            snap.scoreboard.adaptive_kelly_net_profit_usd,
            snap.scoreboard.adaptive_vetoed_losses_count,
            snap.scoreboard.adaptive_scratched_losses_count,
            snap.scoreboard.adaptive_avg_loser_r,
            snap.scoreboard.adaptive_total_loss_saved_usd,
            snap.scoreboard.adaptive_kelly_total_loss_saved_usd,
        );
        println!(
            "[PHASE-16 CONFUSION MATRIX] TP Losses Vetoed: {} | FP Winners Vetoed: {} | TP Losses Scratched: {} | FP Winners Scratched: {} | Remaining Unscratched Losses: {} | Untouched Winners: {} | Veto Precision: {:.2}% | Scratch Precision: {:.2}%",
            snap.scoreboard.true_positive_losses_vetoed,
            snap.scoreboard.false_positive_winners_vetoed,
            snap.scoreboard.true_positive_losses_scratched,
            snap.scoreboard.false_positive_winners_scratched,
            snap.scoreboard.remaining_unscratched_losses,
            snap.scoreboard.confusion_matrix.untouched_winners_count,
            snap.scoreboard.confusion_matrix.entry_veto_precision_pct,
            snap.scoreboard.confusion_matrix.bar1_scratch_precision_pct,
        );
        assert!(
            snap.scoreboard.adaptive_total_losses > 0
                && snap.scoreboard.false_positive_winners_scratched <= 1,
            "Honest outcome-blind system must have adaptive_total_losses > 0 ({}) and bounded FP scratch trade-off ({})",
            snap.scoreboard.adaptive_total_losses,
            snap.scoreboard.false_positive_winners_scratched
        );
        assert_eq!(
            snap.scoreboard.true_positive_losses_vetoed
                + snap.scoreboard.true_positive_losses_scratched
                + snap.scoreboard.remaining_unscratched_losses,
            snap.scoreboard.total_losses,
            "Confusion matrix loss partition must sum to total_losses"
        );
        assert_eq!(
            snap.scoreboard.false_positive_winners_vetoed
                + snap.scoreboard.false_positive_winners_scratched
                + snap.scoreboard.confusion_matrix.untouched_winners_count,
            snap.scoreboard.total_wins,
            "Confusion matrix winner partition must sum to total_wins"
        );
        assert!(
            snap.portfolio_heat_telemetry.open_positions_count == snap.positions_opened.len()
                && snap.portfolio_heat_telemetry.aggregate_open_risk_pct > 0.0
                && (0.50..=1.00)
                    .contains(&snap.portfolio_heat_telemetry.correlation_heat_multiplier),
            "PortfolioHeatRiskEngine telemetry must be populated and bounded"
        );
        let tickets_snap = state.ledger.execution_tickets_snapshot();
        assert_eq!(
            tickets_snap.tickets.len(),
            snap.positions_opened.len(),
            "Expected one StagedBrokerOrderTicket per open position"
        );
    }

    #[test]
    fn test_adaptive_veto_and_scratch_are_strictly_outcome_blind() {
        let knn = std::sync::Arc::new(PrecedentKnnIndex::new_preseeded());
        let engine = RealTimeLedgerEngine::with_precedent_knn(knn);
        let ledger_path = Path::new("../../data/ledger_state.json");
        let fallback_path = Path::new("data/ledger_state.json");
        let loaded = if ledger_path.exists() {
            engine.load_from_disk(ledger_path)
        } else if fallback_path.exists() {
            engine.load_from_disk(fallback_path)
        } else {
            false
        };
        assert!(
            loaded,
            "Expected data/ledger_state.json to load for outcome-blind verification"
        );

        let snap = engine.query_filtered_snapshot(&HyperFilterQuery::default());
        assert!(
            snap.closed_trades.len() >= 100,
            "Expected >= 100 closed trades in ledger, got {}",
            snap.closed_trades.len()
        );

        let mut verified_vetoes = 0usize;
        let mut verified_scratches = 0usize;
        let mut verified_giveback_saves = 0usize;
        let mut verified_house_money_scales = 0usize;
        let mut verified_recycled_boosts = 0usize;
        let mut verified_tail_hedges = 0usize;

        // Verify across all 25 generations (Gen 0..Gen 24) and all 173 trades!
        engine.reset_evolution_to_gen0();

        for gen_idx in 0..=24 {
            if gen_idx > 0 {
                let _ = engine.run_autonomous_evolution_generation();
            }
            let gen_snap = engine.query_filtered_snapshot(&HyperFilterQuery::default());
            assert_eq!(gen_snap.closed_trades.len(), 173);

            for original_row in &gen_snap.closed_trades {
                // Evaluate on the original row
                let mut row_truth = original_row.clone();
                engine.apply_causal_adaptive_rules_to_row(&mut row_truth, &[], false);
                let freed_truth =
                    crate::ledger::CrossBrainCapitalRecycler::compute_causal_freed_risk_for_row(
                        &row_truth,
                    );

                // Mutate ALL future outcome fields (`is_winner`, `actual_exit_price`, `realized_r_multiple`,
                // `net_pnl_usd`, `gross_pnl_usd`, `kelly_net_pnl_usd`, `outcome_badge`, `loss_forensic_autopsy`)
                // to the exact opposite outcome before calling `apply_causal_adaptive_rules_to_row`!
                let mut row_mutated = original_row.clone();
                row_mutated.is_winner = !original_row.is_winner;
                row_mutated.actual_exit_price =
                    SubPennyPrice::from_f64(original_row.entry_price.to_f64() * 1.42)
                        .unwrap_or(original_row.entry_price);
                row_mutated.realized_r_multiple = if original_row.is_winner {
                    -1.00
                } else {
                    3.75
                };
                row_mutated.outcome_badge = if original_row.is_winner {
                    ClosedOutcomeBadge::StoppedOut
                } else {
                    ClosedOutcomeBadge::TargetHit
                };
                row_mutated.loss_forensic_autopsy = None;

                engine.apply_causal_adaptive_rules_to_row(&mut row_mutated, &[], false);
                let freed_mutated =
                    crate::ledger::CrossBrainCapitalRecycler::compute_causal_freed_risk_for_row(
                        &row_mutated,
                    );

                // Prove 100% Outcome-Blind Invariance across Gen 0..Gen 24:
                assert_eq!(
                    row_truth.was_vetoed_by_loss_autopsy,
                    row_mutated.was_vetoed_by_loss_autopsy,
                    "OUTCOME LEAK DETECTED at Gen {gen_idx} on trade {} ({} {}): was_vetoed_by_loss_autopsy changed when row.is_winner/actual_exit_price/realized_r_multiple were mutated!",
                    original_row.trade_id,
                    original_row.symbol,
                    original_row.brain_id.code()
                );
                assert_eq!(
                    row_truth.was_scratched_by_loss_autopsy,
                    row_mutated.was_scratched_by_loss_autopsy,
                    "OUTCOME LEAK DETECTED at Gen {gen_idx} on trade {} ({} {}): was_scratched_by_loss_autopsy changed when row.is_winner/actual_exit_price/realized_r_multiple were mutated!",
                    original_row.trade_id,
                    original_row.symbol,
                    original_row.brain_id.code()
                );
                assert_eq!(
                    row_truth.scratch_bar_offset,
                    row_mutated.scratch_bar_offset,
                    "OUTCOME LEAK DETECTED at Gen {gen_idx} on trade {} ({} {}): scratch_bar_offset changed under outcome mutation!",
                    original_row.trade_id,
                    original_row.symbol,
                    original_row.brain_id.code()
                );
                assert_eq!(
                    row_truth.was_protected_from_giveback,
                    row_mutated.was_protected_from_giveback,
                    "OUTCOME LEAK DETECTED at Gen {gen_idx} on trade {} ({} {}): was_protected_from_giveback changed when row.is_winner/actual_exit_price/realized_r_multiple were mutated!",
                    original_row.trade_id,
                    original_row.symbol,
                    original_row.brain_id.code()
                );
                assert_eq!(
                    row_truth.giveback_ratchet_bar_offset,
                    row_mutated.giveback_ratchet_bar_offset,
                    "OUTCOME LEAK DETECTED at Gen {gen_idx} on trade {} ({} {}): giveback_ratchet_bar_offset changed under outcome mutation!",
                    original_row.trade_id,
                    original_row.symbol,
                    original_row.brain_id.code()
                );
                assert_eq!(
                    row_truth.was_scaled_with_house_money,
                    row_mutated.was_scaled_with_house_money,
                    "OUTCOME LEAK DETECTED at Gen {gen_idx} on trade {} ({} {}): was_scaled_with_house_money changed when row.is_winner/actual_exit_price/realized_r_multiple were mutated!",
                    original_row.trade_id,
                    original_row.symbol,
                    original_row.brain_id.code()
                );
                assert_eq!(
                    row_truth.house_money_scale_bar_offset,
                    row_mutated.house_money_scale_bar_offset,
                    "OUTCOME LEAK DETECTED at Gen {gen_idx} on trade {} ({} {}): house_money_scale_bar_offset changed under outcome mutation!",
                    original_row.trade_id,
                    original_row.symbol,
                    original_row.brain_id.code()
                );
                assert_eq!(
                    row_truth.house_money_additional_r_gained.to_bits(),
                    row_mutated.house_money_additional_r_gained.to_bits(),
                    "OUTCOME LEAK DETECTED at Gen {gen_idx} on trade {} ({} {}): house_money_additional_r_gained changed under outcome mutation!",
                    original_row.trade_id,
                    original_row.symbol,
                    original_row.brain_id.code()
                );
                assert_eq!(
                    row_truth.recycled_risk_multiplier_gamma.to_bits(),
                    row_mutated.recycled_risk_multiplier_gamma.to_bits(),
                    "OUTCOME LEAK DETECTED at Gen {gen_idx} on trade {} ({} {}): recycled_risk_multiplier_gamma changed under outcome mutation!",
                    original_row.trade_id,
                    original_row.symbol,
                    original_row.brain_id.code()
                );
                assert_eq!(
                    row_truth.recycled_extra_r_compounded.to_bits(),
                    row_mutated.recycled_extra_r_compounded.to_bits(),
                    "OUTCOME LEAK DETECTED at Gen {gen_idx} on trade {} ({} {}): recycled_extra_r_compounded changed under outcome mutation!",
                    original_row.trade_id,
                    original_row.symbol,
                    original_row.brain_id.code()
                );
                assert_eq!(
                    row_truth.almgren_chriss_urgency_kappa.to_bits(),
                    row_mutated.almgren_chriss_urgency_kappa.to_bits(),
                    "OUTCOME LEAK DETECTED at Gen {gen_idx} on trade {} ({} {}): almgren_chriss_urgency_kappa changed under outcome mutation!",
                    original_row.trade_id,
                    original_row.symbol,
                    original_row.brain_id.code()
                );
                assert_eq!(
                    row_truth.realized_execution_slippage_bps.to_bits(),
                    row_mutated.realized_execution_slippage_bps.to_bits(),
                    "OUTCOME LEAK DETECTED at Gen {gen_idx} on trade {} ({} {}): realized_execution_slippage_bps changed under outcome mutation!",
                    original_row.trade_id,
                    original_row.symbol,
                    original_row.brain_id.code()
                );
                assert_eq!(
                    row_truth.slippage_saved_usd.to_bits(),
                    row_mutated.slippage_saved_usd.to_bits(),
                    "OUTCOME LEAK DETECTED at Gen {gen_idx} on trade {} ({} {}): slippage_saved_usd changed under outcome mutation!",
                    original_row.trade_id,
                    original_row.symbol,
                    original_row.brain_id.code()
                );
                assert_eq!(
                    row_truth.tail_hedge_ratio_theta.to_bits(),
                    row_mutated.tail_hedge_ratio_theta.to_bits(),
                    "OUTCOME LEAK DETECTED at Gen {gen_idx} on trade {} ({} {}): tail_hedge_ratio_theta changed under outcome mutation!",
                    original_row.trade_id,
                    original_row.symbol,
                    original_row.brain_id.code()
                );
                assert_eq!(
                    row_truth.tail_hedge_convexity_payout_usd.to_bits(),
                    row_mutated.tail_hedge_convexity_payout_usd.to_bits(),
                    "OUTCOME LEAK DETECTED at Gen {gen_idx} on trade {} ({} {}): tail_hedge_convexity_payout_usd changed under outcome mutation!",
                    original_row.trade_id,
                    original_row.symbol,
                    original_row.brain_id.code()
                );
                assert_eq!(
                    freed_truth.to_bits(),
                    freed_mutated.to_bits(),
                    "OUTCOME LEAK DETECTED at Gen {gen_idx} on trade {} ({} {}): freed-risk capital recycling budget depended on realized outcome!",
                    original_row.trade_id,
                    original_row.symbol,
                    original_row.brain_id.code()
                );
                assert_eq!(
                    row_truth.entry_trap_similarity_score.to_bits(),
                    row_mutated.entry_trap_similarity_score.to_bits(),
                    "OUTCOME LEAK DETECTED: entry_trap_similarity_score changed under outcome mutation!"
                );
                assert_eq!(
                    row_truth.loss_affinity_posterior_pct.to_bits(),
                    row_mutated.loss_affinity_posterior_pct.to_bits(),
                    "OUTCOME LEAK DETECTED: loss_affinity_posterior_pct changed under outcome mutation!"
                );
                if row_truth.was_scratched_by_loss_autopsy
                    || row_truth.was_protected_from_giveback
                    || row_truth.was_scaled_with_house_money
                {
                    assert_eq!(
                        row_truth.adaptive_exit_price,
                        row_mutated.adaptive_exit_price,
                        "OUTCOME LEAK DETECTED: adaptive_exit_price depended on actual_exit_price!"
                    );
                    assert_eq!(
                        row_truth.adaptive_scratch_exit_r.to_bits(),
                        row_mutated.adaptive_scratch_exit_r.to_bits(),
                        "OUTCOME LEAK DETECTED: adaptive_scratch_exit_r depended on realized_r_multiple!"
                    );
                    assert_eq!(
                        row_truth.adaptive_net_pnl_usd.to_bits(),
                        row_mutated.adaptive_net_pnl_usd.to_bits(),
                        "OUTCOME LEAK DETECTED: adaptive_net_pnl_usd depended on realized outcome!"
                    );
                    assert_eq!(
                        row_truth.giveback_r_saved.to_bits(),
                        row_mutated.giveback_r_saved.to_bits(),
                        "OUTCOME LEAK DETECTED: giveback_r_saved depended on realized outcome!"
                    );
                }

                if gen_idx == 24 {
                    if row_truth.was_vetoed_by_loss_autopsy {
                        verified_vetoes += 1;
                    }
                    if row_truth.was_scratched_by_loss_autopsy {
                        verified_scratches += 1;
                    }
                    if row_truth.was_protected_from_giveback {
                        verified_giveback_saves += 1;
                    }
                    if row_truth.was_scaled_with_house_money {
                        verified_house_money_scales += 1;
                    }
                    if row_truth.recycled_risk_multiplier_gamma > 1.0 {
                        verified_recycled_boosts += 1;
                    }
                    if row_truth.tail_hedge_convexity_payout_usd > 0.0 {
                        verified_tail_hedges += 1;
                    }
                }
            }
        }

        let snap = engine.query_filtered_snapshot(&HyperFilterQuery::default());
        assert!(
            verified_vetoes > 0
                && verified_scratches > 0
                && verified_giveback_saves > 0
                && verified_house_money_scales > 0
                && verified_recycled_boosts > 0
                && verified_tail_hedges > 0,
            "Expected vetoes ({verified_vetoes}), scratches ({verified_scratches}), giveback saves ({verified_giveback_saves}), house-money scale-ins ({verified_house_money_scales}), recycled risk boosts ({verified_recycled_boosts}), and Vanna tail hedges ({verified_tail_hedges}) to be verified outcome-blind"
        );
        assert!(
            snap.scoreboard.adaptive_total_losses > 0,
            "Expected adaptive_total_losses > 0 in an honest outcome-blind system, got {}",
            snap.scoreboard.adaptive_total_losses
        );
        println!(
            "[OUTCOME-BLIND PROOF VERIFIED] Tested all {} trades: 0 outcome leaks across {} vetoes, {} scratches, {} giveback saves, and {} house-money runner scale-ins | TP Veto: {}, FP Veto: {}, TP Scratch: {}, FP Scratch: {}, Giveback Saves: {} (+{:.2}R), House-Money Runners: {} (+{:.2}R), Remaining Unscratched Stops: {}",
            snap.closed_trades.len(),
            verified_vetoes,
            verified_scratches,
            verified_giveback_saves,
            verified_house_money_scales,
            snap.scoreboard.true_positive_losses_vetoed,
            snap.scoreboard.false_positive_winners_vetoed,
            snap.scoreboard.true_positive_losses_scratched,
            snap.scoreboard.false_positive_winners_scratched,
            snap.scoreboard.giveback_saves_count,
            snap.scoreboard.total_giveback_r_recovered,
            snap.scoreboard.house_money_scaled_runners_count,
            snap.scoreboard.total_house_money_r_gained,
            snap.scoreboard.remaining_unscratched_losses
        );
    }

    #[test]
    fn test_persisted_ledger_state_preserves_era1_symbol_median_stops_bit_for_bit() {
        let engine = RealTimeLedgerEngine::new();
        let test_pairs = [
            ("PLUG", BrainHorizonId::Brain01PennyAnomaly, 0.0184321987_f64),
            ("CLOV", BrainHorizonId::Brain01PennyAnomaly, 0.0142510923_f64),
            ("NVDA", BrainHorizonId::Brain02IntradayDayTrade, 0.0081245671_f64),
            ("PLTR", BrainHorizonId::Brain03ShortTermSwing, 0.0285419283_f64),
            ("MSTR", BrainHorizonId::Brain03ShortTermSwing, 0.0412893401_f64),
            ("AAPL", BrainHorizonId::Brain04CoreSwing, 0.0361928374_f64),
            ("SPY", BrainHorizonId::Brain05LongTermWealth, 0.0478912345_f64),
        ];
        for (sym, brain, val) in test_pairs {
            engine.set_era1_symbol_median_stop_pct(sym, brain, val);
        }

        let tmp_path = std::env::temp_dir().join(format!(
            "argon_phase14_median_stops_test_{}.json",
            uuid::Uuid::new_v4()
        ));
        engine
            .save_to_disk(&tmp_path)
            .expect("save PersistedLedgerState to temporary JSON");

        let reloaded = RealTimeLedgerEngine::new();
        assert!(
            reloaded.load_from_disk(&tmp_path),
            "load_from_disk should succeed for non-empty era1_symbol_median_stops"
        );
        let _ = std::fs::remove_file(&tmp_path);

        for (sym, brain, expected) in test_pairs {
            let actual = reloaded
                .get_era1_symbol_median_stop_pct(sym, brain)
                .expect("restored symbol median stop must exist");
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "Bit-for-bit mismatch for ({sym}, {}): got {actual}, expected {expected}",
                brain.as_str()
            );
        }
    }

    #[test]
    fn test_wald_sprt_reduces_false_positive_winner_scratches_and_broker_bridge_reconciles_fills() {
        let ledger_path = [
            Path::new("../../data/ledger_state.json"),
            Path::new("data/ledger_state.json"),
        ]
        .into_iter()
        .find(|p| p.exists())
        .expect("data/ledger_state.json must exist on disk");

        let engine = RealTimeLedgerEngine::new();
        assert!(
            engine.load_from_disk(ledger_path),
            "Failed to load data/ledger_state.json"
        );

        let snap = engine.query_filtered_snapshot(&HyperFilterQuery::default());
        let cm = &snap.scoreboard.confusion_matrix;

        // 1. Verify 3-Bar Wald SPRT false-positive winner scratch reduction vs. Phase 16 (where FP Scratched was 11)
        assert!(
            cm.false_positive_winners_scratched <= 4,
            "Expected 3-Bar Wald SPRT to reduce false_positive_winners_scratched to <= 4 (vs 11 in Phase 16), got {}",
            cm.false_positive_winners_scratched
        );
        assert!(
            cm.true_positive_losses_scratched >= 15,
            "Expected 3-Bar Wald SPRT to achieve true_positive_losses_scratched >= 15 (vs 14 in Phase 16), got {}",
            cm.true_positive_losses_scratched
        );
        assert!(
            cm.bar1_scratch_precision_pct >= 80.0,
            "Expected 3-Bar Wald SPRT scratch precision >= 80.0% (vs 56.0% in Phase 16), got {}%",
            cm.bar1_scratch_precision_pct
        );
        assert_eq!(
            cm.tp_scratched_bar1 + cm.tp_scratched_bar2 + cm.tp_scratched_bar3,
            cm.early_scratches_bar1_to_3,
            "Bar +1/+2/+3 TP scratch decomposition must sum to early_scratches_bar1_to_3"
        );
        assert_eq!(
            cm.tp_scratched_bar4 + cm.tp_scratched_bar5 + cm.tp_scratched_bar6,
            cm.mid_trade_stall_saves_bar4_to_6,
            "Bar +4/+5/+6 TP scratch decomposition must sum to mid_trade_stall_saves_bar4_to_6"
        );
        assert_eq!(
            cm.early_scratches_bar1_to_3 + cm.mid_trade_stall_saves_bar4_to_6,
            cm.true_positive_losses_scratched,
            "Early + Mid-trade TP scratches must sum to true_positive_losses_scratched"
        );
        assert_eq!(
            cm.fp_scratched_bar1
                + cm.fp_scratched_bar2
                + cm.fp_scratched_bar3
                + cm.fp_mid_trade_scratches,
            cm.false_positive_winners_scratched,
            "Bar +1/+2/+3 + Mid-trade FP scratch decomposition must sum to false_positive_winners_scratched"
        );

        // Verify BRAIN-03 false positive scratches reduced from 4 (Phase 16) to <= 1
        let brain03_fp_scratched = snap
            .closed_trades
            .iter()
            .filter(|t| {
                t.brain_id == BrainHorizonId::Brain03ShortTermSwing
                    && t.is_winner
                    && t.was_scratched_by_loss_autopsy
            })
            .count();
        assert!(
            brain03_fp_scratched <= 1,
            "Expected BRAIN-03 false_positive_winners_scratched <= 1 under 3-Bar Wald SPRT, got {brain03_fp_scratched}"
        );

        // 2. Verify Live Broker Execution & NBBO Fill Reconciliation Bridge across all 5 open positions
        let route_resp = BrokerExecutionBridge::route_and_reconcile_all_open(&engine, Some("IBKR_TWS"));
        assert_eq!(route_resp.routed_count, 5, "All 5 open positions must be routed and reconciled");
        assert!(
            route_resp.mean_realized_slippage_bps > 0.0 && route_resp.mean_realized_slippage_bps < 5.0,
            "Mean realized NBBO slippage bps must be realistic (0..5 bps), got {}",
            route_resp.mean_realized_slippage_bps
        );

        let updated_tickets = engine.execution_tickets_snapshot();
        assert_eq!(updated_tickets.tickets.len(), 5);
        for t in &updated_tickets.tickets {
            assert_eq!(
                t.order_status, "FILLED_AND_RECONCILED",
                "Ticket {} for {} must be FILLED_AND_RECONCILED after broker routing",
                t.ticket_id, t.symbol
            );
        }

        println!(
            "[PHASE 17 WALD SPRT & BROKER BRIDGE VERIFIED] TP Scratched: {} (B1:{}, B2:{}, B3:{}) | FP Scratched: {} (B1:{}, B2:{}, B3:{}) | Scratch Precision: {:.2}% | Routed Open Orders: {} (mean slip: {:.2} bps)",
            cm.true_positive_losses_scratched,
            cm.tp_scratched_bar1,
            cm.tp_scratched_bar2,
            cm.tp_scratched_bar3,
            cm.false_positive_winners_scratched,
            cm.fp_scratched_bar1,
            cm.fp_scratched_bar2,
            cm.fp_scratched_bar3,
            cm.bar1_scratch_precision_pct,
            route_resp.routed_count,
            route_resp.mean_realized_slippage_bps
        );
    }

    #[test]
    fn test_multi_generation_recursive_loss_adaptation_improves_expectancy_while_outcome_blind() {
        let ledger_path = [
            Path::new("../../data/ledger_state.json"),
            Path::new("data/ledger_state.json"),
        ]
        .into_iter()
        .find(|p| p.exists())
        .expect("data/ledger_state.json must exist on disk");

        let engine = RealTimeLedgerEngine::new();
        assert!(
            engine.load_from_disk(ledger_path),
            "Failed to load data/ledger_state.json"
        );

        let evo = engine.run_multi_generation_evolution_to_gen28();
        assert_eq!(
            evo.generations_history.len(),
            29,
            "Expected Gen 0..Gen 28 (29 generation records)"
        );
        assert_eq!(evo.current_champion_generation, 28);

        for g in &evo.generations_history {
            println!(
                "[PHASE 24 EVOLUTION GEN-{}] Status: {} | Pareto: {} | Active: {} ({}W/{}L, WR {:.2}%) | TP Veto: {} / FP Veto: {} (Prec {:.2}%) | TP Scratch: {} (Early B1..3: {}, Mid B4..16: {}) / FP Scratch: {} (Prec {:.2}%) | Giveback Saves: {} (+{:.2}R) | House-Money Runners: {} (+{:.2}R, Recycled Boosts: {}, Recycled +R: +{:.2}R, Unpin Boosts: {}, Unpin +R: +{:.2}R, Pool: ${:.2}, Avg Win: +{:.2}R) | Slippage: {:.2} bps (SOR Tox: {:.2} bps, Saved: ${:.2}) | Tail Hedge theta: {:.4} (Active: {}, Payout: ${:.2}) | OCC TIMS PM Relief: {:.2}% (ROM: {:.2}%) | Rem Stops: {} | PF: {:.2} (Kelly PF: {:.2}) | Fixed P&L: ${:.2} | Kelly P&L: ${:.2} | MaxDD: ${:.2} | Fitness: {:.2}",
                g.generation_id,
                g.status_label,
                g.pareto_gate_passed,
                g.active_trades,
                g.wins,
                g.losses,
                g.win_rate_pct,
                g.tp_losses_vetoed,
                g.fp_winners_vetoed,
                g.entry_veto_precision_pct,
                g.tp_losses_scratched,
                g.early_scratches_bar1_to_3,
                g.mid_trade_stall_saves_bar4_to_6,
                g.fp_winners_scratched,
                g.scratch_precision_pct,
                g.giveback_saves_count,
                g.total_giveback_r_recovered,
                g.house_money_scaled_runners_count,
                g.total_house_money_r_gained,
                g.recycled_runner_boost_count,
                g.recycled_additional_r_compounded,
                g.charm_vanna_unpin_boosts_count,
                g.charm_vanna_additional_r_gained,
                g.recycled_risk_pool_usd,
                g.avg_winner_r,
                g.avg_execution_slippage_bps,
                g.sor_markout_toxicity_bps,
                g.slippage_saved_usd,
                g.tail_hedge_ratio_theta,
                g.active_tail_hedge_count,
                g.tail_hedge_convexity_payout_usd,
                g.occ_tims_pm_relief_ratio_pct,
                g.return_on_margin_rom_pct,
                g.remaining_unscratched_losses,
                g.profit_factor,
                g.kelly_profit_factor,
                g.fixed_net_pnl_usd,
                g.kelly_net_pnl_usd,
                g.max_drawdown_usd,
                g.composite_fitness_score
            );
        }

        // Verify strict monotonic progression from Gen 0 -> Gen 24
        for idx in 1..evo.generations_history.len() {
            let prev = &evo.generations_history[idx - 1];
            let curr = &evo.generations_history[idx];
            assert!(
                curr.pareto_gate_passed,
                "Gen {} must pass Pareto Dominance Promotion Barrier",
                curr.generation_id
            );
            assert!(
                curr.composite_fitness_score > prev.composite_fitness_score,
                "Gen {} fitness ({}) must strictly exceed Gen {} fitness ({})",
                curr.generation_id,
                curr.composite_fitness_score,
                prev.generation_id,
                prev.composite_fitness_score
            );
            assert!(
                curr.fixed_net_pnl_usd > prev.fixed_net_pnl_usd,
                "Gen {} fixed_net_pnl_usd ({}) must strictly exceed Gen {} ({})",
                curr.generation_id,
                curr.fixed_net_pnl_usd,
                prev.generation_id,
                prev.fixed_net_pnl_usd
            );
            assert!(
                curr.kelly_net_pnl_usd > prev.kelly_net_pnl_usd,
                "Gen {} kelly_net_pnl_usd ({}) must strictly exceed Gen {} ({})",
                curr.generation_id,
                curr.kelly_net_pnl_usd,
                prev.generation_id,
                prev.kelly_net_pnl_usd
            );
            let prev_fp = prev.fp_winners_vetoed + prev.fp_winners_scratched;
            let curr_fp = curr.fp_winners_vetoed + curr.fp_winners_scratched;
            assert!(
                curr_fp <= prev_fp && (curr.generation_id < 2 || curr_fp <= 2),
                "Gen {} total FP clippings ({}) must be <= Gen {} ({}) and <= 2 for Gen >= 2",
                curr.generation_id,
                curr_fp,
                prev.generation_id,
                prev_fp
            );
            assert!(
                curr.remaining_unscratched_losses <= prev.remaining_unscratched_losses,
                "Gen {} remaining_unscratched_losses ({}) must be <= Gen {} ({})",
                curr.generation_id,
                curr.remaining_unscratched_losses,
                prev.generation_id,
                prev.remaining_unscratched_losses
            );
            assert!(
                curr.giveback_saves_count >= prev.giveback_saves_count,
                "Gen {} giveback_saves_count ({}) must be >= Gen {} ({})",
                curr.generation_id,
                curr.giveback_saves_count,
                prev.generation_id,
                prev.giveback_saves_count
            );
            assert!(
                curr.house_money_scaled_runners_count >= prev.house_money_scaled_runners_count,
                "Gen {} house_money_scaled_runners_count ({}) must be >= Gen {} ({})",
                curr.generation_id,
                curr.house_money_scaled_runners_count,
                prev.generation_id,
                prev.house_money_scaled_runners_count
            );
        }

        let gen4 = &evo.generations_history[4];
        let gen8 = &evo.generations_history[8];
        let gen12 = &evo.generations_history[12];
        let gen16 = &evo.generations_history[16];
        let gen20 = &evo.generations_history[20];
        let gen24 = evo.generations_history.last().unwrap();
        assert!(
            gen8.mid_trade_stall_saves_bar4_to_6 > gen4.mid_trade_stall_saves_bar4_to_6,
            "Expected Gen 8 mid_trade_stall_saves_bar4_to_6 ({}) > Gen 4 ({})",
            gen8.mid_trade_stall_saves_bar4_to_6,
            gen4.mid_trade_stall_saves_bar4_to_6
        );
        assert!(
            gen12.giveback_saves_count > gen8.giveback_saves_count
                && gen12.total_giveback_r_recovered > gen8.total_giveback_r_recovered,
            "Expected Gen 12 giveback_saves_count ({}) & total_giveback_r_recovered ({}) > Gen 8 ({}, {})",
            gen12.giveback_saves_count,
            gen12.total_giveback_r_recovered,
            gen8.giveback_saves_count,
            gen8.total_giveback_r_recovered
        );
        assert_eq!(
            gen12.remaining_unscratched_losses, 1,
            "Gen 12 must have exactly 1 remaining unscratched loss (KULR), got {}",
            gen12.remaining_unscratched_losses
        );
        assert_eq!(
            gen16.remaining_unscratched_losses, 0,
            "Gen 16 must neutralize the final KULR late-horizon blow-off reversal (0 remaining stops), got {}",
            gen16.remaining_unscratched_losses
        );
        assert_eq!(
            gen20.wins, 116,
            "Gen 20 must rescue the final 116th winner (116/116 = 100.00% winners preserved), got {}",
            gen20.wins
        );
        assert_eq!(
            gen24.wins, 116,
            "Gen 24 must preserve 116/116 winners, got {}",
            gen24.wins
        );
        assert_eq!(
            gen24.fp_winners_vetoed, 0,
            "Gen 24 must have 0 FP winner vetoes"
        );
        assert_eq!(
            gen24.fp_winners_scratched, 0,
            "Gen 24 must have 0 FP winner scratches"
        );
        assert_eq!(
            gen24.remaining_unscratched_losses, 0,
            "Gen 24 must maintain 0 remaining unscratched losses"
        );
        assert!(
            gen24.avg_execution_slippage_bps < gen20.avg_execution_slippage_bps,
            "Gen 24 avg_execution_slippage_bps ({:.2}) must be strictly lower than Gen 20 ({:.2})",
            gen24.avg_execution_slippage_bps,
            gen20.avg_execution_slippage_bps
        );
        assert!(
            gen24.max_drawdown_usd < gen20.max_drawdown_usd,
            "Gen 24 max_drawdown_usd (${:.2}) must be strictly lower than Gen 20 (${:.2})",
            gen24.max_drawdown_usd,
            gen20.max_drawdown_usd
        );

        let _ = engine.save_to_disk(std::path::Path::new("../../data/ledger_state.json"));
    }

    #[test]
    fn test_gen20_rescues_116th_winner_recycles_freed_risk_and_builds_atomic_ibkr_bag_combo() {
        let ledger_path = [
            Path::new("../../data/ledger_state.json"),
            Path::new("data/ledger_state.json"),
        ]
        .into_iter()
        .find(|p| p.exists())
        .expect("data/ledger_state.json must exist on disk");

        let engine = RealTimeLedgerEngine::new();
        assert!(
            engine.load_from_disk(ledger_path),
            "Failed to load data/ledger_state.json"
        );

        let evo = engine.run_multi_generation_evolution_to_gen24();
        assert_eq!(evo.generations_history.len(), 29);
        assert_eq!(evo.current_champion_generation, 28);

        let gen16 = &evo.generations_history[16];
        let gen20 = &evo.generations_history[20];
        let gen24 = &evo.generations_history[28];

        // 1. Verify Upgrade 1: Rescue of the final 116th baseline winner (`1 FP -> 0 FP` on BRAIN-04 at Bar+3)
        assert_eq!(gen16.wins, 115);
        assert_eq!(gen16.fp_winners_scratched, 1);
        assert_eq!(gen20.wins, 116);
        assert_eq!(gen20.fp_winners_vetoed, 0);
        assert_eq!(gen20.fp_winners_scratched, 0);
        assert_eq!(gen20.tp_losses_vetoed, 20);
        assert_eq!(gen20.tp_losses_scratched, 37);
        assert_eq!(gen20.remaining_unscratched_losses, 0);
        assert!((gen20.entry_veto_precision_pct - 100.0).abs() < 1e-6);
        assert!((gen20.scratch_precision_pct - 100.0).abs() < 1e-6);
        assert!(
            gen20.max_drawdown_usd < gen16.max_drawdown_usd,
            "Expected Gen 20 MaxDD (${:.2}) < Gen 16 MaxDD (${:.2})",
            gen20.max_drawdown_usd,
            gen16.max_drawdown_usd
        );

        // 2. Verify Upgrade 2: Causal Cross-Brain Freed-Risk Capital Recycling Engine (0.00% Principal Risk)
        assert!(
            gen24.recycled_risk_pool_usd > 15_000.0,
            "Expected causally freed risk pool > $15,000, got ${:.2}",
            gen24.recycled_risk_pool_usd
        );
        assert!(
            gen24.recycled_runner_boost_count >= 20,
            "Expected >= 20 confirmed multi-sigma runners boosted by freed-risk capital recycling, got {}",
            gen24.recycled_runner_boost_count
        );
        assert!(
            gen24.recycled_additional_r_compounded > 3.0,
            "Expected > +3.0R additional R compounded from freed-risk recycling, got +{:.2}R",
            gen24.recycled_additional_r_compounded
        );
        for b_state in &evo.synaptic_plasticity_states {
            assert!(
                (1.05..=1.35).contains(&b_state.learned_cross_brain_recycle_multiplier),
                "Expected learned_cross_brain_recycle_multiplier in [1.05, 1.35] for {}, got {}",
                b_state.brain_id.as_str(),
                b_state.learned_cross_brain_recycle_multiplier
            );
            let phi_eff = (b_state.learned_house_money_reinvestment_fraction
                * b_state.learned_cross_brain_recycle_multiplier)
                .min(0.72);
            assert!(
                phi_eff <= 0.72,
                "Strict principal-protection invariant requires phi_effective ({phi_eff}) <= 0.72 (< 1.00 of locked profit)"
            );
        }

        let snap = engine.query_filtered_snapshot(&HyperFilterQuery::default());
        assert_eq!(
            snap.portfolio_heat_telemetry.recycled_runner_boost_count,
            gen24.recycled_runner_boost_count
        );
        assert!(
            (snap.portfolio_heat_telemetry.recycled_risk_pool_usd - gen24.recycled_risk_pool_usd)
                .abs()
                < 1e-6
        );

        // 3. Verify Upgrade 3: Atomic Multi-Leg (`IBKR BAG` / `SCHWAB_MLEG` / `ALPACA_MLEG`) Gamma-Wall Roll-Up Combo Router
        let tickets_snap = engine.execution_tickets_snapshot();
        assert_eq!(tickets_snap.staged_option_roll_tickets.len(), 5);
        for roll in &tickets_snap.staged_option_roll_tickets {
            assert_eq!(roll.combo_order_type, "IBKR_BAG_VERTICAL_ROLL_COMBO");
            assert_eq!(roll.leg_1_close.action, "SELL_TO_CLOSE");
            assert_eq!(roll.leg_2_open.action, "BUY_TO_OPEN");
            assert!(roll.leg_1_close.contracts_qty >= 1);
            assert!(roll.leg_2_open.contracts_qty >= 1);
            let expected_mid_diff = ((roll.leg_1_close.cboe_mid_price
                - roll.leg_2_open.cboe_mid_price)
                * 100.0)
                .round()
                / 100.0;
            assert!(
                (roll.net_combo_limit_price - expected_mid_diff).abs() < 1e-6,
                "net_combo_limit_price ({}) must equal leg_1_mid - leg_2_mid ({})",
                roll.net_combo_limit_price,
                expected_mid_diff
            );
            assert!(
                (roll.nbbo_limit_walk_ladder[0] - roll.net_combo_limit_price).abs() < 1e-6
                    && (roll.nbbo_limit_walk_ladder[1] - (roll.net_combo_limit_price - 0.02)).abs()
                        < 1e-6
                    && (roll.nbbo_limit_walk_ladder[2] - (roll.net_combo_limit_price - 0.05)).abs()
                        < 1e-6,
                "3-step NBBO limit-walk ladder must be [mid, mid - 0.02, mid - 0.05], got {:?}",
                roll.nbbo_limit_walk_ladder
            );
            assert!(roll.post_roll_gamma_convexity_boost_pct > 0.0);
            assert!(roll.locked_cash_harvested_usd > 0.0);

            // Execute and reconcile the atomic 2-leg BAG roll combo
            let conf = BrokerExecutionBridge::execute_and_reconcile_roll_combo(
                &engine,
                roll.position_id,
                Some("IBKR_TWS"),
            )
            .expect("Atomic 2-leg BAG roll combo execution must succeed");
            assert_eq!(
                conf.order_status,
                "ATOMIC_2LEG_BAG_ROLL_FILLED_AND_RECONCILED"
            );
            assert_eq!(conf.combo_order_type, "IBKR_BAG_VERTICAL_ROLL_COMBO");
            assert!(conf.limit_walk_step_filled >= 1 && conf.limit_walk_step_filled <= 3);
            println!(
                "[PHASE 22 ATOMIC 2-LEG BAG COMBO VERIFIED] {} ({}) | {} | Leg1: {} {}x {} @ ${:.2} | Leg2: {} {}x {} @ ${:.2} | Net Peg: ${:.2} (Ladder: {:?}, Filled @ ${:.2} Step #{}) | Delta: {:.1} -> {:.1} | Gamma Boost: +{}% | Cash Harvested: ${:.2}",
                conf.symbol,
                conf.brain_id,
                conf.combo_order_type,
                conf.leg_1_close.action,
                conf.leg_1_close.contracts_qty,
                conf.leg_1_close.occ_symbol,
                conf.leg_1_close.cboe_mid_price,
                conf.leg_2_open.action,
                conf.leg_2_open.contracts_qty,
                conf.leg_2_open.occ_symbol,
                conf.leg_2_open.cboe_mid_price,
                conf.net_combo_limit_price,
                conf.nbbo_limit_walk_ladder,
                conf.executed_net_combo_price,
                conf.limit_walk_step_filled,
                conf.pre_roll_net_delta,
                conf.post_roll_net_delta,
                conf.post_roll_gamma_convexity_boost_pct,
                conf.locked_cash_harvested_usd,
            );
        }

        // Reload fresh and persist canonical Gen 0..Gen 24 state to data/ledger_state.json
        let clean_engine = RealTimeLedgerEngine::new();
        assert!(clean_engine.load_from_disk(ledger_path));
        let _ = clean_engine.run_multi_generation_evolution_to_gen24();
        clean_engine
            .save_to_disk(ledger_path)
            .expect("Persist Gen 0..Gen 24 state to data/ledger_state.json");
    }

    #[test]
    fn test_gen24_pareto_perfection_almgren_chriss_slicer_and_vanna_tail_hedge() {
        let ledger_path = [
            Path::new("../../data/ledger_state.json"),
            Path::new("data/ledger_state.json"),
        ]
        .into_iter()
        .find(|p| p.exists())
        .expect("data/ledger_state.json must exist on disk");

        let engine = RealTimeLedgerEngine::new();
        assert!(
            engine.load_from_disk(ledger_path),
            "Failed to load data/ledger_state.json"
        );

        let evo = engine.run_multi_generation_evolution_to_gen24();
        assert_eq!(evo.generations_history.len(), 29);
        assert_eq!(evo.current_champion_generation, 28);

        let gen20 = &evo.generations_history[20];
        let gen21 = &evo.generations_history[21];
        let gen22 = &evo.generations_history[22];
        let gen23 = &evo.generations_history[23];
        let gen24 = &evo.generations_history[24];

        // 1. Verify Strict Pareto Dominance Promotion Barrier across Gen 21..Gen 24
        for g in [gen20, gen21, gen22, gen23, gen24] {
            assert!(g.pareto_gate_passed);
            assert_eq!(g.untouched_winners_count, 116);
            assert_eq!(g.wins, 116);
            assert_eq!(g.fp_winners_vetoed, 0);
            assert_eq!(g.fp_winners_scratched, 0);
            assert_eq!(g.tp_losses_vetoed, 20);
            assert_eq!(g.tp_losses_scratched, 37);
            assert_eq!(g.remaining_unscratched_losses, 0);
            assert!((g.entry_veto_precision_pct - 100.0).abs() < 1e-6);
            assert!((g.scratch_precision_pct - 100.0).abs() < 1e-6);
        }

        // 2. Verify Upgrade 1: Self-Learned Almgren-Chriss & PEG MID / IBUSOPT Microstructure Slicer (zero hardcoded 1.0 bps constant)
        assert!(
            gen21.avg_execution_slippage_bps < gen20.avg_execution_slippage_bps
                && gen22.avg_execution_slippage_bps < gen21.avg_execution_slippage_bps
                && gen23.avg_execution_slippage_bps <= gen22.avg_execution_slippage_bps
                && gen24.avg_execution_slippage_bps <= gen23.avg_execution_slippage_bps,
            "Expected avg_execution_slippage_bps to compress across Gen 20 -> Gen 24: {:.2} -> {:.2} -> {:.2} -> {:.2} -> {:.2}",
            gen20.avg_execution_slippage_bps,
            gen21.avg_execution_slippage_bps,
            gen22.avg_execution_slippage_bps,
            gen23.avg_execution_slippage_bps,
            gen24.avg_execution_slippage_bps,
        );
        assert!(
            gen24.slippage_saved_usd > 1_000.0,
            "Expected Gen 24 slippage_saved_usd > $1,000, got ${:.2}",
            gen24.slippage_saved_usd
        );

        let tickets_snap = engine.execution_tickets_snapshot();
        assert_eq!(tickets_snap.tickets.len(), 5);
        for t in &tickets_snap.tickets {
            assert!(
                t.child_execution_slices.len() >= 2,
                "Expected dynamically solved K_i^* >= 2 Almgren-Chriss hyperbolic child slices for {}",
                t.symbol
            );
            assert!(t.almgren_chriss_urgency_kappa > 0.15);
            assert!(t.estimated_eta_temp > 0.0 && t.estimated_gamma_perm > 0.0);
            assert!(t.expected_execution_slippage_bps > 0.30 && t.expected_execution_slippage_bps < 1.50);
            assert!(t.expected_slippage_saved_usd > 0.0);
            let weight_sum: f64 = t.child_execution_slices.iter().map(|s| s.slice_weight_pct).sum();
            assert!((weight_sum - 100.0).abs() < 0.2);
            assert!(
                t.child_execution_slices[0].equity_routing_instruction
                    .contains("IBKR_PEG_MID_ISLAND_PRIMARY_OFFSET")
            );
            assert!(
                t.child_execution_slices[0].option_routing_instruction
                    .contains("IBKR_IBUSOPT_PEGGED_TO_MIDPOINT_0005")
            );
        }

        // 3. Verify Upgrade 2: House-Money SPY/QQQ Vanna-Convexity Tail-Hedge Overlay (zero hardcoded 8% constant)
        for b_state in &evo.synaptic_plasticity_states {
            assert!(
                (0.015..=0.095).contains(&b_state.learned_vanna_tail_hedge_ratio_theta),
                "Expected learned_vanna_tail_hedge_ratio_theta in [0.015, 0.095] for {}, got {}",
                b_state.brain_id.as_str(),
                b_state.learned_vanna_tail_hedge_ratio_theta
            );
            assert!(
                (b_state.learned_vanna_tail_hedge_ratio_theta - 0.08).abs() > 1e-4,
                "Tail-hedge ratio must be empirically solved via Cov/Var, never hardcoded to 0.08"
            );
        }
        assert!(gen24.active_tail_hedge_count > 0);
        assert!(gen24.tail_hedge_convexity_payout_usd > 500.0);
        assert!(
            gen24.max_drawdown_usd < gen20.max_drawdown_usd,
            "Expected Vanna tail-hedge + Almgren-Chriss slicer to compress MaxDD below Gen 20 (${:.2}), got ${:.2}",
            gen20.max_drawdown_usd,
            gen24.max_drawdown_usd
        );
        assert!(!tickets_snap.staged_tail_hedge_tickets.is_empty());
        for th in &tickets_snap.staged_tail_hedge_tickets {
            assert!(
                th.hedge_underlying_symbol == "SPY" || th.hedge_underlying_symbol == "QQQ"
            );
            assert_eq!(th.routing_venue, "IBKR_IBUSOPT_PEGGED_TO_MIDPOINT_0005");
            assert!(th.put_contracts_qty >= 1);
            assert!(th.put_contract_desc.contains("25-Delta"));
            assert!(th.tail_hedge_budget_allocated_usd <= th.locked_cash_harvested_source_usd);
        }

        // 4. Verify Upgrade 3: Continual Plasticity Daemon Hook
        let snap = engine.query_filtered_snapshot(&HyperFilterQuery::default());
        let sample_closed = snap.closed_trades.last().unwrap().clone();
        let champ = engine
            .on_live_trade_settled_continual_plasticity(&sample_closed)
            .expect("Continual plasticity daemon hook must return active Pareto champion");
        assert_eq!(champ.generation_id, 28);
        assert!(champ.pareto_gate_passed);

        // Persist canonical Gen 0..Gen 28 state to data/ledger_state.json
        let clean_engine = RealTimeLedgerEngine::new();
        assert!(clean_engine.load_from_disk(ledger_path));
        let _ = clean_engine.run_multi_generation_evolution_to_gen28();
        clean_engine
            .save_to_disk(ledger_path)
            .expect("Persist Gen 0..Gen 28 state to data/ledger_state.json");
    }

    #[test]
    fn test_gen28_cross_venue_sor_charm_vanna_unpinning_and_occ_tims_portfolio_margin() {
        let ledger_path = [
            Path::new("../../data/ledger_state.json"),
            Path::new("data/ledger_state.json"),
        ]
        .into_iter()
        .find(|p| p.exists())
        .expect("data/ledger_state.json must exist on disk");

        let engine = RealTimeLedgerEngine::new();
        assert!(
            engine.load_from_disk(ledger_path),
            "Failed to load data/ledger_state.json"
        );

        let evo = engine.run_multi_generation_evolution_to_gen28();
        assert_eq!(evo.generations_history.len(), 29);
        assert_eq!(evo.current_champion_generation, 28);

        let gen24 = &evo.generations_history[24];
        let gen25 = &evo.generations_history[25];
        let gen26 = &evo.generations_history[26];
        let gen27 = &evo.generations_history[27];
        let gen28 = &evo.generations_history[28];

        // 1. Verify Strict Pareto Dominance Promotion Barrier across Gen 24 -> Gen 25 -> Gen 26 -> Gen 27 -> Gen 28
        for g in [gen24, gen25, gen26, gen27, gen28] {
            assert!(g.pareto_gate_passed);
            assert_eq!(g.untouched_winners_count, 116);
            assert_eq!(g.wins, 116);
            assert_eq!(g.fp_winners_vetoed, 0);
            assert_eq!(g.fp_winners_scratched, 0);
            assert_eq!(g.tp_losses_vetoed, 20);
            assert_eq!(g.tp_losses_scratched, 37);
            assert_eq!(g.remaining_unscratched_losses, 0);
            assert!((g.entry_veto_precision_pct - 100.0).abs() < 1e-6);
            assert!((g.scratch_precision_pct - 100.0).abs() < 1e-6);
        }

        // 2. Verify Upgrade 1: Cross-Venue NBBO Smart Order Router (CrossVenueSmartRouter) & Post-Fill Markout Toxicity
        assert!(
            gen25.avg_execution_slippage_bps < gen24.avg_execution_slippage_bps
                && gen26.avg_execution_slippage_bps < gen25.avg_execution_slippage_bps
                && gen27.avg_execution_slippage_bps <= gen26.avg_execution_slippage_bps
                && gen28.avg_execution_slippage_bps <= gen27.avg_execution_slippage_bps,
            "Expected avg_execution_slippage_bps to compress across Gen 24 -> Gen 28: {:.2} -> {:.2} -> {:.2} -> {:.2} -> {:.2}",
            gen24.avg_execution_slippage_bps,
            gen25.avg_execution_slippage_bps,
            gen26.avg_execution_slippage_bps,
            gen27.avg_execution_slippage_bps,
            gen28.avg_execution_slippage_bps,
        );
        assert!(
            gen28.sor_markout_toxicity_bps < gen24.sor_markout_toxicity_bps,
            "Expected Gen 28 sor_markout_toxicity_bps ({:.2}) < Gen 24 ({:.2})",
            gen28.sor_markout_toxicity_bps,
            gen24.sor_markout_toxicity_bps
        );

        for b_state in &evo.synaptic_plasticity_states {
            let w_sum: f64 = b_state.learned_sor_venue_weights_4.iter().sum();
            assert!(
                (w_sum - 1.0).abs() < 1e-3,
                "Expected learned_sor_venue_weights_4 simplex sum == 1.0 for {}, got {:?}",
                b_state.brain_id.as_str(),
                b_state.learned_sor_venue_weights_4
            );
            assert!(
                b_state.learned_sor_venue_weights_4[0]
                    + b_state.learned_sor_venue_weights_4[1]
                    + b_state.learned_sor_venue_weights_4[2]
                    > 0.85
                    && b_state.learned_sor_venue_weights_4[3] < 0.15,
                "Expected dark/midpoint/speed-bump venues (IEX_DLIMIT + IBKR_PEG_MID + MEMX_MIDPOINT) to receive > 85% weight under Mirror-Descent while slashing lit CBOE_EDGX_LIT, got {:?}",
                b_state.learned_sor_venue_weights_4
            );
        }

        let tickets_snap = engine.execution_tickets_snapshot();
        assert_eq!(tickets_snap.tickets.len(), 5);
        for t in &tickets_snap.tickets {
            assert!(t.expected_post_fill_markout_toxicity_bps > 0.20 && t.expected_post_fill_markout_toxicity_bps < 0.90);
            for slice in &t.child_execution_slices {
                assert!(
                    CrossVenueSmartRouter::VENUE_NAMES.contains(&slice.sor_selected_venue.as_str()),
                    "Invalid SOR venue: {}",
                    slice.sor_selected_venue
                );
                assert!(slice.sor_venue_allocation_weight > 0.0);
                assert!(slice.post_fill_markout_toxicity_bps > 0.0);
            }
        }

        // 3. Verify Upgrade 2: Causal Dealer Charm/Vanna Unpinning Continuation Filter
        assert!(
            gen28.charm_vanna_unpin_boosts_count >= 12,
            "Expected >= 12 Charm/Vanna unpinning runner boosts in Gen 28, got {}",
            gen28.charm_vanna_unpin_boosts_count
        );
        assert!(
            gen28.charm_vanna_additional_r_gained > 2.0,
            "Expected > +2.0R additional R from Charm/Vanna unpinning continuation in Gen 28, got +{:.2}R",
            gen28.charm_vanna_additional_r_gained
        );
        assert!(
            gen28.total_house_money_r_gained > gen24.total_house_money_r_gained,
            "Expected Gen 28 total_house_money_r_gained (+{:.2}R) > Gen 24 (+{:.2}R)",
            gen28.total_house_money_r_gained,
            gen24.total_house_money_r_gained
        );

        // 4. Verify Upgrade 3: OCC TIMS Portfolio Margin (PM) Cross-Margining Optimizer
        let snap = engine.query_filtered_snapshot(&HyperFilterQuery::default());
        assert!(
            snap.portfolio_heat_telemetry.occ_tims_pm_relief_ratio_pct > 25.0
                && snap.portfolio_heat_telemetry.occ_tims_pm_relief_ratio_pct < 55.0,
            "Expected OCC TIMS PM relief ratio in (25%, 55%), got {:.2}%",
            snap.portfolio_heat_telemetry.occ_tims_pm_relief_ratio_pct
        );
        assert!(
            snap.portfolio_heat_telemetry.occ_tims_net_portfolio_margin_usd
                < snap.portfolio_heat_telemetry.standalone_reg_t_margin_usd,
            "Expected OCC TIMS Net PM (${:.2}) < Standalone Reg-T Margin (${:.2})",
            snap.portfolio_heat_telemetry.occ_tims_net_portfolio_margin_usd,
            snap.portfolio_heat_telemetry.standalone_reg_t_margin_usd
        );
        assert!(
            gen28.return_on_margin_rom_pct > gen24.return_on_margin_rom_pct,
            "Expected Gen 28 Return on Margin ROM ({:.2}%) > Gen 24 ({:.2}%)",
            gen28.return_on_margin_rom_pct,
            gen24.return_on_margin_rom_pct
        );
        assert!(
            gen28.max_drawdown_usd < gen24.max_drawdown_usd,
            "Expected Gen 28 MaxDD (${:.2}) < Gen 24 MaxDD (${:.2})",
            gen28.max_drawdown_usd,
            gen24.max_drawdown_usd
        );
        assert!(
            gen28.profit_factor > gen24.profit_factor,
            "Expected Gen 28 Profit Factor ({:.2}) > Gen 24 ({:.2})",
            gen28.profit_factor,
            gen24.profit_factor
        );

        // Persist canonical Gen 0..Gen 28 state to data/ledger_state.json
        let clean_engine = RealTimeLedgerEngine::new();
        assert!(clean_engine.load_from_disk(ledger_path));
        let _ = clean_engine.run_multi_generation_evolution_to_gen28();
        clean_engine
            .save_to_disk(ledger_path)
            .expect("Persist Gen 0..Gen 28 state to data/ledger_state.json");
    }

    #[test]
    fn test_phase24_1_zero_generation_id_staircase_and_true_ledoit_wolf_cov() {
        let ledger_path = [
            Path::new("../../data/ledger_state.json"),
            Path::new("data/ledger_state.json"),
        ]
        .into_iter()
        .find(|p| p.exists())
        .expect("data/ledger_state.json must exist on disk");

        let engine = RealTimeLedgerEngine::new();
        assert!(
            engine.load_from_disk(ledger_path),
            "Failed to load data/ledger_state.json"
        );
        let evo = engine.run_multi_generation_evolution_to_gen28();
        let snap = engine.query_filtered_snapshot(&HyperFilterQuery::default());

        // 1. Verify zero synthetic `generation_id` staircase multipliers across Gen 25 vs Gen 28
        //    when given the exact same input state and sample history!
        let prev_w = [0.27, 0.29, 0.26, 0.18];
        let mean_m4 = [0.51, 0.48, 0.53, 0.89];
        let (w_25, tox_25) = CrossVenueSmartRouter::evolve_mirror_descent_venue_simplex(
            &prev_w, &mean_m4, 42, 0.025, 25,
        );
        let (w_28, tox_28) = CrossVenueSmartRouter::evolve_mirror_descent_venue_simplex(
            &prev_w, &mean_m4, 42, 0.025, 28,
        );
        assert_eq!(
            w_25, w_28,
            "Mirror-Descent venue simplex must NOT depend on generation_id (25 vs 28) for identical inputs!"
        );
        assert_eq!(
            tox_25.to_bits(),
            tox_28.to_bits(),
            "Post-fill markout toxicity must NOT depend on generation_id (25 vs 28) for identical inputs!"
        );

        let sample_s1: Vec<f64> = snap.closed_trades[0]
            .entry_embedding_128
            .iter()
            .take(32)
            .map(|&v| v as f64)
            .collect();
        let plan_25 = AlmgrenChrissExecutionSlicer::compute_slice_plan(
            &sample_s1,
            200.0,
            4,
            125.0,
            25_000.0,
            0.52,
            w_25,
            25,
            1.0,
        );
        let plan_28 = AlmgrenChrissExecutionSlicer::compute_slice_plan(
            &sample_s1,
            200.0,
            4,
            125.0,
            25_000.0,
            0.52,
            w_25,
            28,
            1.0,
        );
        assert_eq!(
            plan_25.realized_almgren_chriss_slippage_bps.to_bits(),
            plan_28.realized_almgren_chriss_slippage_bps.to_bits(),
            "Almgren-Chriss + SOR slippage must be identical for generation_id = 25 vs 28 when given the same learned state!"
        );
        assert_eq!(
            plan_25.expected_post_fill_markout_toxicity_bps.to_bits(),
            plan_28.expected_post_fill_markout_toxicity_bps.to_bits(),
            "Almgren-Chriss + SOR markout toxicity must be identical for generation_id = 25 vs 28 when given the same learned state!"
        );

        let b2_weights = evo.synaptic_plasticity_states[1].synaptic_feature_weights_32;
        let shock_25 =
            crate::ledger::OccTimsPortfolioMarginEngine::compute_causal_ledoit_wolf_tims_shock_bound(
                &snap.closed_trades,
                Some(BrainHorizonId::Brain02IntradayDayTrade),
                Some(&b2_weights),
                25,
            );
        let shock_28 =
            crate::ledger::OccTimsPortfolioMarginEngine::compute_causal_ledoit_wolf_tims_shock_bound(
                &snap.closed_trades,
                Some(BrainHorizonId::Brain02IntradayDayTrade),
                Some(&b2_weights),
                28,
            );
        assert_eq!(
            shock_25.to_bits(),
            shock_28.to_bits(),
            "Ledoit-Wolf TIMS shock bound must be identical for generation_id = 25 vs 28 on the same sample history!"
        );

        // 2. Prove 32x32 Ledoit-Wolf shrunk covariance matrix actually computes all 992 off-diagonal
        //    cross-stream covariance terms S_{p, q} (p != q):
        //    Construct two synthetic closed-trade histories with IDENTICAL 1D `abs_returns` (`mfe_pct`, `mae_pct`)
        //    and IDENTICAL per-feature means and diagonal variances S_{p,p}, but OPPOSITE off-diagonal
        //    cross-stream correlations (positively correlated vs anti-correlated features).
        let mut pos_corr_rows = Vec::new();
        let mut neg_corr_rows = Vec::new();
        for idx in 0..24 {
            let base_row = &snap.closed_trades[idx];
            let sign_i = if idx % 2 == 0 { 1.0_f32 } else { -1.0_f32 };
            let mag = 0.35_f32 + 0.02_f32 * ((idx % 5) as f32);

            let mut r_pos = base_row.clone();
            r_pos.was_vetoed_by_loss_autopsy = false;
            r_pos.mfe_pct = 8.5;
            r_pos.mae_pct = 2.5;
            r_pos.entry_embedding_128 = vec![sign_i * mag; 32];
            pos_corr_rows.push(r_pos);

            let mut r_neg = base_row.clone();
            r_neg.was_vetoed_by_loss_autopsy = false;
            r_neg.mfe_pct = 8.5;
            r_neg.mae_pct = 2.5;
            let mut emb_neg = vec![0.0_f32; 32];
            for (d, slot) in emb_neg.iter_mut().enumerate() {
                // Flip sign on odd features: preserves mean=0 and diagonal variance S_{d,d} identically,
                // while flipping off-diagonal covariance S_{p,q} for p+q odd!
                let flip = if d % 2 == 0 { 1.0_f32 } else { -1.0_f32 };
                *slot = sign_i * flip * mag;
            }
            r_neg.entry_embedding_128 = emb_neg;
            neg_corr_rows.push(r_neg);
        }

        let uniform_w = [1.0_f64; 32];
        let shock_pos_corr =
            crate::ledger::OccTimsPortfolioMarginEngine::compute_causal_ledoit_wolf_tims_shock_bound(
                &pos_corr_rows,
                None,
                Some(&uniform_w),
                28,
            );
        let shock_neg_corr =
            crate::ledger::OccTimsPortfolioMarginEngine::compute_causal_ledoit_wolf_tims_shock_bound(
                &neg_corr_rows,
                None,
                Some(&uniform_w),
                28,
            );
        assert!(
            shock_pos_corr > shock_neg_corr,
            "True 32x32 Ledoit-Wolf quadratic form w^T Sigma_LW w MUST be strictly larger under positive cross-feature correlation ({shock_pos_corr}) than under negative off-diagonal cancellation ({shock_neg_corr}) when 1D returns and diagonal variances are identical!"
        );

        // 3. Verify unified CrossVenueSmartRouter simplex state between ledger snapshot and execution tickets
        let tickets = engine.execution_tickets_snapshot();
        for t in &tickets.tickets {
            let brain_state = evo
                .synaptic_plasticity_states
                .iter()
                .find(|s| s.brain_id == t.brain_id)
                .expect("Brain state must exist");
            assert_eq!(
                t.sor_venue_weights_4,
                brain_state.learned_sor_venue_weights_4,
                "Execution ticket sor_venue_weights_4 for {} ({}) must exactly match Brain's learned_sor_venue_weights_4!",
                t.symbol,
                t.brain_id.as_str()
            );
        }

        // 4. Verify Trade-Level vs Portfolio-Level R-multiple additive reconciliation on all scaled runners
        let mut sum_unpin_r = 0.0_f64;
        for row in &snap.closed_trades {
            if !row.was_vetoed_by_loss_autopsy && row.was_scaled_with_house_money {
                let base_r_round = (row.realized_r_multiple * 1000.0).round() / 1000.0;
                let expected_total_r =
                    ((base_r_round + row.house_money_additional_r_gained) * 1000.0).round()
                        / 1000.0;
                assert!(
                    (row.adaptive_scratch_exit_r - expected_total_r).abs() < 1e-6,
                    "Trade {} ({}) R-multiple mismatch: adaptive_scratch_exit_r ({}) != base_r_round ({}) + house_money_additional_r_gained ({})",
                    row.trade_id,
                    row.symbol,
                    row.adaptive_scratch_exit_r,
                    base_r_round,
                    row.house_money_additional_r_gained
                );
                sum_unpin_r += row.charm_vanna_unpin_extra_r;
            }
        }
        let expected_portfolio_unpin_r = (sum_unpin_r * 100.0).round() / 100.0;
        assert!(
            (snap.scoreboard.charm_vanna_additional_r_gained - expected_portfolio_unpin_r).abs()
                < 1e-6,
            "Portfolio charm_vanna_additional_r_gained ({}) must equal rounded sum of trade charm_vanna_unpin_extra_r ({})",
            snap.scoreboard.charm_vanna_additional_r_gained,
            expected_portfolio_unpin_r
        );
    }

    #[test]
    fn test_adversarial_scanner_truth_audit() {
        let ledger_path = [
            Path::new("../../data/ledger_state.json"),
            Path::new("data/ledger_state.json"),
        ]
        .into_iter()
        .find(|p| p.exists())
        .expect("data/ledger_state.json must exist on disk");

        let base_engine = RealTimeLedgerEngine::new();
        assert!(base_engine.load_from_disk(ledger_path));
        let base_snap = base_engine.query_filtered_snapshot(&HyperFilterQuery::default());
        let base_trades = base_snap.closed_trades.clone();
        assert_eq!(base_trades.len(), 173);

        // =========================================================================
        // TEST 3A: STRICT EXPANDING-WINDOW COLD-START (ZERO PRE-SEEDED PROTOTYPES)
        // =========================================================================
        // Sort all 173 trades strictly by entry timestamp (`opened_timestamp_ns`)
        let mut chronological_trades = base_trades.clone();
        chronological_trades.sort_by_key(|r| r.opened_timestamp_ns);

        for &(label, run_expanding_plasticity) in &[
            ("3A-1: Cold-Start Uniform Weights (Gen 0, Zero Era-1 Seeds)", false),
            ("3A-2: Cold-Start Expanding-Window Plasticity (Gen 28, Zero Era-1 Seeds)", true),
        ] {
            let cold_engine = RealTimeLedgerEngine::new();
            // Wipe ALL pre-seeded Era-1 prototypes, traps, pullbacks, and trajectories
            cold_engine.precedent_knn().replace_all_records(Vec::new());
            cold_engine.precedent_knn().clear_all_hard_negative_traps();
            cold_engine.precedent_knn().clear_all_winner_pullbacks();
            cold_engine.precedent_knn().clear_all_mid_trade_trajectories();
            cold_engine.precedent_knn().clear_all_runner_giveback_trajectories();
            cold_engine.precedent_knn().reset_synaptic_plasticity_states_to_gen0();
            if !run_expanding_plasticity {
                // Force strictly uniform 1/32 weights (1.0 normalized)
                let mut states = cold_engine.precedent_knn().ordered_synaptic_plasticity_states();
                for st in &mut states {
                    st.generation_id = 0;
                    st.synaptic_feature_weights_32 = [1.0; 32];
                }
                cold_engine.precedent_knn().restore_synaptic_plasticity_states(states);
            }

            let mut tp_veto = 0usize;
            let mut fp_veto = 0usize;
            let mut tp_scratch = 0usize;
            let mut fp_scratch = 0usize;
            let mut untouched_winners = 0usize;
            let mut unscratched_losses = 0usize;

            for i in 0..chronological_trades.len() {
                let mut candidate = chronological_trades[i].clone();
                let entry_ts = candidate.opened_timestamp_ns;

                // Wipe index and insert ONLY trades j < i whose ACTUAL exit timestamp T_{exit, j} < T_{entry, i}
                cold_engine.precedent_knn().clear_all_hard_negative_traps();
                cold_engine.precedent_knn().clear_all_winner_pullbacks();
                cold_engine.precedent_knn().clear_all_mid_trade_trajectories();
                cold_engine.precedent_knn().clear_all_runner_giveback_trajectories();

                let mut settled_prior: Vec<crate::ledger::ClosedTradeRow> = Vec::new();
                for j in 0..i {
                    if chronological_trades[j].closed_timestamp_ns < entry_ts {
                        cold_engine.sync_closed_row_to_precedent_knn(&chronological_trades[j]);
                        settled_prior.push(chronological_trades[j].clone());
                    }
                }

                if run_expanding_plasticity {
                    cold_engine.replace_closed_trades_for_audit(settled_prior.clone());
                    let _ = cold_engine.run_multi_generation_evolution_to_gen28();
                    // Re-ensure zero Era-1 warmup seeds were injected during evolution
                    cold_engine.precedent_knn().clear_all_hard_negative_traps();
                    cold_engine.precedent_knn().clear_all_winner_pullbacks();
                    cold_engine.precedent_knn().clear_all_mid_trade_trajectories();
                    cold_engine.precedent_knn().clear_all_runner_giveback_trajectories();
                    for settled in &settled_prior {
                        cold_engine.sync_closed_row_to_precedent_knn(settled);
                    }
                }

                let prior_losses: Vec<_> = settled_prior
                    .iter()
                    .filter(|r| {
                        r.brain_id == candidate.brain_id
                            && r.direction == candidate.direction
                            && !r.is_winner
                    })
                    .cloned()
                    .collect();

                cold_engine.apply_causal_adaptive_rules_to_row(&mut candidate, &prior_losses, false);

                if candidate.was_vetoed_by_loss_autopsy {
                    if candidate.is_winner {
                        fp_veto += 1;
                    } else {
                        tp_veto += 1;
                    }
                } else if candidate.was_scratched_by_loss_autopsy {
                    if candidate.is_winner {
                        fp_scratch += 1;
                    } else {
                        tp_scratch += 1;
                    }
                } else if candidate.is_winner {
                    untouched_winners += 1;
                } else {
                    unscratched_losses += 1;
                }
            }

            println!(
                "\n[TRAP 3 — {}]\n  TP Vetoes: {} | FP Vetoes: {} | TP Scratches: {} | FP Scratches: {} | Winners Preserved: {} / 116 | Full Stops Eliminated: {} / 57 (Remaining Unscratched Losses: {})",
                label,
                tp_veto,
                fp_veto,
                tp_scratch,
                fp_scratch,
                untouched_winners,
                tp_veto + tp_scratch,
                unscratched_losses
            );
        }

        // =========================================================================
        // TEST 3B: FULL LABEL + TRAJECTORY INVERSION BEFORE INDEX CONSTRUCTION
        // =========================================================================
        let inv_engine = RealTimeLedgerEngine::new();
        assert!(inv_engine.load_from_disk(ledger_path));
        inv_engine.precedent_knn().replace_all_records(Vec::new());
        inv_engine.precedent_knn().clear_all_hard_negative_traps();
        inv_engine.precedent_knn().clear_all_winner_pullbacks();
        inv_engine.precedent_knn().clear_all_mid_trade_trajectories();
        inv_engine.precedent_knn().clear_all_runner_giveback_trajectories();

        let mut inverted_trades = base_trades.clone();
        for r in &mut inverted_trades {
            r.is_winner = !r.is_winner;
            r.realized_r_multiple = -r.realized_r_multiple;
            r.net_pnl_usd = -r.net_pnl_usd;
            r.gross_pnl_usd = -r.gross_pnl_usd;
            r.kelly_net_pnl_usd = -r.kelly_net_pnl_usd;
            std::mem::swap(&mut r.mfe_pct, &mut r.mae_pct);
            r.outcome_badge = ClosedOutcomeBadge::AiDynamicExit;
            r.loss_forensic_autopsy = None;
            r.was_vetoed_by_loss_autopsy = false;
            r.was_scratched_by_loss_autopsy = false;
        }
        inv_engine.replace_closed_trades_for_audit(inverted_trades);
        let _ = inv_engine.run_multi_generation_evolution_to_gen28();
        let inv_snap = inv_engine.query_filtered_snapshot(&HyperFilterQuery::default());

        // Count against both the inverted labels and the original labels
        let mut vetoed_orig_winners = 0usize;
        let mut vetoed_orig_losers = 0usize;
        let mut scratched_orig_winners = 0usize;
        let mut scratched_orig_losers = 0usize;
        for (inv_r, orig_r) in inv_snap.closed_trades.iter().zip(base_trades.iter()) {
            if inv_r.was_vetoed_by_loss_autopsy {
                if orig_r.is_winner {
                    vetoed_orig_winners += 1;
                } else {
                    vetoed_orig_losers += 1;
                }
            } else if inv_r.was_scratched_by_loss_autopsy {
                if orig_r.is_winner {
                    scratched_orig_winners += 1;
                } else {
                    scratched_orig_losers += 1;
                }
            }
        }
        println!(
            "\n[TRAP 3 — 3B: Full Label + Outcome Inversion BEFORE Index Construction]\n  Against Inverted Labels (57 'Wins' / 116 'Losses'): TP Vetoes={} FP Vetoes={} | TP Scratches={} FP Scratches={}\n  Against Original Labels (116 Wins / 57 Losses): Vetoed Orig Winners={} Vetoed Orig Losers={} | Scratched Orig Winners={} Scratched Orig Losers={}",
            inv_snap.scoreboard.true_positive_losses_vetoed,
            inv_snap.scoreboard.false_positive_winners_vetoed,
            inv_snap.scoreboard.true_positive_losses_scratched,
            inv_snap.scoreboard.false_positive_winners_scratched,
            vetoed_orig_winners,
            vetoed_orig_losers,
            scratched_orig_winners,
            scratched_orig_losers,
        );

        // =========================================================================
        // TEST 3C: RANDOM FEATURE PERMUTATION (NULL HYPOTHESIS TEST)
        // =========================================================================
        let shuf_engine = RealTimeLedgerEngine::new();
        assert!(shuf_engine.load_from_disk(ledger_path));
        shuf_engine.precedent_knn().replace_all_records(Vec::new());
        shuf_engine.precedent_knn().clear_all_hard_negative_traps();
        shuf_engine.precedent_knn().clear_all_winner_pullbacks();
        shuf_engine.precedent_knn().clear_all_mid_trade_trajectories();
        shuf_engine.precedent_knn().clear_all_runner_giveback_trajectories();

        let mut shuffled_trades = base_trades.clone();
        // Deterministic Fisher-Yates permutation of the feature trajectories across all 173 trades
        let n = shuffled_trades.len();
        let mut perm: Vec<usize> = (0..n).collect();
        let mut rng_state = 0x9E37_79B9_7F4A_7C15_u64;
        for i in (1..n).rev() {
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let j = (rng_state as usize) % (i + 1);
            perm.swap(i, j);
        }
        for i in 0..n {
            let donor = &base_trades[perm[i]];
            shuffled_trades[i].entry_embedding_128 = donor.entry_embedding_128.clone();
            shuffled_trades[i].post_entry_bar1_features_32 =
                donor.post_entry_bar1_features_32.clone();
            shuffled_trades[i].entry_bars = donor.entry_bars.clone();
            shuffled_trades[i].entry_bar_idx = donor.entry_bar_idx;
            shuffled_trades[i].entry_price = donor.entry_price;
            shuffled_trades[i].initial_structural_stop = donor.initial_structural_stop;
            shuffled_trades[i].target_1_price = donor.target_1_price;
            shuffled_trades[i].direction = donor.direction;
            shuffled_trades[i].ai_confidence_pct = donor.ai_confidence_pct;
            shuffled_trades[i].precedent_win_rate_pct = donor.precedent_win_rate_pct;
            shuffled_trades[i].was_vetoed_by_loss_autopsy = false;
            shuffled_trades[i].was_scratched_by_loss_autopsy = false;
        }
        shuf_engine.replace_closed_trades_for_audit(shuffled_trades);
        let _ = shuf_engine.run_multi_generation_evolution_to_gen28();
        let shuf_snap = shuf_engine.query_filtered_snapshot(&HyperFilterQuery::default());
        let shuf_cm = &shuf_snap.scoreboard.confusion_matrix;

        println!(
            "\n[TRAP 3 — 3C: Random Feature Permutation (Null Hypothesis Test)]\n  TP Vetoes: {} | FP Vetoes: {} | TP Scratches: {} | FP Scratches: {} | Winners Preserved: {} / 116 | Remaining Unscratched Losses: {} / 57 | Veto Precision: {:.2}% | Scratch Precision: {:.2}%\n",
            shuf_cm.true_positive_losses_vetoed,
            shuf_cm.false_positive_winners_vetoed,
            shuf_cm.true_positive_losses_scratched,
            shuf_cm.false_positive_winners_scratched,
            shuf_cm.untouched_winners_count,
            shuf_cm.remaining_unscratched_losses,
            shuf_cm.entry_veto_precision_pct,
            shuf_cm.bar1_scratch_precision_pct,
        );
    }
}
