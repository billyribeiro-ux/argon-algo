//! `live-execution-server` — Sub-Millisecond Axum 0.8.9 Live Execution Server,
//! ONNX 5-Brain Inference Engine, `pgvector` k-NN Precedent Matcher,
//! `POSITIONS OPENED` Active Monitor, `closed_trade_ledger` Archiver, and
//! 1-Click `.CSV` / Multi-Sheet `.XLSX` OpenXML Workbook Exporter.

pub mod exporter;
pub mod ledger;
pub mod onnx_engine;
pub mod precedent_knn;
pub mod routes;

pub use exporter::{ExportError, ExportScope, export_to_csv_bytes, export_to_xlsx_bytes};
pub use ledger::{
    BrainBreakdownSummary, ClosedOutcomeBadge, ClosedTradeRow, FilteredLedgerSnapshot,
    HyperFilterQuery, MasterExecutiveScoreboard, OpenPositionRow, RealTimeLedgerEngine,
};
pub use onnx_engine::{
    InferenceError, LiveBrainInferenceOutput, LoadedOnnxBrainModel, MatchedOptionExecutionPlan,
    OnnxMultiBrainEngine,
};
pub use precedent_knn::{HistoricalTraceRecord, PrecedentKnnIndex, PrecedentMatchSummary};
pub use routes::{AppState, build_axum_router};

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
}
