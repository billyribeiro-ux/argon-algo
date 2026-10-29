//! Enterprise 1-Click `.CSV` & Multi-Sheet `.XLSX` OpenXML Workbook Exporter (`exporter.rs`).
//!
//! Supports all 3 export scopes:
//! - `ActiveFilteredSlice`
//! - `PositionsOpenedOnly`
//! - `FullMasterDatabase`
//!
//! Generates:
//! 1. RFC-4180 UTF-8 `.CSV` with 30 columns and exact 5-decimal `$0.00001` sub-penny precision.
//! 2. Formatted 4-Sheet `.XLSX` OpenXML Workbook (`Executive_Summary` with native Excel formulas,
//!    `Positions_Opened`, `Closed_Trade_Ledger` with frozen top header, auto-filters & conditional
//!    green/red P&L styling, and `By_Brain_Breakdowns`).

use std::io::{Cursor, Write};

use csv::WriterBuilder;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use zip::ZipWriter;
use zip::write::SimpleFileOptions;

use crate::ledger::{ClosedTradeRow, FilteredLedgerSnapshot, OpenPositionRow};

#[derive(Debug, Error)]
pub enum ExportError {
    #[error("CSV serialization error: {0}")]
    Csv(#[from] csv::Error),
    #[error("XLSX ZIP archive error: {0}")]
    Zip(#[from] zip::result::ZipError),
    #[error("I/O error during export: {0}")]
    Io(#[from] std::io::Error),
}

/// Export scope selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExportScope {
    #[serde(rename = "ACTIVE_FILTERED_SLICE")]
    ActiveFilteredSlice,
    #[serde(rename = "POSITIONS_OPENED_ONLY")]
    PositionsOpenedOnly,
    #[serde(rename = "FULL_MASTER_DATABASE")]
    FullMasterDatabase,
}

impl ExportScope {
    pub fn from_query_str(s: Option<&str>) -> Self {
        match s.unwrap_or("ACTIVE_FILTERED_SLICE").to_uppercase().as_str() {
            "POSITIONS_OPENED_ONLY" | "POSITIONS_OPENED" => Self::PositionsOpenedOnly,
            "FULL_MASTER_DATABASE" | "FULL" => Self::FullMasterDatabase,
            _ => Self::ActiveFilteredSlice,
        }
    }
}

/// Formats a millisecond duration into human-readable institutional hold time (`01h 14m 08s` or `4d 06h`).
pub fn format_hold_duration(duration_ms: i64) -> String {
    let total_secs = (duration_ms.max(0)) / 1_000;
    let days = total_secs / 86_400;
    let hours = (total_secs % 86_400) / 3_600;
    let mins = (total_secs % 3_600) / 60;
    let secs = total_secs % 60;

    if days > 0 {
        format!("{days}d {hours:02}h {mins:02}m")
    } else if hours > 0 {
        format!("{hours:02}h {mins:02}m {secs:02}s")
    } else {
        format!("{mins:02}m {secs:02}s")
    }
}

/// Exports the requested snapshot scope to an RFC-4180 compliant 30-column UTF-8 `.CSV` buffer.
pub fn export_to_csv_bytes(
    snapshot: &FilteredLedgerSnapshot,
    scope: ExportScope,
) -> Result<Vec<u8>, ExportError> {
    let mut wtr = WriterBuilder::new()
        .has_headers(true)
        .from_writer(Vec::new());

    wtr.write_record([
        "record_type",
        "trade_or_position_id",
        "status_or_outcome_badge",
        "brain_id",
        "brain_name",
        "symbol",
        "exchange_index",
        "direction",
        "opened_at_iso",
        "closed_at_iso",
        "hold_duration_formatted",
        "hold_duration_ms",
        "entry_price_usd",
        "live_or_exit_price_usd",
        "structural_stop_price_usd",
        "target_1_price_usd",
        "target_2_price_usd",
        "quantity_shares",
        "option_structure",
        "matched_option_contract",
        "option_entry_premium_usd",
        "option_live_or_exit_premium_usd",
        "option_stop_premium_usd",
        "gross_pnl_usd",
        "net_pnl_usd",
        "option_pnl_usd",
        "stock_return_pct",
        "r_multiple",
        "mae_pct",
        "mfe_pct",
        "cumulative_net_profit_usd",
        "ai_confidence_pct",
        "primary_fingerprint_driver",
    ])?;

    for open in &snapshot.positions_opened {
        wtr.write_record([
            "POSITION_OPENED",
            &open.position_id.to_string(),
            open.trace_health_badge.as_str(),
            open.brain_id.code(),
            &open.brain_name,
            &open.symbol,
            &open.exchange_index,
            open.direction.as_str(),
            &open.opened_at_iso,
            "OPEN_IN_TRADE",
            &format_hold_duration(open.live_hold_duration_ms),
            &open.live_hold_duration_ms.to_string(),
            &open.entry_price.to_string(),
            &open.live_price.to_string(),
            &open.structural_stop_price.to_string(),
            &open.target_1_price.to_string(),
            &open.target_2_price.to_string(),
            &format!("{:.4}", open.quantity_shares),
            &open.option_structure,
            &open.matched_option_contract,
            &open.option_entry_premium.to_string(),
            &open.option_live_premium.to_string(),
            &open.option_stop_premium.to_string(),
            &format!("{:.5}", open.unrealized_stock_pnl_usd),
            &format!("{:.5}", open.unrealized_stock_pnl_usd),
            &format!("{:.5}", open.unrealized_option_pnl_usd),
            &format!("{:.4}", open.unrealized_stock_pnl_pct),
            &format!("{:.2}", open.current_r_multiple),
            &format!("{:.4}", open.mae_pct),
            &format!("{:.4}", open.mfe_pct),
            &format!("{:.5}", snapshot.scoreboard.combined_net_profit_usd),
            &format!("{:.2}", open.ai_confidence_pct),
            open.primary_driver.as_str(),
        ])?;
    }

    if scope != ExportScope::PositionsOpenedOnly {
        for closed in &snapshot.closed_trades {
            wtr.write_record([
                "CLOSED_TRADE",
                &closed.trade_id.to_string(),
                closed.outcome_badge.as_str(),
                closed.brain_id.code(),
                &closed.brain_name,
                &closed.symbol,
                &closed.exchange_index,
                closed.direction.as_str(),
                &closed.opened_at_iso,
                &closed.closed_at_iso,
                &format_hold_duration(closed.hold_duration_ms),
                &closed.hold_duration_ms.to_string(),
                &closed.entry_price.to_string(),
                &closed.actual_exit_price.to_string(),
                &closed.initial_structural_stop.to_string(),
                &closed.target_1_price.to_string(),
                &closed.target_2_price.to_string(),
                &format!("{:.4}", closed.quantity_shares),
                &closed.option_structure,
                &closed.matched_option_contract,
                &closed.option_entry_premium.to_string(),
                &closed.option_exit_premium.to_string(),
                &closed.option_stop_premium.to_string(),
                &format!("{:.5}", closed.gross_pnl_usd),
                &format!("{:.5}", closed.net_pnl_usd),
                &format!("{:.5}", closed.option_net_pnl_usd),
                &format!("{:.4}", closed.stock_return_pct),
                &format!("{:.2}", closed.realized_r_multiple),
                &format!("{:.4}", closed.mae_pct),
                &format!("{:.4}", closed.mfe_pct),
                &format!("{:.5}", closed.cumulative_net_profit_usd),
                &format!("{:.2}", closed.ai_confidence_pct),
                closed.primary_driver.as_str(),
            ])?;
        }
    }

    let bytes = wtr
        .into_inner()
        .map_err(|e| ExportError::Io(e.into_error()))?;
    Ok(bytes)
}

/// Generates a complete 4-sheet OpenXML `.XLSX` workbook with native Excel formulas,
/// frozen headers, auto-filters, currency/percentage cell styles, and green/red P&L styling.
pub fn export_to_xlsx_bytes(snapshot: &FilteredLedgerSnapshot) -> Result<Vec<u8>, ExportError> {
    let buf = Cursor::new(Vec::new());
    let mut zip = ZipWriter::new(buf);
    let opts = SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated);

    // 1. [Content_Types].xml
    zip.start_file("[Content_Types].xml", opts)?;
    zip.write_all(CONTENT_TYPES_XML.as_bytes())?;

    // 2. _rels/.rels
    zip.start_file("_rels/.rels", opts)?;
    zip.write_all(ROOT_RELS_XML.as_bytes())?;

    // 3. xl/workbook.xml (4 tabs: Executive_Summary, Positions_Opened, Closed_Trade_Ledger, By_Brain_Breakdowns)
    zip.start_file("xl/workbook.xml", opts)?;
    zip.write_all(WORKBOOK_XML.as_bytes())?;

    // 4. xl/_rels/workbook.xml.rels
    zip.start_file("xl/_rels/workbook.xml.rels", opts)?;
    zip.write_all(WORKBOOK_RELS_XML.as_bytes())?;

    // 5. xl/styles.xml (Header Obsidian Fill, Currency/5-Decimal Formats, Emerald Winner & Crimson Loser Styles)
    zip.start_file("xl/styles.xml", opts)?;
    zip.write_all(STYLES_XML.as_bytes())?;

    // 6. Sheet 1: Executive_Summary (with native Excel formulas SUM, AVERAGE, COUNTIF)
    let sheet1 = build_sheet1_executive_summary_xml(snapshot);
    zip.start_file("xl/worksheets/sheet1.xml", opts)?;
    zip.write_all(sheet1.as_bytes())?;

    // 7. Sheet 2: Positions_Opened
    let sheet2 = build_sheet2_positions_opened_xml(&snapshot.positions_opened);
    zip.start_file("xl/worksheets/sheet2.xml", opts)?;
    zip.write_all(sheet2.as_bytes())?;

    // 8. Sheet 3: Closed_Trade_Ledger (with frozen top header, autoFilter, conditional P&L styling)
    let sheet3 = build_sheet3_closed_ledger_xml(&snapshot.closed_trades);
    zip.start_file("xl/worksheets/sheet3.xml", opts)?;
    zip.write_all(sheet3.as_bytes())?;

    // 9. Sheet 4: By_Brain_Breakdowns
    let sheet4 = build_sheet4_brain_breakdowns_xml(snapshot);
    zip.start_file("xl/worksheets/sheet4.xml", opts)?;
    zip.write_all(sheet4.as_bytes())?;

    let cursor = zip.finish()?;
    Ok(cursor.into_inner())
}

fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

fn col_letter(col_idx_1based: usize) -> String {
    let mut n = col_idx_1based;
    let mut out = String::new();
    while n > 0 {
        let rem = (n - 1) % 26;
        out.insert(0, (b'A' + rem as u8) as char);
        n = (n - 1) / 26;
    }
    out
}

fn cell_str(row: usize, col: usize, style: u32, text: &str) -> String {
    let c = col_letter(col);
    let escaped = xml_escape(text);
    format!(r#"<c r="{c}{row}" s="{style}" t="inlineStr"><is><t>{escaped}</t></is></c>"#)
}

fn cell_num(row: usize, col: usize, style: u32, val: f64) -> String {
    let c = col_letter(col);
    let clean = if val.is_finite() { val } else { 0.0 };
    format!(r#"<c r="{c}{row}" s="{style}"><v>{clean:.5}</v></c>"#)
}

fn cell_formula(row: usize, col: usize, style: u32, formula: &str, cached_val: f64) -> String {
    let c = col_letter(col);
    let clean = if cached_val.is_finite() {
        cached_val
    } else {
        0.0
    };
    format!(r#"<c r="{c}{row}" s="{style}"><f>{formula}</f><v>{clean:.5}</v></c>"#)
}

fn build_sheet1_executive_summary_xml(snapshot: &FilteredLedgerSnapshot) -> String {
    let sb = &snapshot.scoreboard;
    let closed_end_row = (snapshot.closed_trades.len() + 1).max(2);
    let open_end_row = (snapshot.positions_opened.len() + 1).max(2);

    let mut xml = String::with_capacity(8192);
    xml.push_str(r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>"#);
    xml.push_str(
        r#"<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">"#,
    );
    xml.push_str(r#"<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>"#);
    xml.push_str("<sheetData>");

    // Row 1: Title Header
    xml.push_str(r#"<row r="1">"#);
    xml.push_str(&cell_str(
        1,
        1,
        1,
        "QUANT INTELLIGENCE PLATFORM — EXECUTIVE SCOREBOARD",
    ));
    xml.push_str(&cell_str(1, 2, 1, "LIVE & HISTORICAL VALUE"));
    xml.push_str(&cell_str(1, 3, 1, "EXCEL AUDIT FORMULA / TELEMETRY SOURCE"));
    xml.push_str("</row>");

    // Row 2: Realized Net Profit ($) with native Excel SUM formula referencing Closed_Trade_Ledger!N
    xml.push_str(r#"<row r="2">"#);
    xml.push_str(&cell_str(2, 1, 0, "Realized Net Profit ($)"));
    xml.push_str(&cell_formula(
        2,
        2,
        2,
        &format!("SUM(Closed_Trade_Ledger!N2:N{closed_end_row})"),
        sb.realized_net_profit_usd,
    ));
    xml.push_str(&cell_str(2, 3, 0, "SUM of Closed_Trade_Ledger Net P&L ($)"));
    xml.push_str("</row>");

    // Row 3: Live Open Unrealized P&L ($) with native Excel SUM formula referencing Positions_Opened!M
    xml.push_str(r#"<row r="3">"#);
    xml.push_str(&cell_str(3, 1, 0, "Live Open Unrealized P&L ($)"));
    xml.push_str(&cell_formula(
        3,
        2,
        2,
        &format!("SUM(Positions_Opened!M2:M{open_end_row})"),
        sb.unrealized_open_pnl_usd,
    ));
    xml.push_str(&cell_str(
        3,
        3,
        0,
        "SUM of Positions_Opened Live Stock P&L ($)",
    ));
    xml.push_str("</row>");

    // Row 4: Combined Total Portfolio Net Profit ($)
    xml.push_str(r#"<row r="4">"#);
    xml.push_str(&cell_str(
        4,
        1,
        0,
        "Combined Total Portfolio Net Profit ($)",
    ));
    xml.push_str(&cell_formula(4, 2, 2, "B2+B3", sb.combined_net_profit_usd));
    xml.push_str(&cell_str(
        4,
        3,
        0,
        "Realized + Live Open Unrealized Net Profit",
    ));
    xml.push_str("</row>");

    // Row 5: Total Closed Trades Count
    xml.push_str(r#"<row r="5">"#);
    xml.push_str(&cell_str(5, 1, 0, "Total Closed Trades (#)"));
    xml.push_str(&cell_formula(
        5,
        2,
        0,
        &format!("COUNTA(Closed_Trade_Ledger!A2:A{closed_end_row})"),
        sb.total_closed_trades as f64,
    ));
    xml.push_str(&cell_str(
        5,
        3,
        0,
        "Total Immutable Rows in Closed_Trade_Ledger",
    ));
    xml.push_str("</row>");

    // Row 6: Winning Trades Count via COUNTIF
    xml.push_str(r#"<row r="6">"#);
    xml.push_str(&cell_str(6, 1, 0, "Total Winning Trades (#)"));
    xml.push_str(&cell_formula(
        6,
        2,
        3,
        &format!(r#"COUNTIF(Closed_Trade_Ledger!N2:N{closed_end_row},">0")"#),
        sb.total_wins as f64,
    ));
    xml.push_str(&cell_str(6, 3, 0, "COUNTIF Net P&L > $0.00"));
    xml.push_str("</row>");

    // Row 7: Win Rate (%)
    xml.push_str(r#"<row r="7">"#);
    xml.push_str(&cell_str(7, 1, 0, "Win Rate (%)"));
    xml.push_str(&cell_formula(
        7,
        2,
        0,
        "IF(B5>0,(B6/B5)*100,0)",
        sb.win_rate_pct,
    ));
    xml.push_str(&cell_str(7, 3, 0, "Winning Trades / Total Closed Trades"));
    xml.push_str("</row>");

    // Row 8: Average Realized R-Multiple via AVERAGE
    let avg_r = if !snapshot.closed_trades.is_empty() {
        snapshot
            .closed_trades
            .iter()
            .map(|r| r.realized_r_multiple)
            .sum::<f64>()
            / (snapshot.closed_trades.len() as f64)
    } else {
        0.0
    };
    xml.push_str(r#"<row r="8">"#);
    xml.push_str(&cell_str(8, 1, 0, "Average Realized R-Multiple (R)"));
    xml.push_str(&cell_formula(
        8,
        2,
        0,
        &format!("AVERAGE(Closed_Trade_Ledger!P2:P{closed_end_row})"),
        avg_r,
    ));
    xml.push_str(&cell_str(
        8,
        3,
        0,
        "AVERAGE of Closed_Trade_Ledger R-Multiple",
    ));
    xml.push_str("</row>");

    // Row 9: Profit Factor
    xml.push_str(r#"<row r="9">"#);
    xml.push_str(&cell_str(
        9,
        1,
        0,
        "Profit Factor (Gross Profit / Gross Loss)",
    ));
    xml.push_str(&cell_num(9, 2, 0, sb.profit_factor));
    xml.push_str(&cell_str(
        9,
        3,
        0,
        "Institutional Expectancy Efficiency Ratio",
    ));
    xml.push_str("</row>");

    // Row 10: Active Positions Opened Count
    xml.push_str(r#"<row r="10">"#);
    xml.push_str(&cell_str(10, 1, 0, "Active Positions Opened (#)"));
    xml.push_str(&cell_num(10, 2, 0, sb.positions_opened_count as f64));
    xml.push_str(&cell_str(
        10,
        3,
        0,
        "Currently Active Signals in Positions_Opened",
    ));
    xml.push_str("</row>");

    xml.push_str("</sheetData></worksheet>");
    xml
}

fn build_sheet2_positions_opened_xml(open_rows: &[OpenPositionRow]) -> String {
    let mut xml = String::with_capacity(16384);
    xml.push_str(r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>"#);
    xml.push_str(
        r#"<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">"#,
    );
    xml.push_str(r#"<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>"#);
    xml.push_str("<sheetData>");

    let headers = [
        "Position_ID",
        "Opened_At_ISO",
        "Hold_Time",
        "Brain_ID",
        "Symbol",
        "Direction",
        "Entry_Price_USD",
        "Live_Price_USD",
        "Structural_Stop_USD",
        "Target_1_USD",
        "Target_2_USD",
        "Matched_Option_Contract",
        "Unrealized_Stock_PnL_USD",
        "Unrealized_Option_PnL_USD",
        "Current_R_Multiple",
        "Trace_Health_Badge",
        "Primary_Driver",
        "AI_Confidence_Pct",
    ];

    xml.push_str(r#"<row r="1">"#);
    for (idx, h) in headers.iter().enumerate() {
        xml.push_str(&cell_str(1, idx + 1, 1, h));
    }
    xml.push_str("</row>");

    for (r_idx, row) in open_rows.iter().enumerate() {
        let excel_row = r_idx + 2;
        let pnl_style = if row.unrealized_stock_pnl_usd >= 0.0 {
            3
        } else {
            4
        };
        xml.push_str(&format!(r#"<row r="{excel_row}">"#));
        xml.push_str(&cell_str(excel_row, 1, 0, &row.position_id.to_string()));
        xml.push_str(&cell_str(excel_row, 2, 0, &row.opened_at_iso));
        xml.push_str(&cell_str(
            excel_row,
            3,
            0,
            &format_hold_duration(row.live_hold_duration_ms),
        ));
        xml.push_str(&cell_str(excel_row, 4, 0, row.brain_id.code()));
        xml.push_str(&cell_str(excel_row, 5, 0, &row.symbol));
        xml.push_str(&cell_str(excel_row, 6, 0, row.direction.as_str()));
        xml.push_str(&cell_num(excel_row, 7, 2, row.entry_price.to_f64()));
        xml.push_str(&cell_num(excel_row, 8, 2, row.live_price.to_f64()));
        xml.push_str(&cell_num(
            excel_row,
            9,
            2,
            row.structural_stop_price.to_f64(),
        ));
        xml.push_str(&cell_num(excel_row, 10, 2, row.target_1_price.to_f64()));
        xml.push_str(&cell_num(excel_row, 11, 2, row.target_2_price.to_f64()));
        xml.push_str(&cell_str(excel_row, 12, 0, &row.matched_option_contract));
        xml.push_str(&cell_num(
            excel_row,
            13,
            pnl_style,
            row.unrealized_stock_pnl_usd,
        ));
        xml.push_str(&cell_num(
            excel_row,
            14,
            pnl_style,
            row.unrealized_option_pnl_usd,
        ));
        xml.push_str(&cell_num(excel_row, 15, 0, row.current_r_multiple));
        xml.push_str(&cell_str(excel_row, 16, 0, row.trace_health_badge.as_str()));
        xml.push_str(&cell_str(excel_row, 17, 0, row.primary_driver.as_str()));
        xml.push_str(&cell_num(excel_row, 18, 0, row.ai_confidence_pct));
        xml.push_str("</row>");
    }

    let last_row = (open_rows.len() + 1).max(1);
    xml.push_str("</sheetData>");
    xml.push_str(&format!(r#"<autoFilter ref="A1:R{last_row}"/>"#));
    xml.push_str("</worksheet>");
    xml
}

fn build_sheet3_closed_ledger_xml(closed_rows: &[ClosedTradeRow]) -> String {
    let mut xml = String::with_capacity(32768);
    xml.push_str(r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>"#);
    xml.push_str(
        r#"<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">"#,
    );
    xml.push_str(r#"<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>"#);
    xml.push_str("<sheetData>");

    let headers = [
        "Trade_ID",
        "Outcome_Badge",
        "Brain_ID",
        "Symbol",
        "Exchange_Index",
        "Direction",
        "Opened_At_ISO",
        "Closed_At_ISO",
        "Hold_Duration",
        "Entry_Price_USD",
        "Structural_Stop_USD",
        "Actual_Exit_Price_USD",
        "Gross_PnL_USD",
        "Net_PnL_USD",
        "Stock_Return_Pct",
        "Realized_R_Multiple",
        "MAE_Pct",
        "MFE_Pct",
        "Cumulative_Net_Profit_USD",
        "Matched_Option_Contract",
        "Option_Net_PnL_USD",
        "Primary_Driver",
        "AI_Confidence_Pct",
    ];

    xml.push_str(r#"<row r="1">"#);
    for (idx, h) in headers.iter().enumerate() {
        xml.push_str(&cell_str(1, idx + 1, 1, h));
    }
    xml.push_str("</row>");

    for (r_idx, row) in closed_rows.iter().enumerate() {
        let excel_row = r_idx + 2;
        // Style 3 = Electric Emerald Winner, Style 4 = Laser Crimson Loser
        let pnl_style = if row.net_pnl_usd >= 0.0 { 3 } else { 4 };

        xml.push_str(&format!(r#"<row r="{excel_row}">"#));
        xml.push_str(&cell_str(excel_row, 1, 0, &row.trade_id.to_string()));
        xml.push_str(&cell_str(
            excel_row,
            2,
            pnl_style,
            row.outcome_badge.as_str(),
        ));
        xml.push_str(&cell_str(excel_row, 3, 0, row.brain_id.code()));
        xml.push_str(&cell_str(excel_row, 4, 0, &row.symbol));
        xml.push_str(&cell_str(excel_row, 5, 0, &row.exchange_index));
        xml.push_str(&cell_str(excel_row, 6, 0, row.direction.as_str()));
        xml.push_str(&cell_str(excel_row, 7, 0, &row.opened_at_iso));
        xml.push_str(&cell_str(excel_row, 8, 0, &row.closed_at_iso));
        xml.push_str(&cell_str(
            excel_row,
            9,
            0,
            &format_hold_duration(row.hold_duration_ms),
        ));
        xml.push_str(&cell_num(excel_row, 10, 2, row.entry_price.to_f64()));
        xml.push_str(&cell_num(
            excel_row,
            11,
            2,
            row.initial_structural_stop.to_f64(),
        ));
        xml.push_str(&cell_num(excel_row, 12, 2, row.actual_exit_price.to_f64()));
        xml.push_str(&cell_num(excel_row, 13, pnl_style, row.gross_pnl_usd));
        xml.push_str(&cell_num(excel_row, 14, pnl_style, row.net_pnl_usd));
        xml.push_str(&cell_num(excel_row, 15, pnl_style, row.stock_return_pct));
        xml.push_str(&cell_num(excel_row, 16, pnl_style, row.realized_r_multiple));
        xml.push_str(&cell_num(excel_row, 17, 0, row.mae_pct));
        xml.push_str(&cell_num(excel_row, 18, 0, row.mfe_pct));
        xml.push_str(&cell_num(excel_row, 19, 2, row.cumulative_net_profit_usd));
        xml.push_str(&cell_str(excel_row, 20, 0, &row.matched_option_contract));
        xml.push_str(&cell_num(excel_row, 21, pnl_style, row.option_net_pnl_usd));
        xml.push_str(&cell_str(excel_row, 22, 0, row.primary_driver.as_str()));
        xml.push_str(&cell_num(excel_row, 23, 0, row.ai_confidence_pct));
        xml.push_str("</row>");
    }

    let last_row = (closed_rows.len() + 1).max(1);
    xml.push_str("</sheetData>");
    xml.push_str(&format!(r#"<autoFilter ref="A1:W{last_row}"/>"#));
    xml.push_str("</worksheet>");
    xml
}

fn build_sheet4_brain_breakdowns_xml(snapshot: &FilteredLedgerSnapshot) -> String {
    let mut xml = String::with_capacity(8192);
    xml.push_str(r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>"#);
    xml.push_str(
        r#"<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">"#,
    );
    xml.push_str(r#"<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>"#);
    xml.push_str("<sheetData>");

    let headers = [
        "Brain_ID",
        "Brain_Name",
        "Positions_Opened_Count",
        "Closed_Trades_Count",
        "Wins_Count",
        "Losses_Count",
        "Win_Rate_Pct",
        "Realized_Net_Profit_USD",
        "Unrealized_Open_PnL_USD",
        "Profit_Factor",
        "Avg_R_Multiple",
        "Avg_Hold_Duration",
    ];

    xml.push_str(r#"<row r="1">"#);
    for (idx, h) in headers.iter().enumerate() {
        xml.push_str(&cell_str(1, idx + 1, 1, h));
    }
    xml.push_str("</row>");

    for (r_idx, b) in snapshot.scoreboard.by_brain_breakdowns.iter().enumerate() {
        let excel_row = r_idx + 2;
        let pnl_style = if b.realized_net_profit_usd >= 0.0 {
            3
        } else {
            4
        };
        xml.push_str(&format!(r#"<row r="{excel_row}">"#));
        xml.push_str(&cell_str(excel_row, 1, 0, b.brain_id.code()));
        xml.push_str(&cell_str(excel_row, 2, 0, &b.brain_name));
        xml.push_str(&cell_num(excel_row, 3, 0, b.open_positions_count as f64));
        xml.push_str(&cell_num(excel_row, 4, 0, b.closed_trades_count as f64));
        xml.push_str(&cell_num(excel_row, 5, 3, b.wins_count as f64));
        xml.push_str(&cell_num(excel_row, 6, 4, b.losses_count as f64));
        xml.push_str(&cell_num(excel_row, 7, 0, b.win_rate_pct));
        xml.push_str(&cell_num(
            excel_row,
            8,
            pnl_style,
            b.realized_net_profit_usd,
        ));
        xml.push_str(&cell_num(excel_row, 9, 2, b.unrealized_open_pnl_usd));
        xml.push_str(&cell_num(excel_row, 10, 0, b.profit_factor));
        xml.push_str(&cell_num(excel_row, 11, 0, b.avg_r_multiple));
        xml.push_str(&cell_str(
            excel_row,
            12,
            0,
            &format_hold_duration(b.avg_hold_duration_ms),
        ));
        xml.push_str("</row>");
    }

    xml.push_str("</sheetData>");
    xml.push_str(r#"<autoFilter ref="A1:L6"/>"#);
    xml.push_str("</worksheet>");
    xml
}

const CONTENT_TYPES_XML: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/worksheets/sheet2.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/worksheets/sheet3.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/worksheets/sheet4.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
</Types>"#;

const ROOT_RELS_XML: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>"#;

const WORKBOOK_XML: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>
    <sheet name="Executive_Summary" sheetId="1" r:id="rId1"/>
    <sheet name="Positions_Opened" sheetId="2" r:id="rId2"/>
    <sheet name="Closed_Trade_Ledger" sheetId="3" r:id="rId3"/>
    <sheet name="By_Brain_Breakdowns" sheetId="4" r:id="rId4"/>
  </sheets>
</workbook>"#;

const WORKBOOK_RELS_XML: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet2.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet3.xml"/>
  <Relationship Id="rId4" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet4.xml"/>
  <Relationship Id="rId5" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>"#;

const STYLES_XML: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <numFmts count="1">
    <numFmt numFmtId="164" formatCode="$#,##0.00000;($#,##0.00000)"/>
  </numFmts>
  <fonts count="4">
    <font><sz val="10"/><name val="JetBrains Mono"/></font>
    <font><b/><sz val="10"/><color rgb="FFFFFFFF"/><name val="Inter"/></font>
    <font><b/><sz val="10"/><color rgb="FF00E676"/><name val="JetBrains Mono"/></font>
    <font><b/><sz val="10"/><color rgb="FFFF5252"/><name val="JetBrains Mono"/></font>
  </fonts>
  <fills count="5">
    <fill><patternFill patternType="none"/></fill>
    <fill><patternFill patternType="gray125"/></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FF0F1419"/></patternFill></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FF0D2B1D"/></patternFill></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FF2D1215"/></patternFill></fill>
  </fills>
  <borders count="1">
    <border><left/><right/><top/><bottom/><diagonal/></border>
  </borders>
  <cellStyleXfs count="1">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0"/>
  </cellStyleXfs>
  <cellXfs count="5">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>
    <xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyFont="1" applyFill="1"/>
    <xf numFmtId="164" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/>
    <xf numFmtId="164" fontId="2" fillId="3" borderId="0" xfId="0" applyNumberFormat="1" applyFont="1" applyFill="1"/>
    <xf numFmtId="164" fontId="3" fillId="4" borderId="0" xfId="0" applyNumberFormat="1" applyFont="1" applyFill="1"/>
  </cellXfs>
</styleSheet>"#;
