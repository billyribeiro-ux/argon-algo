//! `macro_rates.rs` — 36-Year (1990–2026) Volatility Surface & Macro Interest Rates Pipeline.
//!
//! Handles:
//! - CBOE Official CDN CSVs: VIX, VIX9D, VIX3M, VVIX, SKEW.
//!   Clean handling of pre-June-2004 CBOE rows where OPEN=HIGH=LOW=CLOSE.
//! - FRED Official CSVs: DFF, DGS2, DGS10, T10Y2Y, BAMLH0A0HYM2 (forward-filling '.' holidays).
//! - Point-in-Time Causal Query Engine: guarantees zero leakage of day D's EOD values into 09:35 ET intraday bars.
//! - Compilation of binary shard `data/shards/context/macro_vix_daily.bin`.

use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use thiserror::Error;
use tracing::{debug, info};

#[derive(Debug, Error)]
pub enum MacroRatesError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("CSV parse error: {0}")]
    Csv(String),
}

/// A single day's causal macro snapshot (72 bytes, 8-byte aligned, zero padding).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable, Serialize, Deserialize)]
pub struct MacroDailyRecord {
    pub eod_ts_sec: i64,
    pub date_yyyymmdd: u32,
    pub vix_close: f32,
    pub vix_open: f32,
    pub vix_high: f32,
    pub vix_low: f32,
    pub vix9d_close: f32,
    pub vix3m_close: f32,
    pub vvix_close: f32,
    pub skew_close: f32,
    pub dff_rate: f32,
    pub dgs2_rate: f32,
    pub dgs10_rate: f32,
    pub t10y2y_spread: f32,
    pub hy_oas_spread: f32,
    pub flags: u32, // bitfield: e.g. 1 = interpolated, 2 = holiday forward-fill
    pub _pad0: u32,
}

pub const CBOE_VIX_URL: &str = "https://cdn-api.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv";
pub const CBOE_VIX9D_URL: &str = "https://cdn-api.cboe.com/api/global/us_indices/daily_prices/VIX9D_History.csv";
pub const CBOE_VIX3M_URL: &str = "https://cdn-api.cboe.com/api/global/us_indices/daily_prices/VIX3M_History.csv";
pub const CBOE_VVIX_URL: &str = "https://cdn-api.cboe.com/api/global/us_indices/daily_prices/VVIX_History.csv";
pub const CBOE_SKEW_URL: &str = "https://cdn-api.cboe.com/api/global/us_indices/daily_prices/SKEW_History.csv";

pub const FRED_DFF_URL: &str = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFF";
pub const FRED_DGS2_URL: &str = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS2";
pub const FRED_DGS10_URL: &str = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10";
pub const FRED_T10Y2Y_URL: &str = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=T10Y2Y";
pub const FRED_HY_OAS_URL: &str = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=BAMLH0A0HYM2";

#[derive(Debug, Default)]
pub struct MacroRatesSurface {
    records: BTreeMap<u32, MacroDailyRecord>,
}

impl MacroRatesSurface {
    pub fn new() -> Self {
        Self::default()
    }

    /// Ingests CBOE VIX CSV (`DATE,OPEN,HIGH,LOW,CLOSE`).
    pub fn parse_cboe_vix_csv(&mut self, csv_content: &str) -> Result<usize, MacroRatesError> {
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(true)
            .flexible(true)
            .from_reader(csv_content.as_bytes());

        let mut count = 0;
        for record in reader.records() {
            let r = record.map_err(|e| MacroRatesError::Csv(e.to_string()))?;
            if r.len() < 5 {
                continue;
            }

            let date_str = &r[0];
            let yyyymmdd = parse_cboe_date(date_str);
            if yyyymmdd == 0 {
                continue;
            }

            let open: f32 = r[1].trim().parse().unwrap_or(0.0);
            let high: f32 = r[2].trim().parse().unwrap_or(open);
            let low: f32 = r[3].trim().parse().unwrap_or(open);
            let close: f32 = r[4].trim().parse().unwrap_or(open);

            if close <= 0.0 || !close.is_finite() {
                continue;
            }

            // Pre-June 2004 CBOE rows: OPEN=HIGH=LOW=CLOSE. Enforce geometry safely:
            let safe_low = low.min(open).min(close);
            let safe_high = high.max(open).max(close);

            let eod_ts = crate::corporate_actions::yyyymmdd_to_unix_day_range(yyyymmdd).0 + 57600; // 16:00 ET approx

            let entry = self.records.entry(yyyymmdd).or_insert_with(|| MacroDailyRecord {
                eod_ts_sec: eod_ts,
                date_yyyymmdd: yyyymmdd,
                vix_close: close,
                vix_open: open,
                vix_high: safe_high,
                vix_low: safe_low,
                vix9d_close: close,
                vix3m_close: close,
                vvix_close: 85.0,
                skew_close: 120.0,
                dff_rate: 3.5,
                dgs2_rate: 3.8,
                dgs10_rate: 4.2,
                t10y2y_spread: 0.4,
                hy_oas_spread: 350.0,
                flags: 0,
                _pad0: 0,
            });

            entry.vix_open = open;
            entry.vix_high = safe_high;
            entry.vix_low = safe_low;
            entry.vix_close = close;
            count += 1;
        }

        info!("Parsed {count} CBOE VIX daily records (span {} to {})", 
            self.records.keys().next().copied().unwrap_or(0),
            self.records.keys().last().copied().unwrap_or(0)
        );
        Ok(count)
    }

    /// Ingests auxiliary CBOE volatility index CSV (e.g. VIX9D, VIX3M, VVIX, SKEW).
    pub fn parse_cboe_aux_csv(&mut self, csv_content: &str, field_tag: &str) -> Result<usize, MacroRatesError> {
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(true)
            .flexible(true)
            .from_reader(csv_content.as_bytes());

        let mut count = 0;
        for record in reader.records() {
            let r = record.map_err(|e| MacroRatesError::Csv(e.to_string()))?;
            if r.len() < 2 {
                continue;
            }

            let date_str = &r[0];
            let yyyymmdd = parse_cboe_date(date_str);
            if yyyymmdd == 0 {
                continue;
            }

            // Close is either in column 4 (if OHLC) or column 1 (if DATE,CLOSE)
            let close_col = if r.len() >= 5 { 4 } else { 1 };
            if let Ok(val) = r[close_col].trim().parse::<f32>()
                && val > 0.0 && val.is_finite()
                    && let Some(entry) = self.records.get_mut(&yyyymmdd) {
                        match field_tag {
                            "VIX9D" => entry.vix9d_close = val,
                            "VIX3M" => entry.vix3m_close = val,
                            "VVIX" => entry.vvix_close = val,
                            "SKEW" => entry.skew_close = val,
                            _ => {}
                        }
                        count += 1;
                    }
        }

        debug!("Updated {count} records with CBOE field {field_tag}");
        Ok(count)
    }

    /// Ingests FRED daily interest rate or spread CSV (`DATE,VALUE`).
    pub fn parse_fred_csv(&mut self, csv_content: &str, field_tag: &str) -> Result<usize, MacroRatesError> {
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(true)
            .flexible(true)
            .from_reader(csv_content.as_bytes());

        let mut count = 0;
        let mut last_valid_val = 0.0_f32;

        for record in reader.records() {
            let r = record.map_err(|e| MacroRatesError::Csv(e.to_string()))?;
            if r.len() < 2 {
                continue;
            }

            let yyyymmdd = crate::corporate_actions::parse_date_to_yyyymmdd(&r[0]);
            if yyyymmdd == 0 {
                continue;
            }

            let val_str = r[1].trim();
            let val = if val_str == "." || val_str.is_empty() {
                // Forward fill missing holiday value
                last_valid_val
            } else if let Ok(parsed) = val_str.parse::<f32>() {
                last_valid_val = parsed;
                parsed
            } else {
                last_valid_val
            };

            if let Some(entry) = self.records.get_mut(&yyyymmdd) {
                match field_tag {
                    "DFF" => entry.dff_rate = val,
                    "DGS2" => entry.dgs2_rate = val,
                    "DGS10" => entry.dgs10_rate = val,
                    "T10Y2Y" => entry.t10y2y_spread = val,
                    "HY_OAS" => entry.hy_oas_spread = val,
                    _ => {}
                }
                count += 1;
            }
        }

        debug!("Updated {count} records with FRED field {field_tag}");
        Ok(count)
    }

    /// Forward-fills any missing companion fields across the timeline.
    pub fn forward_fill_gaps(&mut self) {
        let mut last_vix = 18.0_f32;
        let mut last_vix9d = 18.0_f32;
        let mut last_vix3m = 19.0_f32;
        let mut last_vvix = 88.0_f32;
        let mut last_skew = 120.0_f32;
        let mut last_dff = 3.5_f32;
        let mut last_dgs2 = 3.8_f32;
        let mut last_dgs10 = 4.2_f32;
        let mut last_hy_oas = 350.0_f32;

        for entry in self.records.values_mut() {
            if entry.vix_close > 0.0 { last_vix = entry.vix_close; } else { entry.vix_close = last_vix; }
            if entry.vix9d_close > 0.0 { last_vix9d = entry.vix9d_close; } else { entry.vix9d_close = last_vix9d.max(last_vix * 0.98); }
            if entry.vix3m_close > 0.0 { last_vix3m = entry.vix3m_close; } else { entry.vix3m_close = last_vix3m.max(last_vix * 1.05); }
            if entry.vvix_close > 0.0 { last_vvix = entry.vvix_close; } else { entry.vvix_close = last_vvix; }
            if entry.skew_close > 0.0 { last_skew = entry.skew_close; } else { entry.skew_close = last_skew; }
            if entry.dff_rate > 0.0 { last_dff = entry.dff_rate; } else { entry.dff_rate = last_dff; }
            if entry.dgs2_rate > 0.0 { last_dgs2 = entry.dgs2_rate; } else { entry.dgs2_rate = last_dgs2; }
            if entry.dgs10_rate > 0.0 { last_dgs10 = entry.dgs10_rate; } else { entry.dgs10_rate = last_dgs10; }
            entry.t10y2y_spread = entry.dgs10_rate - entry.dgs2_rate;
            if entry.hy_oas_spread > 0.0 { last_hy_oas = entry.hy_oas_spread; } else { entry.hy_oas_spread = last_hy_oas; }
        }
    }

    /// Causal Point-in-Time lookup:
    /// For any query timestamp `query_ts_sec` (e.g. 2024-06-10 09:35 ET), returns the strictly prior
    /// macro record (`eod_ts_sec < query_ts_sec`), ensuring ZERO future leakage.
    pub fn query_causal_point_in_time(&self, query_ts_sec: i64) -> Option<&MacroDailyRecord> {
        self.records
            .values()
            .rfind(|r| r.eod_ts_sec < query_ts_sec)
    }

    /// Writes the aligned binary shard `data/shards/context/macro_vix_daily.bin`.
    pub fn write_binary_shard(&self, dest_path: &Path) -> Result<usize, MacroRatesError> {
        if let Some(parent) = dest_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let slice: Vec<MacroDailyRecord> = self.records.values().copied().collect();
        let bytes: &[u8] = bytemuck::cast_slice(&slice);

        let mut file = File::create(dest_path)?;
        // Write 8-byte magic header: b"ARGNMAC1"
        file.write_all(b"ARGNMAC1")?;
        // Write record count as u64
        file.write_all(&(slice.len() as u64).to_le_bytes())?;
        // Write byte payload
        file.write_all(bytes)?;
        file.flush()?;

        info!("Saved {} macro records to binary shard {} ({} bytes)", slice.len(), dest_path.display(), bytes.len() + 16);
        Ok(slice.len())
    }

    /// Reads an aligned binary shard `data/shards/context/macro_vix_daily.bin`.
    pub fn read_binary_shard(path: &Path) -> Result<Vec<MacroDailyRecord>, MacroRatesError> {
        let mut file = File::open(path)?;
        let mut magic = [0u8; 8];
        file.read_exact(&mut magic)?;
        if &magic != b"ARGNMAC1" {
            return Err(MacroRatesError::Csv(format!("Invalid magic header in {}: {:?}", path.display(), magic)));
        }

        let mut len_bytes = [0u8; 8];
        file.read_exact(&mut len_bytes)?;
        let count = u64::from_le_bytes(len_bytes) as usize;

        let record_size = std::mem::size_of::<MacroDailyRecord>();
        let mut buffer = vec![0u8; count * record_size];
        file.read_exact(&mut buffer)?;

        let records: &[MacroDailyRecord] = bytemuck::cast_slice(&buffer);
        Ok(records.to_vec())
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }
}

/// Parses CBOE date strings in formats: "MM/DD/YYYY" or "YYYY-MM-DD".
fn parse_cboe_date(s: &str) -> u32 {
    let clean = s.trim();
    if clean.contains('/') {
        let parts: Vec<&str> = clean.split('/').collect();
        if parts.len() == 3 {
            let m: u32 = parts[0].parse().unwrap_or(0);
            let d: u32 = parts[1].parse().unwrap_or(0);
            let y: u32 = parts[2].parse().unwrap_or(0);
            if y > 1900 && (1..=12).contains(&m) && (1..=31).contains(&d) {
                return y * 10000 + m * 100 + d;
            }
        }
    } else if clean.contains('-') {
        return crate::corporate_actions::parse_date_to_yyyymmdd(clean);
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cboe_vix_parsing_and_geometry() {
        let csv = "DATE,OPEN,HIGH,LOW,CLOSE\n\
                   01/02/1990,17.24,17.24,17.24,17.24\n\
                   06/15/2004,14.50,15.20,14.10,14.95\n";

        let mut surface = MacroRatesSurface::new();
        let count = surface.parse_cboe_vix_csv(csv).expect("parse CBOE VIX");
        assert_eq!(count, 2);

        let rec_1990 = surface.records.get(&19900102).expect("1990 record");
        assert_eq!(rec_1990.vix_close, 17.24);
        assert_eq!(rec_1990.vix_low, 17.24);
        assert_eq!(rec_1990.vix_high, 17.24);

        let rec_2004 = surface.records.get(&20040615).expect("2004 record");
        assert_eq!(rec_2004.vix_close, 14.95);
        assert!(rec_2004.vix_low <= rec_2004.vix_close);
        assert!(rec_2004.vix_high >= rec_2004.vix_close);
    }

    #[test]
    fn test_causal_point_in_time_query() {
        let mut surface = MacroRatesSurface::new();
        surface.records.insert(
            20240607,
            MacroDailyRecord {
                date_yyyymmdd: 20240607,
                eod_ts_sec: 1717790400, // 2024-06-07 16:00 ET
                vix_close: 12.5,
                vix_open: 12.8,
                vix_high: 13.0,
                vix_low: 12.2,
                vix9d_close: 12.0,
                vix3m_close: 14.0,
                vvix_close: 80.0,
                skew_close: 125.0,
                dff_rate: 5.33,
                dgs2_rate: 4.88,
                dgs10_rate: 4.45,
                t10y2y_spread: -0.43,
                hy_oas_spread: 320.0,
                flags: 0,
                _pad0: 0,
            },
        );
        surface.records.insert(
            20240610,
            MacroDailyRecord {
                date_yyyymmdd: 20240610,
                eod_ts_sec: 1718049600, // 2024-06-10 16:00 ET
                vix_close: 13.2,
                vix_open: 12.7,
                vix_high: 13.5,
                vix_low: 12.6,
                vix9d_close: 12.9,
                vix3m_close: 14.5,
                vvix_close: 82.0,
                skew_close: 126.0,
                dff_rate: 5.33,
                dgs2_rate: 4.86,
                dgs10_rate: 4.47,
                t10y2y_spread: -0.39,
                hy_oas_spread: 325.0,
                flags: 0,
                _pad0: 0,
            },
        );

        // At 09:35 ET on 2024-06-10 (1718026500), query MUST return 2024-06-07 (prior day EOD)
        let query_ts = 1718026500;
        let causal_rec = surface.query_causal_point_in_time(query_ts).expect("causal query");
        assert_eq!(causal_rec.date_yyyymmdd, 20240607, "Must return prior day EOD!");
        assert_eq!(causal_rec.vix_close, 12.5);
    }
}
