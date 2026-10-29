//! Standardized Provider-Agnostic Market Data Structures (`UnifiedBar`, `UnifiedTick`,
//! `UnifiedOptionChain`, `UnifiedMarketInternals`, and `TickerSnapshot`).

use std::collections::HashMap;

use raw_fingerprint_core::{
    RawBarRecord, RawMarketInternalsSnapshot, RawOptionsSnapshot, RawShortFloatSnapshot,
    SubPennyPrice,
};
use serde::{Deserialize, Serialize};

/// Multi-scale temporal resolution grid supported across the 5 AI Brains.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Timeframe {
    #[serde(rename = "Tick")]
    Tick,
    #[serde(rename = "1m")]
    M1,
    #[serde(rename = "3m")]
    M3,
    #[serde(rename = "5m")]
    M5,
    #[serde(rename = "15m")]
    M15,
    #[serde(rename = "1H")]
    H1,
    #[serde(rename = "4H")]
    H4,
    #[serde(rename = "1D")]
    D1,
    #[serde(rename = "1W")]
    W1,
    #[serde(rename = "1M")]
    Mo1,
}

impl Timeframe {
    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Tick => "Tick",
            Self::M1 => "1m",
            Self::M3 => "3m",
            Self::M5 => "5m",
            Self::M15 => "15m",
            Self::H1 => "1H",
            Self::H4 => "4H",
            Self::D1 => "1D",
            Self::W1 => "1W",
            Self::Mo1 => "1M",
        }
    }

    #[inline]
    pub const fn nominal_duration_ms(self) -> u64 {
        match self {
            Self::Tick => 100,
            Self::M1 => 60_000,
            Self::M3 => 180_000,
            Self::M5 => 300_000,
            Self::M15 => 900_000,
            Self::H1 => 3_600_000,
            Self::H4 => 14_400_000,
            Self::D1 => 86_400_000,
            Self::W1 => 604_800_000,
            Self::Mo1 => 2_592_000_000,
        }
    }
}

/// Standardized OHLCV + Microstructure Bar (`Stream 1`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UnifiedBar {
    pub symbol: String,
    pub timestamp_ns: i64,
    pub timeframe: Timeframe,
    pub open: SubPennyPrice,
    pub high: SubPennyPrice,
    pub low: SubPennyPrice,
    pub close: SubPennyPrice,
    pub volume: u64,
    pub vwap: SubPennyPrice,
    pub dollar_turnover: f64,
    pub trade_count: u32,
    pub bid_ask_spread: SubPennyPrice,
}

impl UnifiedBar {
    /// Converts directly into `raw_fingerprint_core::RawBarRecord` for zero-skew tensor creation.
    #[inline]
    pub fn to_raw_bar_record(&self) -> RawBarRecord {
        RawBarRecord {
            timestamp_ns: self.timestamp_ns,
            open: self.open,
            high: self.high,
            low: self.low,
            close: self.close,
            volume: self.volume,
            vwap: self.vwap,
            dollar_turnover: self.dollar_turnover,
            trade_count: self.trade_count,
            bid_ask_spread: self.bid_ask_spread,
            time_delta_ms: self.timeframe.nominal_duration_ms(),
        }
    }
}

/// Standardized Tape Trade Tick (`Stream 1`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UnifiedTick {
    pub symbol: String,
    pub timestamp_ns: i64,
    pub price: SubPennyPrice,
    pub size: u32,
    pub exchange: String,
    pub conditions: Vec<String>,
    pub bid: SubPennyPrice,
    pub ask: SubPennyPrice,
}

/// Option contract right (`Call` or `Put`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OptionRight {
    #[serde(rename = "C")]
    Call,
    #[serde(rename = "P")]
    Put,
}

/// Single strike/expiration contract within a `UnifiedOptionChain`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UnifiedOptionContract {
    pub contract_symbol: String,
    pub underlying: String,
    pub timestamp_ns: i64,
    pub expiration: String,
    pub dte: u32,
    pub strike: SubPennyPrice,
    pub right: OptionRight,
    pub bid: SubPennyPrice,
    pub ask: SubPennyPrice,
    pub last: SubPennyPrice,
    pub volume: u64,
    pub open_interest: u64,
    pub ask_fill_aggression: f64,
    pub bid_fill_aggression: f64,
    pub iv: f64,
    pub delta: f64,
    pub gamma: f64,
    pub vanna: f64,
    pub theta: f64,
    pub vega: f64,
}

impl UnifiedOptionContract {
    /// Mid-market premium in exact 5-decimal sub-penny resolution.
    #[inline]
    pub fn mid_premium(&self) -> SubPennyPrice {
        let mid_u = (self.bid.scaled_units() + self.ask.scaled_units()) / 2;
        SubPennyPrice::from_scaled_units_allow_zero(mid_u.max(0)).unwrap_or(self.last)
    }
}

/// Full Underlying Option Chain (`Stream 2`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UnifiedOptionChain {
    pub underlying: String,
    pub underlying_price: SubPennyPrice,
    pub timestamp_ns: i64,
    pub contracts: Vec<UnifiedOptionContract>,
}

impl UnifiedOptionChain {
    /// Aggregates strike-by-strike option contracts into a `RawOptionsSnapshot` for Stream 2 physics.
    pub fn to_raw_options_snapshot(&self) -> RawOptionsSnapshot {
        if self.contracts.is_empty() {
            return RawOptionsSnapshot {
                timestamp_ns: self.timestamp_ns,
                dominant_gamma_wall_strike: self.underlying_price,
                ..RawOptionsSnapshot::default()
            };
        }

        let mut call_vol = 0u64;
        let mut put_vol = 0u64;
        let mut total_oi = 0u64;
        let mut weighted_ask_agg = 0.0_f64;
        let mut weighted_bid_agg = 0.0_f64;
        let mut net_gex = 0.0_f64;
        let mut net_vanna = 0.0_f64;
        let mut zero_dte_vol = 0u64;
        let mut front_iv_sum = 0.0_f64;
        let mut front_iv_cnt = 0usize;
        let mut back_iv_sum = 0.0_f64;
        let mut back_iv_cnt = 0usize;
        let mut max_gamma_score = -1.0_f64;
        let mut dominant_strike = self.underlying_price;

        let spot = self.underlying_price.to_f64().max(0.00001);

        for c in &self.contracts {
            match c.right {
                OptionRight::Call => call_vol += c.volume,
                OptionRight::Put => put_vol += c.volume,
            }
            total_oi += c.open_interest;
            let vol_f = c.volume as f64;
            weighted_ask_agg += c.ask_fill_aggression * vol_f;
            weighted_bid_agg += c.bid_fill_aggression * vol_f;

            let sign = match c.right {
                OptionRight::Call => 1.0,
                OptionRight::Put => -1.0,
            };
            let strike_gex = (c.open_interest as f64) * c.gamma * spot * spot * 0.01 * sign / 1e6;
            net_gex += strike_gex;
            net_vanna += (c.open_interest as f64) * c.vanna * sign / 1e5;

            if c.dte <= 5 {
                zero_dte_vol += c.volume;
                front_iv_sum += c.iv;
                front_iv_cnt += 1;
            } else {
                back_iv_sum += c.iv;
                back_iv_cnt += 1;
            }

            let abs_gamma_oi = (c.open_interest as f64) * c.gamma.abs();
            if abs_gamma_oi > max_gamma_score {
                max_gamma_score = abs_gamma_oi;
                dominant_strike = c.strike;
            }
        }

        let total_vol = (call_vol + put_vol).max(1) as f64;
        let front_iv = if front_iv_cnt > 0 {
            front_iv_sum / (front_iv_cnt as f64)
        } else {
            0.30
        };
        let back_iv = if back_iv_cnt > 0 {
            back_iv_sum / (back_iv_cnt as f64)
        } else {
            front_iv
        };

        RawOptionsSnapshot {
            timestamp_ns: self.timestamp_ns,
            call_volume: call_vol,
            put_volume: put_vol,
            total_open_interest: total_oi.max(1),
            ask_fill_aggression_ratio: (weighted_ask_agg / total_vol).clamp(0.0, 1.0),
            bid_fill_aggression_ratio: (weighted_bid_agg / total_vol).clamp(0.0, 1.0),
            net_dealer_gamma_gex: net_gex,
            net_dealer_vanna: net_vanna,
            zero_dte_volume_share: ((zero_dte_vol as f64) / total_vol).clamp(0.0, 1.0),
            front_iv,
            back_iv,
            dominant_gamma_wall_strike: dominant_strike,
        }
    }
}

/// Standardized Market Internals Heartbeat (`Stream 4`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UnifiedMarketInternals {
    pub timestamp_ns: i64,
    pub nyse_tick: i32,
    pub nasdaq_tick: i32,
    pub add: i32,
    pub vold: f64,
    pub vix_spot: f64,
    pub vix_front_month: f64,
    pub sector_flows: HashMap<String, f64>,
}

impl UnifiedMarketInternals {
    #[inline]
    pub fn to_raw_internals_snapshot(&self, sector_etf: &str) -> RawMarketInternalsSnapshot {
        let sector_flow = self.sector_flows.get(sector_etf).copied().unwrap_or(0.0);
        RawMarketInternalsSnapshot {
            timestamp_ns: self.timestamp_ns,
            nyse_tick: self.nyse_tick,
            nasdaq_tick: self.nasdaq_tick,
            add_breadth: self.add,
            vold_up_down_ratio: self.vold,
            vix_spot: self.vix_spot,
            vix_front_month: self.vix_front_month,
            sector_relative_flow: sector_flow,
        }
    }
}

/// Wide-Net Market Snapshot for a single ticker in the 11,000+ Equity/OTC Universe.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TickerSnapshot {
    pub symbol: String,
    pub exchange_index: String,
    pub sector_etf: String,
    pub timestamp_ns: i64,
    pub last_price: SubPennyPrice,
    pub day_open: SubPennyPrice,
    pub day_high: SubPennyPrice,
    pub day_low: SubPennyPrice,
    pub prev_close: SubPennyPrice,
    pub vwap: SubPennyPrice,
    pub bid_ask_spread: SubPennyPrice,
    pub day_volume: u64,
    pub avg_daily_volume_20d: u64,
    pub day_dollar_turnover: f64,
    pub public_float_shares: u64,
    pub shares_outstanding: u64,
    pub short_interest_shares: u64,
    pub days_to_cover: f64,
    pub intraday_short_volume_ratio: f64,
    pub cost_to_borrow_pct: f64,
    pub is_optionable: bool,
    pub is_otc_penny: bool,
}

impl TickerSnapshot {
    /// Extracts the `RawShortFloatSnapshot` (`Stream 3`) from this ticker's live snapshot.
    #[inline]
    pub fn to_raw_short_float_snapshot(&self) -> RawShortFloatSnapshot {
        RawShortFloatSnapshot {
            timestamp_ns: self.timestamp_ns,
            public_float_shares: self.public_float_shares.max(1),
            shares_outstanding: self.shares_outstanding.max(self.public_float_shares).max(1),
            short_interest_shares: self.short_interest_shares,
            days_to_cover: self.days_to_cover,
            intraday_short_volume_ratio: self.intraday_short_volume_ratio,
            cost_to_borrow_annual_pct: self.cost_to_borrow_pct,
            cumulative_session_volume: self.day_volume,
        }
    }
}

/// Unified real-time event emitted over the Market Data Bus broadcast channel.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MarketStreamEvent {
    Bar(UnifiedBar),
    Tick(UnifiedTick),
    OptionChain(UnifiedOptionChain),
    Internals(UnifiedMarketInternals),
}
