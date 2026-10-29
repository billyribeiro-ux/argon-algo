//! Data structures for the 4 Synchronized Raw Fingerprint Streams and 5 Selectable AI Brains.
//!
//! Zero lagging human indicators are permitted in these structures. All fields represent
//! pure physical market observations (chart/tape, options gamma/vanna, short/float, internals).

use serde::{Deserialize, Serialize};

use crate::price::SubPennyPrice;

/// The 5 Selectable Specialized Goal-Horizon AI Brains.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BrainHorizonId {
    /// `BRAIN-01`: Penny & Sub-Penny Anomaly Brain (`$0.00001` to `$5.00`, Tick/1m/3m/5m/1D)
    #[serde(rename = "BRAIN-01")]
    Brain01PennyAnomaly,
    /// `BRAIN-02`: Intraday Day-Trading Brain (`5m to 6.5h`, flat by 4:00 PM EST close)
    #[serde(rename = "BRAIN-02")]
    Brain02IntradayDayTrade,
    /// `BRAIN-03`: Short-Term Swing Brain (`1 to 5 Trading Days`, 15m/1H/4H/1D)
    #[serde(rename = "BRAIN-03")]
    Brain03ShortTermSwing,
    /// `BRAIN-04`: Core Swing Trading Brain (`1 to 8 Weeks`, 4H/1D/1W)
    #[serde(rename = "BRAIN-04")]
    Brain04CoreSwing,
    /// `BRAIN-05`: Long-Term Portfolio & Wealth Brain (`6 Months to 3+ Years`, 1D/1W/1M)
    #[serde(rename = "BRAIN-05")]
    Brain05LongTermWealth,
}

impl BrainHorizonId {
    #[inline]
    pub const fn code(self) -> &'static str {
        match self {
            Self::Brain01PennyAnomaly => "BRAIN-01",
            Self::Brain02IntradayDayTrade => "BRAIN-02",
            Self::Brain03ShortTermSwing => "BRAIN-03",
            Self::Brain04CoreSwing => "BRAIN-04",
            Self::Brain05LongTermWealth => "BRAIN-05",
        }
    }

    #[inline]
    pub const fn display_name(self) -> &'static str {
        match self {
            Self::Brain01PennyAnomaly => "Penny & Sub-Penny Anomaly Algo",
            Self::Brain02IntradayDayTrade => "Intraday Day-Trading Algo",
            Self::Brain03ShortTermSwing => "Short-Term Swing Algo",
            Self::Brain04CoreSwing => "Core Swing Trading Algo",
            Self::Brain05LongTermWealth => "Long-Term Portfolio & Wealth Algo",
        }
    }

    #[inline]
    pub const fn default_hold_duration_ms(self) -> i64 {
        match self {
            Self::Brain01PennyAnomaly => 45 * 60 * 1_000, // 45 mins
            Self::Brain02IntradayDayTrade => 75 * 60 * 1_000, // 1h 15m
            Self::Brain03ShortTermSwing => 3 * 24 * 3600 * 1_000, // 3 days
            Self::Brain04CoreSwing => 21 * 24 * 3600 * 1_000, // 3 weeks
            Self::Brain05LongTermWealth => 240 * 24 * 3600 * 1_000, // 8 months
        }
    }
}

/// Trade direction (`LONG` or `SHORT`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum TradeDirection {
    Long,
    Short,
}

impl TradeDirection {
    #[inline]
    pub const fn sign(self) -> f64 {
        match self {
            Self::Long => 1.0,
            Self::Short => -1.0,
        }
    }

    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Long => "LONG",
            Self::Short => "SHORT",
        }
    }
}

/// Primary raw fingerprint driver discovered by the model's 4-stream attribution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PrimaryFingerprintDriver {
    #[serde(rename = "VOLUME_ABSORPTION")]
    VolumeAbsorption,
    #[serde(rename = "GAMMA_SQUEEZE")]
    GammaSqueeze,
    #[serde(rename = "SHORT_SQUEEZE")]
    ShortSqueeze,
    #[serde(rename = "INTERNALS_ALIGNMENT")]
    InternalsAlignment,
}

impl PrimaryFingerprintDriver {
    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::VolumeAbsorption => "VOLUME_ABSORPTION",
            Self::GammaSqueeze => "GAMMA_SQUEEZE",
            Self::ShortSqueeze => "SHORT_SQUEEZE",
            Self::InternalsAlignment => "INTERNALS_ALIGNMENT",
        }
    }
}

/// Real-time AI Trace Health status evaluated bar-by-bar for every open position.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TraceHealthBadge {
    /// `🟢 TRACE ACCELERATING (HOLD)`
    #[serde(rename = "TRACE_ACCELERATING")]
    Accelerating,
    /// `🟡 T1 HIT - TRAILING STOP RAISED`
    #[serde(rename = "TRAILING_T1")]
    TrailingTarget1,
    /// `🟠 BUYING EXHAUSTION DETECTED - PREPARING EXIT`
    #[serde(rename = "EXHAUSTION_WARNING")]
    ExhaustionWarning,
    /// Immediate dynamic exit triggered due to opposing absorption or trace invalidation
    #[serde(rename = "DYNAMIC_EXIT_TRIGGERED")]
    DynamicExitTriggered,
}

impl TraceHealthBadge {
    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Accelerating => "TRACE_ACCELERATING",
            Self::TrailingTarget1 => "TRAILING_T1",
            Self::ExhaustionWarning => "EXHAUSTION_WARNING",
            Self::DynamicExitTriggered => "DYNAMIC_EXIT_TRIGGERED",
        }
    }
}

/// Stream 1: Raw Chart & Tape Physics observation for a single bar/tick slice.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RawBarRecord {
    pub timestamp_ns: i64,
    pub open: SubPennyPrice,
    pub high: SubPennyPrice,
    pub low: SubPennyPrice,
    pub close: SubPennyPrice,
    pub volume: u64,
    pub vwap: SubPennyPrice,
    pub dollar_turnover: f64,
    pub trade_count: u32,
    pub bid_ask_spread: SubPennyPrice,
    pub time_delta_ms: u64,
}

/// Stream 2: Raw Options & Dealer Positioning (Gamma / Vanna Squeeze Physics).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RawOptionsSnapshot {
    pub timestamp_ns: i64,
    pub call_volume: u64,
    pub put_volume: u64,
    pub total_open_interest: u64,
    /// Volume executed at or above the ask (`0.0` to `1.0`).
    pub ask_fill_aggression_ratio: f64,
    /// Volume executed at or below the bid (`0.0` to `1.0`).
    pub bid_fill_aggression_ratio: f64,
    /// Net Dealer Gamma Exposure (GEX) normalized per 1% underlying move ($ millions).
    pub net_dealer_gamma_gex: f64,
    /// Net Dealer Vanna sensitivity ($\partial \Delta / \partial \sigma$).
    pub net_dealer_vanna: f64,
    /// Fraction of option volume in 0-5 DTE expirations (`0.0` to `1.0`).
    pub zero_dte_volume_share: f64,
    /// Front-month ATM Implied Volatility (`0.01` to `5.0+`).
    pub front_iv: f64,
    /// Back-month ATM Implied Volatility (`0.01` to `5.0+`).
    pub back_iv: f64,
    /// Dominant strike with maximum absolute gamma concentration (`$0.00001+`).
    pub dominant_gamma_wall_strike: SubPennyPrice,
}

impl Default for RawOptionsSnapshot {
    fn default() -> Self {
        Self {
            timestamp_ns: 0,
            call_volume: 0,
            put_volume: 0,
            total_open_interest: 1,
            ask_fill_aggression_ratio: 0.5,
            bid_fill_aggression_ratio: 0.5,
            net_dealer_gamma_gex: 0.0,
            net_dealer_vanna: 0.0,
            zero_dte_volume_share: 0.0,
            front_iv: 0.25,
            back_iv: 0.25,
            dominant_gamma_wall_strike: SubPennyPrice::MIN_VALID,
        }
    }
}

/// Stream 3: Raw Short & Float Mechanics (Short Squeeze vs. Dilution Trap Physics).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RawShortFloatSnapshot {
    pub timestamp_ns: i64,
    pub public_float_shares: u64,
    pub shares_outstanding: u64,
    pub short_interest_shares: u64,
    pub days_to_cover: f64,
    pub intraday_short_volume_ratio: f64,
    pub cost_to_borrow_annual_pct: f64,
    pub cumulative_session_volume: u64,
}

impl Default for RawShortFloatSnapshot {
    fn default() -> Self {
        Self {
            timestamp_ns: 0,
            public_float_shares: 50_000_000,
            shares_outstanding: 55_000_000,
            short_interest_shares: 2_500_000,
            days_to_cover: 1.5,
            intraday_short_volume_ratio: 0.45,
            cost_to_borrow_annual_pct: 1.2,
            cumulative_session_volume: 1_000_000,
        }
    }
}

/// Stream 4: Raw Market Internals (The Macro Tide).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RawMarketInternalsSnapshot {
    pub timestamp_ns: i64,
    /// NYSE net upticks minus downticks (typically `-2000` to `+2000`).
    pub nyse_tick: i32,
    /// Nasdaq net upticks minus downticks (typically `-2000` to `+2000`).
    pub nasdaq_tick: i32,
    /// Advance-Decline net differential (`$ADD`).
    pub add_breadth: i32,
    /// Up-Volume to Down-Volume ratio (`$VOLD`, e.g., `2.5` or `0.4`).
    pub vold_up_down_ratio: f64,
    /// Spot VIX index level.
    pub vix_spot: f64,
    /// Front-month VIX futures level.
    pub vix_front_month: f64,
    /// Target ticker's sector ETF relative net flow (`-1.0` to `+1.0`).
    pub sector_relative_flow: f64,
}

impl Default for RawMarketInternalsSnapshot {
    fn default() -> Self {
        Self {
            timestamp_ns: 0,
            nyse_tick: 0,
            nasdaq_tick: 0,
            add_breadth: 0,
            vold_up_down_ratio: 1.0,
            vix_spot: 15.0,
            vix_front_month: 16.0,
            sector_relative_flow: 0.0,
        }
    }
}

/// Normalized 4-Stream attribution weights summing to `1.0`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FourStreamWeights {
    pub chart_tape_physics_weight: f32,
    pub options_gamma_vanna_weight: f32,
    pub short_float_squeeze_weight: f32,
    pub market_internals_macro_weight: f32,
}

impl FourStreamWeights {
    /// Returns the dominant stream driver from the 4 normalized stream weights.
    #[inline]
    pub fn primary_driver(&self) -> PrimaryFingerprintDriver {
        let mut max_w = self.chart_tape_physics_weight;
        let mut driver = PrimaryFingerprintDriver::VolumeAbsorption;

        if self.options_gamma_vanna_weight > max_w {
            max_w = self.options_gamma_vanna_weight;
            driver = PrimaryFingerprintDriver::GammaSqueeze;
        }
        if self.short_float_squeeze_weight > max_w {
            max_w = self.short_float_squeeze_weight;
            driver = PrimaryFingerprintDriver::ShortSqueeze;
        }
        if self.market_internals_macro_weight > max_w {
            driver = PrimaryFingerprintDriver::InternalsAlignment;
        }
        driver
    }
}

/// Complete autonomous signal evaluation output from the Raw Fingerprint Engine.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AutonomousSignalBlueprint {
    pub symbol: String,
    pub brain_id: BrainHorizonId,
    pub direction: TradeDirection,
    pub timestamp_ns: i64,
    pub entry_price: SubPennyPrice,
    pub structural_stop_price: SubPennyPrice,
    pub target_1_price: SubPennyPrice,
    pub target_2_price: SubPennyPrice,
    pub predicted_hold_duration_ms: i64,
    pub ai_confidence_pct: f64,
    pub expected_r_multiple: f64,
    pub primary_driver: PrimaryFingerprintDriver,
    pub stream_weights: FourStreamWeights,
    pub latent_embedding_128: Vec<f32>,
}
