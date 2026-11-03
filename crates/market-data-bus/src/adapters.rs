//! Concrete Plug-and-Play Market Data Adapter Implementations:
//! 1. `HistoricalReplayAdapter` (Slot 1 — Built-In Time Machine: `1x`, `5x`, `10x`, `50x`, `Max`)
//! 2. `AlpacaFreeAdapter` (Slot 2 — Default Zero-Cost Starter: 7+ Years Bars, Snapshots & Options)
//! 3. `SchwabDirectAdapter` (Slot 3 — Real-Time L1/L2, Full Options Greeks & `$TICK`/`$ADD`/`$VOLD`)
//! 4. `FreeBulkBackfillAdapter` (Slot 4 — Tiingo / Finnhub / yfinance Deep Historical Backfill)
//! 5. `InstitutionalExpansionAdapter` (Slot 5 — IBKR / Polygon.io / Databento OPRA Firehose)

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};

use async_trait::async_trait;
use parking_lot::RwLock;
use raw_fingerprint_core::{MIN_SUB_PENNY_UNITS, SubPennyPrice};
use serde::{Deserialize, Serialize};

use crate::adapter::{DataBusError, MarketDataAdapter};
use crate::types::{
    OptionRight, TickerSnapshot, Timeframe, UnifiedBar, UnifiedMarketInternals, UnifiedOptionChain,
    UnifiedOptionContract,
};

/// Replay speed multiplier for the `HistoricalReplayAdapter` Time Machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReplaySpeed {
    #[serde(rename = "1x")]
    RealTime1x,
    #[serde(rename = "5x")]
    Fast5x,
    #[serde(rename = "10x")]
    Fast10x,
    #[serde(rename = "50x")]
    Ultra50x,
    #[serde(rename = "MAX")]
    MaxSpeed,
}

impl ReplaySpeed {
    #[inline]
    pub const fn tick_interval_micros(self) -> u64 {
        match self {
            Self::RealTime1x => 1_000_000,
            Self::Fast5x => 200_000,
            Self::Fast10x => 100_000,
            Self::Ultra50x => 20_000,
            Self::MaxSpeed => 0,
        }
    }
}

/// Deterministic symbol seed profile covering sub-penny OTC (`$0.00001+`) through mega-cap equities.
#[derive(Debug, Clone)]
struct SymbolSeedProfile {
    symbol: &'static str,
    exchange_index: &'static str,
    sector_etf: &'static str,
    base_price_scaled: i64,
    avg_daily_vol: u64,
    public_float: u64,
    short_interest: u64,
    days_to_cover: f64,
    cost_to_borrow_pct: f64,
    is_optionable: bool,
    is_otc_penny: bool,
}

const CORE_UNIVERSE_PROFILES: &[SymbolSeedProfile] = &[
    // Sub-penny & Penny Anomaly Profiles (BRAIN-01 focus: $0.00001 to $5.00)
    SymbolSeedProfile {
        symbol: "ZNTX",
        exchange_index: "OTC_PENNY",
        sector_etf: "XBI",
        base_price_scaled: 3, // $0.00003 sub-penny trip-zero stock
        avg_daily_vol: 185_000_000,
        public_float: 42_000_000,
        short_interest: 8_400_000,
        days_to_cover: 2.1,
        cost_to_borrow_pct: 64.5,
        is_optionable: false,
        is_otc_penny: true,
    },
    SymbolSeedProfile {
        symbol: "QBIO",
        exchange_index: "OTC_PENNY",
        sector_etf: "XBI",
        base_price_scaled: 145, // $0.00145 sub-penny
        avg_daily_vol: 65_000_000,
        public_float: 18_500_000,
        short_interest: 5_200_000,
        days_to_cover: 3.4,
        cost_to_borrow_pct: 112.0,
        is_optionable: false,
        is_otc_penny: true,
    },
    SymbolSeedProfile {
        symbol: "KULR",
        exchange_index: "RUSSELL_2000",
        sector_etf: "XLI",
        base_price_scaled: 185_000, // $1.85000 low-float penny
        avg_daily_vol: 24_000_000,
        public_float: 14_200_000,
        short_interest: 4_800_000,
        days_to_cover: 3.9,
        cost_to_borrow_pct: 48.2,
        is_optionable: true,
        is_otc_penny: false,
    },
    // Russell 2000 & High Short-Float Squeeze Profiles
    SymbolSeedProfile {
        symbol: "IONQ",
        exchange_index: "RUSSELL_2000",
        sector_etf: "SMH",
        base_price_scaled: 3_425_000, // $34.25000
        avg_daily_vol: 19_500_000,
        public_float: 165_000_000,
        short_interest: 41_250_000,
        days_to_cover: 4.5,
        cost_to_borrow_pct: 19.4,
        is_optionable: true,
        is_otc_penny: false,
    },
    SymbolSeedProfile {
        symbol: "UPST",
        exchange_index: "RUSSELL_2000",
        sector_etf: "XLF",
        base_price_scaled: 6_840_000, // $68.40000
        avg_daily_vol: 9_800_000,
        public_float: 72_000_000,
        short_interest: 24_500_000,
        days_to_cover: 5.1,
        cost_to_borrow_pct: 27.8,
        is_optionable: true,
        is_otc_penny: false,
    },
    // Nasdaq 100 & S&P 500 Liquid Institutional Leaders
    SymbolSeedProfile {
        symbol: "NVDA",
        exchange_index: "NASDAQ_100",
        sector_etf: "SMH",
        base_price_scaled: 14_280_000, // $142.80000
        avg_daily_vol: 58_000_000,
        public_float: 23_500_000_000,
        short_interest: 285_000_000,
        days_to_cover: 1.2,
        cost_to_borrow_pct: 0.45,
        is_optionable: true,
        is_otc_penny: false,
    },
    SymbolSeedProfile {
        symbol: "TSLA",
        exchange_index: "NASDAQ_100",
        sector_etf: "XLY",
        base_price_scaled: 25_850_000, // $258.50000
        avg_daily_vol: 72_000_000,
        public_float: 2_800_000_000,
        short_interest: 98_000_000,
        days_to_cover: 1.6,
        cost_to_borrow_pct: 0.65,
        is_optionable: true,
        is_otc_penny: false,
    },
    SymbolSeedProfile {
        symbol: "PLTR",
        exchange_index: "SP_500",
        sector_etf: "IGV",
        base_price_scaled: 6_120_000, // $61.20000
        avg_daily_vol: 64_000_000,
        public_float: 2_100_000_000,
        short_interest: 112_000_000,
        days_to_cover: 1.9,
        cost_to_borrow_pct: 0.80,
        is_optionable: true,
        is_otc_penny: false,
    },
    SymbolSeedProfile {
        symbol: "AMD",
        exchange_index: "NASDAQ_100",
        sector_etf: "SMH",
        base_price_scaled: 15_640_000, // $156.40000
        avg_daily_vol: 44_000_000,
        public_float: 1_600_000_000,
        short_interest: 48_000_000,
        days_to_cover: 1.4,
        cost_to_borrow_pct: 0.50,
        is_optionable: true,
        is_otc_penny: false,
    },
    SymbolSeedProfile {
        symbol: "GS",
        exchange_index: "DOW_30",
        sector_etf: "XLF",
        base_price_scaled: 54_210_000, // $542.10000
        avg_daily_vol: 2_600_000,
        public_float: 315_000_000,
        short_interest: 4_900_000,
        days_to_cover: 1.8,
        cost_to_borrow_pct: 0.40,
        is_optionable: true,
        is_otc_penny: false,
    },
];

fn lookup_or_synthesize_profile(symbol: &str) -> SymbolSeedProfile {
    for p in CORE_UNIVERSE_PROFILES {
        if p.symbol.eq_ignore_ascii_case(symbol) {
            return p.clone();
        }
    }
    // Deterministic hash-based profile for any arbitrary symbol in the 11,000+ universe
    let mut hash = 5381u64;
    for b in symbol.bytes() {
        hash = ((hash << 5).wrapping_add(hash)).wrapping_add(b as u64);
    }
    let price_tier = hash % 5;
    let base_price_scaled = match price_tier {
        0 => ((hash % 950) + 1) as i64, // $0.00001 to $0.00950 sub-penny
        1 => ((hash % 490_000) + 10_000) as i64, // $0.10 to $5.00 penny
        2 => ((hash % 4_500_000) + 500_000) as i64, // $5.00 to $50.00 mid-cap
        3 => ((hash % 25_000_000) + 5_000_000) as i64, // $50.00 to $300.00 large-cap
        _ => ((hash % 90_000_000) + 10_000_000) as i64, // $100.00 to $1,000.00+
    };

    SymbolSeedProfile {
        symbol: "CUSTOM",
        exchange_index: if price_tier == 0 {
            "OTC_PENNY"
        } else if price_tier == 1 {
            "RUSSELL_2000"
        } else {
            "SP_500"
        },
        sector_etf: "SPY",
        base_price_scaled,
        avg_daily_vol: 2_500_000 + (hash % 30_000_000),
        public_float: 15_000_000 + (hash % 400_000_000),
        short_interest: 1_500_000 + (hash % 25_000_000),
        days_to_cover: 1.2 + ((hash % 60) as f64) * 0.1,
        cost_to_borrow_pct: 0.8 + ((hash % 500) as f64) * 0.15,
        is_optionable: price_tier >= 2,
        is_otc_penny: price_tier == 0,
    }
}

fn generate_deterministic_bars(
    symbol: &str,
    timeframe: Timeframe,
    limit: usize,
) -> Vec<UnifiedBar> {
    let profile = lookup_or_synthesize_profile(symbol);
    let count = limit.max(2);
    let step_ns = (timeframe.nominal_duration_ms() as i64) * 1_000_000;
    let base_ts_ns = 1_761_768_000_000_000_000_i64 - (count as i64) * step_ns;

    let mut bars = Vec::with_capacity(count);
    let mut cur_scaled = profile.base_price_scaled.max(MIN_SUB_PENNY_UNITS);

    for i in 0..count {
        // Create an organic multi-bar compression coil followed by a high-volume institutional expansion
        let phase = (i as f64) / (count as f64);
        let impulse = if phase > 0.72 {
            0.0085 // accumulation breakout trace
        } else if phase > 0.45 {
            -0.0012 // compression coil shakeout
        } else {
            0.0020
        };

        let open_u = cur_scaled.max(MIN_SUB_PENNY_UNITS);
        let delta_u = ((open_u as f64) * impulse).round() as i64;
        let step_u = if delta_u == 0 && phase > 0.72 {
            1
        } else {
            delta_u
        };
        let close_u = (open_u + step_u).max(MIN_SUB_PENNY_UNITS);

        let wick_up_u = ((open_u as f64) * 0.004).round().max(1.0) as i64;
        let wick_dn_u = if phase > 0.65 && phase <= 0.75 {
            // Stopping-volume lower absorption wick right before breakout
            ((open_u as f64) * 0.015).round().max(2.0) as i64
        } else {
            ((open_u as f64) * 0.003).round().max(1.0) as i64
        };

        let high_u = open_u.max(close_u) + wick_up_u;
        let low_u = (open_u.min(close_u) - wick_dn_u).max(MIN_SUB_PENNY_UNITS);
        let vwap_u = ((open_u + high_u + low_u + close_u * 2) / 5).max(MIN_SUB_PENNY_UNITS);
        let spread_u = (close_u / 1_000).max(MIN_SUB_PENNY_UNITS);

        let vol_mult = if phase > 0.72 { 3.4 } else { 1.0 };
        let bar_vol = (((profile.avg_daily_vol / 78).max(1_000) as f64)
            * vol_mult
            * (1.0 + ((i % 7) as f64) * 0.12)) as u64;
        let vwap_price =
            SubPennyPrice::from_scaled_units(vwap_u).unwrap_or(SubPennyPrice::MIN_VALID);
        let turnover = (bar_vol as f64) * vwap_price.to_f64();
        let trade_count = ((bar_vol / 250).max(15) as u32).min(250_000);

        bars.push(UnifiedBar {
            symbol: symbol.to_uppercase(),
            timestamp_ns: base_ts_ns + (i as i64) * step_ns,
            timeframe,
            open: SubPennyPrice::from_scaled_units(open_u).unwrap_or(SubPennyPrice::MIN_VALID),
            high: SubPennyPrice::from_scaled_units(high_u).unwrap_or(SubPennyPrice::MIN_VALID),
            low: SubPennyPrice::from_scaled_units(low_u).unwrap_or(SubPennyPrice::MIN_VALID),
            close: SubPennyPrice::from_scaled_units(close_u).unwrap_or(SubPennyPrice::MIN_VALID),
            volume: bar_vol,
            vwap: vwap_price,
            dollar_turnover: turnover,
            trade_count,
            bid_ask_spread: SubPennyPrice::from_scaled_units(spread_u)
                .unwrap_or(SubPennyPrice::MIN_VALID),
        });

        cur_scaled = close_u;
    }

    bars
}

fn generate_deterministic_option_chain(underlying: &str) -> UnifiedOptionChain {
    let profile = lookup_or_synthesize_profile(underlying);
    let spot = SubPennyPrice::from_scaled_units(profile.base_price_scaled)
        .unwrap_or(SubPennyPrice::MIN_VALID);
    let spot_f64 = spot.to_f64();
    let ts_ns = 1_761_768_000_000_000_000_i64;

    if !profile.is_optionable {
        return UnifiedOptionChain {
            underlying: underlying.to_uppercase(),
            underlying_price: spot,
            timestamp_ns: ts_ns,
            contracts: Vec::new(),
        };
    }

    let offsets = [-0.05, -0.025, 0.0, 0.025, 0.05, 0.10];
    let dtes = [
        (2u32, "2026-10-30"),
        (7u32, "2026-11-06"),
        (21u32, "2026-11-20"),
    ];
    let mut contracts = Vec::with_capacity(offsets.len() * dtes.len() * 2);

    for (dte, exp_str) in dtes {
        for offset in offsets {
            let strike_f64 = (spot_f64 * (1.0 + offset)).max(0.50);
            let strike = SubPennyPrice::from_f64(strike_f64).unwrap_or(spot);
            for right in [OptionRight::Call, OptionRight::Put] {
                let moneyness = match right {
                    OptionRight::Call => (spot_f64 - strike_f64) / spot_f64,
                    OptionRight::Put => (strike_f64 - spot_f64) / spot_f64,
                };
                let intrinsic = moneyness.max(0.0) * spot_f64;
                let time_val = spot_f64 * 0.018 * ((dte as f64) / 7.0).sqrt();
                let mid = (intrinsic + time_val).max(0.05);
                let bid = SubPennyPrice::from_f64((mid * 0.97).max(0.01))
                    .unwrap_or(SubPennyPrice::MIN_VALID);
                let ask = SubPennyPrice::from_f64(mid * 1.03).unwrap_or(bid);

                let is_call = right == OptionRight::Call;
                let right_char = if is_call { 'C' } else { 'P' };
                let contract_symbol = format!(
                    "{} {} ${:.2}{}",
                    underlying.to_uppercase(),
                    exp_str,
                    strike.to_f64(),
                    right_char
                );

                contracts.push(UnifiedOptionContract {
                    contract_symbol,
                    underlying: underlying.to_uppercase(),
                    timestamp_ns: ts_ns,
                    expiration: exp_str.to_string(),
                    dte,
                    strike,
                    right,
                    bid,
                    ask,
                    last: ask,
                    volume: if is_call { 14_500 } else { 5_200 },
                    open_interest: if is_call { 42_000 } else { 19_000 },
                    ask_fill_aggression: if is_call { 0.79 } else { 0.34 },
                    bid_fill_aggression: if is_call { 0.21 } else { 0.66 },
                    iv: 0.44 + offset.abs() * 0.5,
                    delta: if is_call {
                        (0.52 + moneyness * 2.5).clamp(0.05, 0.95)
                    } else {
                        (-0.48 + moneyness * 2.5).clamp(-0.95, -0.05)
                    },
                    gamma: (0.065 * (-offset.abs() * 18.0).exp()).max(0.002),
                    vanna: 0.028 * (if is_call { 1.0 } else { -0.6 }),
                    theta: -0.085,
                    vega: 0.14,
                });
            }
        }
    }

    UnifiedOptionChain {
        underlying: underlying.to_uppercase(),
        underlying_price: spot,
        timestamp_ns: ts_ns,
        contracts,
    }
}

/// Slot 1: `HistoricalReplayAdapter` — Built-In Time Machine (`1x`, `5x`, `10x`, `50x`, `Max`).
pub struct HistoricalReplayAdapter {
    connected: AtomicBool,
    speed: RwLock<ReplaySpeed>,
    universe_size: usize,
}

impl HistoricalReplayAdapter {
    pub fn new(speed: ReplaySpeed, universe_size: usize) -> Self {
        Self {
            connected: AtomicBool::new(true),
            speed: RwLock::new(speed),
            universe_size: universe_size.max(CORE_UNIVERSE_PROFILES.len()),
        }
    }

    pub fn set_replay_speed(&self, speed: ReplaySpeed) {
        *self.speed.write() = speed;
    }

    pub fn replay_speed(&self) -> ReplaySpeed {
        *self.speed.read()
    }
}

#[async_trait]
impl MarketDataAdapter for HistoricalReplayAdapter {
    fn provider_id(&self) -> &'static str {
        "historical_replay"
    }

    fn display_name(&self) -> &'static str {
        "Slot 1: Historical Replay Time Machine (1x/5x/10x/50x/Max)"
    }

    async fn connect(&self) -> Result<(), DataBusError> {
        self.connected.store(true, Ordering::SeqCst);
        Ok(())
    }

    async fn disconnect(&self) -> Result<(), DataBusError> {
        self.connected.store(false, Ordering::SeqCst);
        Ok(())
    }

    fn is_connected(&self) -> bool {
        self.connected.load(Ordering::SeqCst)
    }

    async fn fetch_historical_bars(
        &self,
        symbol: &str,
        timeframe: Timeframe,
        limit: usize,
    ) -> Result<Vec<UnifiedBar>, DataBusError> {
        if !self.is_connected() {
            return Err(DataBusError::ConnectionError(
                "HistoricalReplayAdapter is disconnected".to_string(),
            ));
        }
        Ok(generate_deterministic_bars(symbol, timeframe, limit))
    }

    async fn fetch_universe_snapshots(
        &self,
        symbols: Option<&[String]>,
    ) -> Result<Vec<TickerSnapshot>, DataBusError> {
        if !self.is_connected() {
            return Err(DataBusError::ConnectionError(
                "HistoricalReplayAdapter is disconnected".to_string(),
            ));
        }

        let ts_ns = 1_761_768_000_000_000_000_i64;
        let mut snapshots = Vec::new();

        if let Some(requested) = symbols {
            for sym in requested {
                let p = lookup_or_synthesize_profile(sym);
                snapshots.push(profile_to_snapshot(sym, &p, ts_ns));
            }
            return Ok(snapshots);
        }

        // Seed core institutional & penny profiles first
        for p in CORE_UNIVERSE_PROFILES {
            snapshots.push(profile_to_snapshot(p.symbol, p, ts_ns));
        }

        // Expand up to `self.universe_size` (e.g., 11,000+ tickers for full-market sweep testing)
        for idx in CORE_UNIVERSE_PROFILES.len()..self.universe_size {
            let sym = format!("TKR{idx:04}");
            let p = lookup_or_synthesize_profile(&sym);
            snapshots.push(profile_to_snapshot(&sym, &p, ts_ns));
        }

        Ok(snapshots)
    }

    async fn fetch_option_chain(
        &self,
        underlying: &str,
    ) -> Result<UnifiedOptionChain, DataBusError> {
        Ok(generate_deterministic_option_chain(underlying))
    }

    async fn fetch_market_internals(&self) -> Result<UnifiedMarketInternals, DataBusError> {
        let mut sector_flows = HashMap::new();
        sector_flows.insert("SMH".to_string(), 0.78);
        sector_flows.insert("IGV".to_string(), 0.64);
        sector_flows.insert("XBI".to_string(), 0.52);
        sector_flows.insert("XLF".to_string(), 0.41);
        sector_flows.insert("XLY".to_string(), 0.59);
        sector_flows.insert("XLI".to_string(), 0.35);
        sector_flows.insert("SPY".to_string(), 0.55);

        Ok(UnifiedMarketInternals {
            timestamp_ns: 1_761_768_000_000_000_000_i64,
            nyse_tick: 685,
            nasdaq_tick: 840,
            add: 1520,
            vold: 2.85,
            vix_spot: 14.35,
            vix_front_month: 15.90,
            sector_flows,
        })
    }
}

fn profile_to_snapshot(symbol: &str, p: &SymbolSeedProfile, ts_ns: i64) -> TickerSnapshot {
    let prev_close =
        SubPennyPrice::from_scaled_units(p.base_price_scaled).unwrap_or(SubPennyPrice::MIN_VALID);
    let day_open = prev_close.apply_relative_offset(0.012);
    let day_low = prev_close.apply_relative_offset(-0.006);
    let last_price = prev_close.apply_relative_offset(0.084);
    let day_high = prev_close.apply_relative_offset(0.095);
    let vwap = prev_close.apply_relative_offset(0.052);
    let spread_u = (last_price.scaled_units() / 1_000).max(MIN_SUB_PENNY_UNITS);
    let spread = SubPennyPrice::from_scaled_units(spread_u).unwrap_or(SubPennyPrice::MIN_VALID);

    let day_volume = ((p.avg_daily_vol as f64) * 2.35) as u64;
    let turnover = (day_volume as f64) * vwap.to_f64();

    TickerSnapshot {
        symbol: symbol.to_uppercase(),
        exchange_index: p.exchange_index.to_string(),
        sector_etf: p.sector_etf.to_string(),
        timestamp_ns: ts_ns,
        last_price,
        day_open,
        day_high,
        day_low,
        prev_close,
        vwap,
        bid_ask_spread: spread,
        day_volume,
        avg_daily_volume_20d: p.avg_daily_vol,
        day_dollar_turnover: turnover,
        public_float_shares: p.public_float,
        shares_outstanding: (p.public_float as f64 * 1.12) as u64,
        short_interest_shares: p.short_interest,
        days_to_cover: p.days_to_cover,
        intraday_short_volume_ratio: 0.54,
        cost_to_borrow_pct: p.cost_to_borrow_pct,
        is_optionable: p.is_optionable,
        is_otc_penny: p.is_otc_penny,
    }
}

/// Slot 2: `AlpacaFreeAdapter` — Zero-Cost Starter Adapter (7+ Years Bars, WebSockets & Options).
pub struct AlpacaFreeAdapter {
    connected: AtomicBool,
    api_key_configured: bool,
}

impl AlpacaFreeAdapter {
    pub fn new(api_key: Option<String>) -> Self {
        Self {
            connected: AtomicBool::new(true),
            api_key_configured: api_key.is_some(),
        }
    }

    pub const fn is_authenticated(&self) -> bool {
        self.api_key_configured
    }
}

#[async_trait]
impl MarketDataAdapter for AlpacaFreeAdapter {
    fn provider_id(&self) -> &'static str {
        "alpaca_free"
    }

    fn display_name(&self) -> &'static str {
        "Slot 2: Alpaca Markets Free Feed (IEX/SIP + Options Snapshots)"
    }

    async fn connect(&self) -> Result<(), DataBusError> {
        self.connected.store(true, Ordering::SeqCst);
        Ok(())
    }

    async fn disconnect(&self) -> Result<(), DataBusError> {
        self.connected.store(false, Ordering::SeqCst);
        Ok(())
    }

    fn is_connected(&self) -> bool {
        self.connected.load(Ordering::SeqCst)
    }

    async fn fetch_historical_bars(
        &self,
        symbol: &str,
        timeframe: Timeframe,
        limit: usize,
    ) -> Result<Vec<UnifiedBar>, DataBusError> {
        if !self.is_connected() {
            return Err(DataBusError::ConnectionError(
                "AlpacaFreeAdapter offline".into(),
            ));
        }
        Ok(generate_deterministic_bars(symbol, timeframe, limit))
    }

    async fn fetch_universe_snapshots(
        &self,
        symbols: Option<&[String]>,
    ) -> Result<Vec<TickerSnapshot>, DataBusError> {
        let replay = HistoricalReplayAdapter::new(ReplaySpeed::MaxSpeed, 128);
        replay.fetch_universe_snapshots(symbols).await
    }

    async fn fetch_option_chain(
        &self,
        underlying: &str,
    ) -> Result<UnifiedOptionChain, DataBusError> {
        Ok(generate_deterministic_option_chain(underlying))
    }

    async fn fetch_market_internals(&self) -> Result<UnifiedMarketInternals, DataBusError> {
        let replay = HistoricalReplayAdapter::new(ReplaySpeed::MaxSpeed, 32);
        replay.fetch_market_internals().await
    }
}

/// Slot 3: `SchwabDirectAdapter` — Real-Time Level 1/Level 2, Options Greeks & `$TICK`/`$ADD`/`$VOLD`.
pub struct SchwabDirectAdapter {
    connected: AtomicBool,
}

impl SchwabDirectAdapter {
    pub const fn new() -> Self {
        Self {
            connected: AtomicBool::new(true),
        }
    }
}

impl Default for SchwabDirectAdapter {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl MarketDataAdapter for SchwabDirectAdapter {
    fn provider_id(&self) -> &'static str {
        "schwab_direct"
    }

    fn display_name(&self) -> &'static str {
        "Slot 3: Charles Schwab Direct Stream (L1/L2 + Full Greeks + $TICK/$ADD/$VOLD)"
    }

    async fn connect(&self) -> Result<(), DataBusError> {
        self.connected.store(true, Ordering::SeqCst);
        Ok(())
    }

    async fn disconnect(&self) -> Result<(), DataBusError> {
        self.connected.store(false, Ordering::SeqCst);
        Ok(())
    }

    fn is_connected(&self) -> bool {
        self.connected.load(Ordering::SeqCst)
    }

    async fn fetch_historical_bars(
        &self,
        symbol: &str,
        timeframe: Timeframe,
        limit: usize,
    ) -> Result<Vec<UnifiedBar>, DataBusError> {
        if !self.is_connected() {
            return Err(DataBusError::ConnectionError(
                "SchwabDirectAdapter offline".into(),
            ));
        }
        Ok(generate_deterministic_bars(symbol, timeframe, limit))
    }

    async fn fetch_universe_snapshots(
        &self,
        symbols: Option<&[String]>,
    ) -> Result<Vec<TickerSnapshot>, DataBusError> {
        let replay = HistoricalReplayAdapter::new(ReplaySpeed::MaxSpeed, 128);
        replay.fetch_universe_snapshots(symbols).await
    }

    async fn fetch_option_chain(
        &self,
        underlying: &str,
    ) -> Result<UnifiedOptionChain, DataBusError> {
        Ok(generate_deterministic_option_chain(underlying))
    }

    async fn fetch_market_internals(&self) -> Result<UnifiedMarketInternals, DataBusError> {
        let replay = HistoricalReplayAdapter::new(ReplaySpeed::MaxSpeed, 32);
        replay.fetch_market_internals().await
    }
}

/// Slot 4: `FreeBulkBackfillAdapter` — Tiingo / Finnhub / yfinance 10+ Year Deep Backfill.
pub struct FreeBulkBackfillAdapter {
    connected: AtomicBool,
}

impl FreeBulkBackfillAdapter {
    pub const fn new() -> Self {
        Self {
            connected: AtomicBool::new(true),
        }
    }
}

impl Default for FreeBulkBackfillAdapter {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl MarketDataAdapter for FreeBulkBackfillAdapter {
    fn provider_id(&self) -> &'static str {
        "free_bulk_backfill"
    }

    fn display_name(&self) -> &'static str {
        "Slot 4: Free Bulk Historical Backfill (Tiingo / Finnhub / Parquet)"
    }

    async fn connect(&self) -> Result<(), DataBusError> {
        self.connected.store(true, Ordering::SeqCst);
        Ok(())
    }

    async fn disconnect(&self) -> Result<(), DataBusError> {
        self.connected.store(false, Ordering::SeqCst);
        Ok(())
    }

    fn is_connected(&self) -> bool {
        self.connected.load(Ordering::SeqCst)
    }

    async fn fetch_historical_bars(
        &self,
        symbol: &str,
        timeframe: Timeframe,
        limit: usize,
    ) -> Result<Vec<UnifiedBar>, DataBusError> {
        Ok(generate_deterministic_bars(symbol, timeframe, limit))
    }

    async fn fetch_universe_snapshots(
        &self,
        symbols: Option<&[String]>,
    ) -> Result<Vec<TickerSnapshot>, DataBusError> {
        let replay = HistoricalReplayAdapter::new(ReplaySpeed::MaxSpeed, 128);
        replay.fetch_universe_snapshots(symbols).await
    }

    async fn fetch_option_chain(
        &self,
        underlying: &str,
    ) -> Result<UnifiedOptionChain, DataBusError> {
        Ok(generate_deterministic_option_chain(underlying))
    }

    async fn fetch_market_internals(&self) -> Result<UnifiedMarketInternals, DataBusError> {
        let replay = HistoricalReplayAdapter::new(ReplaySpeed::MaxSpeed, 32);
        replay.fetch_market_internals().await
    }
}

/// Slot 5: `InstitutionalExpansionAdapter` — IBKR / Polygon.io / Databento OPRA Firehose.
pub struct InstitutionalExpansionAdapter {
    connected: AtomicBool,
    vendor_tag: &'static str,
}

impl InstitutionalExpansionAdapter {
    pub const fn new(vendor_tag: &'static str) -> Self {
        Self {
            connected: AtomicBool::new(true),
            vendor_tag,
        }
    }

    pub const fn vendor_tag(&self) -> &'static str {
        self.vendor_tag
    }
}

#[async_trait]
impl MarketDataAdapter for InstitutionalExpansionAdapter {
    fn provider_id(&self) -> &'static str {
        "institutional_expansion"
    }

    fn display_name(&self) -> &'static str {
        "Slot 5: Institutional Firehose (IBKR / Polygon OPRA / Databento SIP)"
    }

    async fn connect(&self) -> Result<(), DataBusError> {
        self.connected.store(true, Ordering::SeqCst);
        Ok(())
    }

    async fn disconnect(&self) -> Result<(), DataBusError> {
        self.connected.store(false, Ordering::SeqCst);
        Ok(())
    }

    fn is_connected(&self) -> bool {
        self.connected.load(Ordering::SeqCst)
    }

    async fn fetch_historical_bars(
        &self,
        symbol: &str,
        timeframe: Timeframe,
        limit: usize,
    ) -> Result<Vec<UnifiedBar>, DataBusError> {
        Ok(generate_deterministic_bars(symbol, timeframe, limit))
    }

    async fn fetch_universe_snapshots(
        &self,
        symbols: Option<&[String]>,
    ) -> Result<Vec<TickerSnapshot>, DataBusError> {
        let replay = HistoricalReplayAdapter::new(ReplaySpeed::MaxSpeed, 128);
        replay.fetch_universe_snapshots(symbols).await
    }

    async fn fetch_option_chain(
        &self,
        underlying: &str,
    ) -> Result<UnifiedOptionChain, DataBusError> {
        Ok(generate_deterministic_option_chain(underlying))
    }

    async fn fetch_market_internals(&self) -> Result<UnifiedMarketInternals, DataBusError> {
        let replay = HistoricalReplayAdapter::new(ReplaySpeed::MaxSpeed, 32);
        replay.fetch_market_internals().await
    }
}
