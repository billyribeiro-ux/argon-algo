//! Phase 36 Pillar 4: Microprice & Queue-Depletion Adverse-Selection Execution Router
//!
//! Deterministically routes approved equity/option tickets using causal L1 queue-imbalance and
//! Stream-3 orderflow toxicity (`bid_price`, `ask_price`, `bid_size`, `ask_size`,
//! `orderflow_toxicity_vpin`, `atr_bar`, `direction`, `regime_bucket`):
//! - Volume-Weighted Microprice:
//!   $$P_{\text{micro}} = \frac{Q_{\text{bid}} \cdot P_{\text{ask}} + Q_{\text{ask}} \cdot P_{\text{bid}}}{Q_{\text{bid}} + Q_{\text{ask}}}$$
//! - Signed Microprice Alpha Skew:
//!   $$\delta_{\text{micro}} = \text{dir} \cdot \frac{P_{\text{micro}} - P_{\text{mid}}}{\max(P_{\text{ask}} - P_{\text{bid}}, 10^{-4})} \in [-0.5, +0.5]$$
//! - Adverse-Selection Urgency Score:
//!   $$u_{\text{exec}} = \operatorname{sigmoid}\big(3.0 \cdot \delta_{\text{micro}} + 2.2 \cdot (\text{toxicity} - 0.50) - 2.5 \cdot (s_{\text{atr}} - 0.12)\big) \in [0, 1]$$

use crate::autopsy_evolver::RegimeBucket;
use raw_fingerprint_core::TradeDirection;
use serde::{Deserialize, Serialize};

/// Deterministic microstructure order-execution tactic selected by `MicropriceExecutionRouter`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionTactic {
    #[serde(rename = "PASSIVE_MID_PEG_LIMIT")]
    PassiveMidPegLimit,
    #[serde(rename = "ADAPTIVE_MICROPRICE_PEG")]
    AdaptiveMicropricePeg,
    #[serde(rename = "IMMEDIATE_ALMGREN_CHRISS_SWEEP")]
    ImmediateAlmgrenChrissSweep,
}

impl ExecutionTactic {
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::PassiveMidPegLimit => "PASSIVE_MID_PEG_LIMIT",
            Self::AdaptiveMicropricePeg => "ADAPTIVE_MICROPRICE_PEG",
            Self::ImmediateAlmgrenChrissSweep => "IMMEDIATE_ALMGREN_CHRISS_SWEEP",
        }
    }
}

/// Causal L1 + Stream-3 microstructure quote and toxicity inputs.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MicrostructureQuoteInput {
    pub bid_price: f64,
    pub ask_price: f64,
    pub bid_size: f64,
    pub ask_size: f64,
    pub orderflow_toxicity_vpin: f64,
    pub atr_bar: f64,
    pub structural_risk_per_share: f64,
    pub direction: TradeDirection,
    pub regime_bucket: RegimeBucket,
}

/// Concrete limit price, tactic, step-up chase, and expected slippage telemetry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MicropriceExecutionPlan {
    pub execution_tactic: ExecutionTactic,
    pub mid_price: f64,
    pub microprice: f64,
    pub microprice_limit_price: f64,
    pub microprice_alpha_skew: f64,
    pub spread_to_atr_ratio: f64,
    pub execution_urgency_score: f64,
    pub step_up_chase_ms: u64,
    pub expected_slippage_bps: f64,
    pub expected_slippage_r: f64,
}

pub struct MicropriceExecutionRouter;

impl MicropriceExecutionRouter {
    #[inline]
    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x.clamp(-25.0, 25.0)).exp())
    }

    /// Computes Volume-Weighted Microprice $P_{\text{micro}}$, signed skew $\delta_{\text{micro}}$,
    /// urgency $u_{\text{exec}}$, and optimal execution tactic + limit price.
    pub fn route_order(input: &MicrostructureQuoteInput) -> MicropriceExecutionPlan {
        let bid = input.bid_price.max(0.0001);
        let ask = input.ask_price.max(bid + 1e-4);
        let spread = (ask - bid).max(1e-4);
        let mid_price = 0.5 * (bid + ask);

        let q_bid = input.bid_size.max(1.0);
        let q_ask = input.ask_size.max(1.0);
        let microprice = (q_bid * ask + q_ask * bid) / (q_bid + q_ask);

        let dir = input.direction.sign();
        let microprice_alpha_skew = (dir * (microprice - mid_price) / spread).clamp(-0.5, 0.5);
        let spread_to_atr_ratio = spread / input.atr_bar.max(1e-4);
        let toxicity = input.orderflow_toxicity_vpin.clamp(0.0, 1.0);

        let logit = 3.0 * microprice_alpha_skew + 2.2 * (toxicity - 0.50)
            - 2.5 * (spread_to_atr_ratio - 0.12);
        let execution_urgency_score = Self::sigmoid(logit);

        let (execution_tactic, raw_limit_px, step_up_chase_ms, expected_slip_dollars) =
            if execution_urgency_score < 0.38
                || (input.regime_bucket == RegimeBucket::PosGammaPinMeanRevert
                    && spread_to_atr_ratio > 0.15)
            {
                (
                    ExecutionTactic::PassiveMidPegLimit,
                    mid_price,
                    0_u64,
                    0.10 * spread,
                )
            } else if execution_urgency_score < 0.72 {
                (
                    ExecutionTactic::AdaptiveMicropricePeg,
                    microprice,
                    1_500_u64,
                    (microprice - mid_price).abs() + 0.20 * spread,
                )
            } else {
                let sweep_cap = if dir > 0.0 {
                    ask + 0.10 * spread
                } else {
                    (bid - 0.10 * spread).max(0.0001)
                };
                (
                    ExecutionTactic::ImmediateAlmgrenChrissSweep,
                    sweep_cap,
                    0_u64,
                    0.55 * spread,
                )
            };

        let expected_slippage_bps = (expected_slip_dollars / mid_price.max(1e-4)) * 10_000.0;
        let expected_slippage_r =
            expected_slip_dollars / input.structural_risk_per_share.abs().max(1e-4);

        MicropriceExecutionPlan {
            execution_tactic,
            mid_price,
            microprice,
            microprice_limit_price: raw_limit_px,
            microprice_alpha_skew,
            spread_to_atr_ratio,
            execution_urgency_score,
            step_up_chase_ms,
            expected_slippage_bps,
            expected_slippage_r,
        }
    }

    /// Conservative historical/OHLCV routing without live L1 quote depth:
    /// bypasses synthetic queue-depth guessing (`q_bid`/`q_ask`) and applies explicit
    /// tier-based conservative half-spread + market-impact slippage (`5 / 12 / 35 bps`).
    pub fn route_ohlcv_conservative(
        symbol: &str,
        mid_price: f64,
        structural_risk_per_share: f64,
        atr_bar: f64,
        direction: TradeDirection,
        is_penny: bool,
    ) -> MicropriceExecutionPlan {
        let mid = mid_price.max(1e-4);
        let one_way_bps = explicit_one_way_slippage_bps(symbol, mid, is_penny);
        let slip_dollars = mid * (one_way_bps / 10_000.0);
        let limit_px = match direction {
            TradeDirection::Long => mid + slip_dollars,
            TradeDirection::Short => (mid - slip_dollars).max(0.0001),
        };
        let spread_to_atr_ratio = ((2.0 * slip_dollars) / atr_bar.max(1e-4)).clamp(0.0, 2.0);
        let expected_slippage_r = slip_dollars / structural_risk_per_share.abs().max(1e-4);

        MicropriceExecutionPlan {
            execution_tactic: ExecutionTactic::AdaptiveMicropricePeg,
            mid_price: mid,
            microprice: mid,
            microprice_limit_price: limit_px,
            microprice_alpha_skew: 0.0,
            spread_to_atr_ratio,
            execution_urgency_score: 0.50,
            step_up_chase_ms: 0,
            expected_slippage_bps: one_way_bps,
            expected_slippage_r,
        }
    }
}

pub const LIQUID_MEGA_CAP_SYMBOLS: &[&str] = &[
    "SPY", "QQQ", "IWM", "DIA", "AAPL", "MSFT", "NVDA", "TSLA", "META", "AMZN", "GOOGL", "AMD",
    "AVGO", "NFLX",
];

pub const PENNY_BRAIN01_SYMBOLS: &[&str] = &[
    "SOFI", "MARA", "RIOT", "CLSK", "IONQ", "RGTI", "BBAI", "SOUN", "UPST", "AFRM", "PLUG",
    "LCID", "RIVN", "OPEN", "CLOV", "KULR", "AMC", "GME",
];

/// Returns explicit one-way adverse half-spread + market-impact slippage in basis points:
/// - `5.0 bps` for liquid index / mega-caps
/// - `35.0 bps` for sub-$5 / low-float / `BRAIN-01` names
/// - `12.0 bps` for mid/high-beta equities
#[inline]
pub fn explicit_one_way_slippage_bps(symbol: &str, price: f64, is_penny_flag: bool) -> f64 {
    let clean = symbol.trim().to_uppercase();
    if is_penny_flag || price < 5.0 || PENNY_BRAIN01_SYMBOLS.contains(&clean.as_str()) {
        35.0
    } else if LIQUID_MEGA_CAP_SYMBOLS.contains(&clean.as_str()) {
        5.0
    } else {
        12.0
    }
}

/// Conservative round-trip option bid-ask spread per share per leg (`$0.03–$0.08`).
#[inline]
pub fn estimate_option_spread_per_leg(
    symbol: &str,
    spot_price: f64,
    is_vertical_combo: bool,
) -> f64 {
    let clean = symbol.trim().to_uppercase();
    let base: f64 = if matches!(clean.as_str(), "SPY" | "QQQ" | "IWM" | "DIA") {
        0.035
    } else if LIQUID_MEGA_CAP_SYMBOLS.contains(&clean.as_str()) {
        0.045
    } else if spot_price < 10.0 || PENNY_BRAIN01_SYMBOLS.contains(&clean.as_str()) {
        0.075
    } else {
        0.060
    };
    if is_vertical_combo {
        (base + 0.015).clamp(0.03, 0.08)
    } else {
        base.clamp(0.03, 0.08)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RoundTripCostEstimate {
    pub one_way_equity_slippage_bps: f64,
    pub expected_t1_payoff_per_unit: f64,
    pub round_trip_friction_per_unit: f64,
    pub t1_to_cost_ratio: f64,
    pub tradable: bool,
}

/// Returns true only when both contract label and structure specify an actual option contract
/// rather than cash shares (`SHARES_ONLY`, `EQUITY`, or empty).
#[inline]
pub fn is_option_routed_structure(matched_contract: &str, option_structure: &str) -> bool {
    let mc = matched_contract.trim();
    let st = option_structure.trim();
    !mc.is_empty()
        && !mc.eq_ignore_ascii_case("SHARES_ONLY")
        && !st.eq_ignore_ascii_case("SHARES_ONLY")
        && !st.eq_ignore_ascii_case("EQUITY")
}

/// Evaluates honest round-trip friction (spread + slippage + commissions/SEC-FINRA fees)
/// and checks Stage 0 Check 3 (**Tradability & Cost Gate**: expected T1 payoff >= 2.5x RT cost).
pub fn evaluate_tradability_and_cost_gate(
    symbol: &str,
    entry_price: f64,
    target_1_price: f64,
    is_option_routed: bool,
    is_vertical_combo: bool,
    option_entry_premium: f64,
    option_target_premium: f64,
    contract_delta: f64,
    is_penny_flag: bool,
) -> RoundTripCostEstimate {
    let entry = entry_price.max(1e-6);
    let one_way_bps = explicit_one_way_slippage_bps(symbol, entry, is_penny_flag);
    let underlying_t1_move = (target_1_price - entry).abs();

    if is_option_routed {
        let legs = if is_vertical_combo { 2.0 } else { 1.0 };
        let opt_spread_per_leg = estimate_option_spread_per_leg(symbol, entry, is_vertical_combo);
        let comm_per_share_equiv_rt = legs * 2.0 * 0.0065; // $0.65/contract/leg = $0.0065/share per leg side
        // Entry/exit adverse fill includes option bid-ask spread across legs + commissions + underlying impact
        let rt_friction = legs * opt_spread_per_leg
            + comm_per_share_equiv_rt
            + (contract_delta.abs().clamp(0.15, 1.0) * entry * (one_way_bps / 10_000.0) * 0.5);
        let exp_opt_payoff = if option_target_premium > option_entry_premium && option_entry_premium > 0.01 {
            (option_target_premium - option_entry_premium)
                .max(underlying_t1_move * contract_delta.abs().clamp(0.20, 0.90))
        } else {
            underlying_t1_move * contract_delta.abs().clamp(0.25, 0.90)
        };
        let ratio = exp_opt_payoff / rt_friction.max(1e-9);
        RoundTripCostEstimate {
            one_way_equity_slippage_bps: one_way_bps,
            expected_t1_payoff_per_unit: exp_opt_payoff,
            round_trip_friction_per_unit: rt_friction,
            t1_to_cost_ratio: ratio,
            tradable: ratio >= 2.5,
        }
    } else {
        // Equity shares: 2 * explicit slippage (entry + exit) + SEC/FINRA regulatory fee (~0.003%)
        let rt_friction = entry * ((2.0 * one_way_bps / 10_000.0) + 0.00003);
        let ratio = underlying_t1_move / rt_friction.max(1e-9);
        RoundTripCostEstimate {
            one_way_equity_slippage_bps: one_way_bps,
            expected_t1_payoff_per_unit: underlying_t1_move,
            round_trip_friction_per_unit: rt_friction,
            t1_to_cost_ratio: ratio,
            tradable: ratio >= 2.5,
        }
    }
}

/// Converts Unix epoch seconds into exact US/Eastern (`America/New_York`, EST/EDT DST-aware)
/// `(NaiveDate, Weekday, hour, minute, second)`.
pub fn unix_sec_to_ny_et(ts_sec: i64) -> (chrono::NaiveDate, chrono::Weekday, u32, u32, u32) {
    use chrono::{Datelike, Duration as ChronoDuration, NaiveDate, Timelike};
    let utc_dt = chrono::DateTime::from_timestamp(ts_sec, 0)
        .unwrap_or_default()
        .naive_utc();
    let year = utc_dt.year();
    // 2nd Sunday in March: start from March 8..=14
    let mar1 = NaiveDate::from_ymd_opt(year, 3, 1).unwrap_or_default();
    let first_sun_mar = 1 + ((7 - mar1.weekday().num_days_from_sunday()) % 7);
    let second_sun_mar = first_sun_mar + 7;
    let dst_start_utc = NaiveDate::from_ymd_opt(year, 3, second_sun_mar)
        .unwrap_or_default()
        .and_hms_opt(7, 0, 0) // 02:00 EST == 07:00 UTC
        .unwrap_or_default();

    // 1st Sunday in November: Nov 1..=7
    let nov1 = NaiveDate::from_ymd_opt(year, 11, 1).unwrap_or_default();
    let first_sun_nov = 1 + ((7 - nov1.weekday().num_days_from_sunday()) % 7);
    let dst_end_utc = NaiveDate::from_ymd_opt(year, 11, first_sun_nov)
        .unwrap_or_default()
        .and_hms_opt(6, 0, 0) // 02:00 EDT == 06:00 UTC
        .unwrap_or_default();

    let offset_hours = if utc_dt >= dst_start_utc && utc_dt < dst_end_utc {
        4_i64
    } else {
        5_i64
    };
    let et = utc_dt - ChronoDuration::hours(offset_hours);
    (et.date(), et.weekday(), et.hour(), et.minute(), et.second())
}

#[inline]
pub fn ny_et_hhmm(ts_sec: i64) -> (chrono::Weekday, u32) {
    let (_, wd, h, m, _) = unix_sec_to_ny_et(ts_sec);
    (wd, h * 100 + m)
}

#[inline]
pub fn is_us_regular_market_hours_et(ts_sec: i64) -> bool {
    let (wd, hhmm) = ny_et_hhmm(ts_sec);
    !matches!(wd, chrono::Weekday::Sat | chrono::Weekday::Sun) && (930..=1600).contains(&hhmm)
}

