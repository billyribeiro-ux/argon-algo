//! Zero-Cheat Chronological Walk-Forward Proof Engine (`proof_harness.rs`) driven 100% autonomously
//! by `PureAutonomousRlBrain` (`src/pure_rl_brain.rs`).
//!
//! Architectural Laws Enforced:
//! 1. **Zero Hardcoded Strategies:** Zero human indicators (`EMA`, `SMA`, `RSI`, `MACD`, `ORB`, `VCP`),
//!    zero chart-pattern templates, and zero hardcoded stop/horizon gears. Continuous exposure
//!    `a_t in [-1, +1]` (exact `0.0` FLAT inside learned cost-scaled deadzone), companion vehicle
//!    routing `v_t in [0, 1]`, dynamic tail-stop `s_t`, and intraday `NEURAL_POLICY_EXIT` are learned
//!    end-to-end via analytical BPTT + Straight-Through Surrogate Gradients + Multi-Horizon World Model.
//! 2. **Session Episode Slicing & Differentiable EOD Closure:** All intraday brains (`SCALP_0DTE 1m`,
//!    `SCALP_0DTE 5m`, `BRAIN-01`, `BRAIN-02`) slice history into individual `(Symbol, TradingDay)`
//!    session episodes (`[09:35 ET .. 15:50/15:55 ET]`), carry only causal cross-day regime context
//!    across session boundaries, and hard-enforce `a_{T_eod} = 0.0` with exact EOD flatten friction.
//!    `SCALP_0DTE (1m)` pre-trains causally on `5m` session episodes (`t_end <= T_1m_oos_start`).
//! 3. **Zero Lookahead Curtain:** Every feature tensor, standardization statistic, World-Model target,
//!    Fisher EWC diagonal, and Loss-Autopsy re-test sees ONLY causal resolved bars `<= t_current`.
//! 4. **Next-Bar Open Entry + Pessimistic Stop-First + Full Post-Cost Accounting:** Signals on bar `t`
//!    fill at `bar_{t+1}.open` with `5 / 12 / 35 bps` equity slippage, `$0.03-$0.08`/sh/leg option
//!    spread, `$0.65`/contract/leg commission, real `^VIX9D`/`^VIX` BSM IV + Theta, and mandatory
//!    `15:50 / 15:55 ET` intraday EOD flattens.
//! 5. **Mandatory 4-Step Closed-Loop Red-Trade Loss Audit -> RL Update -> Full Re-Test -> Promotion:**
//!    Every settled losing trade triggers Counterfactual + BPTT Saliency Audit -> Prioritized Red-Trade
//!    Replay (`50%` batch, `1.0 + 4.0 * |net_r|^2`) + Diagonal Fisher EWC Winner Lock -> Full Causal
//!    Discrete Re-Test -> Promotion ONLY IF `Net PnL`, `Net PF`, and `Red Loss` strictly improve
//!    without regressing winners.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;
use std::sync::Arc;

use chrono::{Datelike, Utc};
use market_data_bus::yahoo::{
    MACRO_BREADTH_ETFS, REAL_YAHOO_CORE_UNIVERSE, RealExchangeOhlcvBar, VIX_TERM_SYMBOLS,
    YahooFinanceLiveAdapter, build_causal_universe_snapshots_at_timestamp,
    compute_causal_internals_at_timestamp, est_trading_day_key, lookup_bar_le_timestamp,
    lookup_real_universe_meta,
};
use market_data_bus::{MarketDataAdapter, TickerSnapshot, Timeframe};
use raw_fingerprint_core::{
    BrainHorizonId, FINGERPRINT_FEATURE_DIM, FourStreamWeights, PrimaryFingerprintDriver,
    RawBarRecord, RawFingerprintEngine, RawOptionsSnapshot, RawShortFloatSnapshot, SubPennyPrice,
    TradeDirection,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::execution_microstructure::{
    PENNY_BRAIN01_SYMBOLS, estimate_option_spread_per_leg, explicit_one_way_slippage_bps,
    ny_et_hhmm, unix_sec_to_ny_et,
};
use crate::ledger::{
    BENCHMARK_LEDGER_FIXTURE_PATH, ClosedOutcomeBadge, ClosedTradeRow, EngineExecutionMode,
    LIVE_RUNTIME_STATE_PATH, RealTimeLedgerEngine, StrategyArchetype,
};
use crate::onnx_engine::OnnxMultiBrainEngine;
use crate::options_convexity::{OptionRight, compute_bsm_leg_greeks, round_cboe_strike};
use crate::pure_rl_brain::{
    BucketAutopsyTelemetry, CausalSessionEpisode, CausalTrajectoryBar, HIDDEN_DIM, LossFaultKind,
    POPULATION_SIZE, PhysicsWorldModel, PopulationMember, PureAutonomousRlBrain, RL_CROSS_SECT_DIM,
    RL_MULTI_SCALE_HTF_DIM, RL_STATE_DIM, SelfAwarenessState, SettledWinnerTrajectory, WM_DIM,
    WM_HORIZONS, WM_OUT_DIM, build_raw_212_state_vector, normalize_212_with_stats,
};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct WarmupTrainingTelemetry {
    pub brain_label: String,
    pub population_size: usize,
    pub train_sequences: usize,
    pub best_val_net_pf: f64,
    pub best_val_net_pnl: f64,
    pub best_val_trades: usize,
    pub pbt_generations: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationPromotionLedgerRow {
    pub gen_id: u32,
    pub phase_label: String,
    pub brain_label: String,
    pub red_trades_audited: usize,
    pub entry_faults: usize,
    pub exit_faults: usize,
    pub mean_saliency_dx: f32,
    pub mean_saliency_dh: f32,
    pub pf_before: f64,
    pub pf_after: f64,
    pub pnl_before_usd: f64,
    pub pnl_after_usd: f64,
    pub red_loss_reduced_usd: f64,
    pub winners_preserved_desc: String,
    pub promoted: bool,
}

pub const WALKFORWARD_BLOTTER_CSV_PATH: &str = "data/walkforward_proof_blotter.csv";
pub const PROOF_REPORT_MD_PATH: &str = "PROOF_REPORT.md";

const SCALP_0DTE_SYMBOLS: &[&str] = &[
    "SPY", "QQQ", "NVDA", "TSLA", "AAPL", "AMD", "META", "AMZN", "IWM",
];
const DATASET1_MOVERS: &[&str] = &[
    "PLTR", "COIN", "MSTR", "SMCI", "ARM", "HOOD", "SOFI", "MARA", "RIOT", "IONQ", "SOUN", "UPST",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProofRunMode {
    RunAFrozenWarmupMl,
    RunBOnlineRl,
}

impl ProofRunMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::RunAFrozenWarmupMl => "RUN_A_FROZEN_WARMUP_ML",
            Self::RunBOnlineRl => "RUN_B_ONLINE_RL",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ProofBucket {
    Scalp0Dte1m,
    Scalp0Dte5m,
    Brain01PennyDay,
    Brain02LargeCapDay,
    Brain03ShortSwing,
    Brain04CoreSwing,
}

impl ProofBucket {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Scalp0Dte1m => "SCALP_0DTE (1m)",
            Self::Scalp0Dte5m => "SCALP_0DTE (5m)",
            Self::Brain01PennyDay => "BRAIN-01 (Penny Day)",
            Self::Brain02LargeCapDay => "BRAIN-02 (Large-Cap Day)",
            Self::Brain03ShortSwing => "BRAIN-03 (Short Swing)",
            Self::Brain04CoreSwing => "BRAIN-04 (Core Swing)",
        }
    }

    pub const fn runtime_brain_key(self) -> &'static str {
        match self {
            Self::Scalp0Dte1m => "SCALP_0DTE_1M",
            Self::Scalp0Dte5m => "SCALP_0DTE",
            Self::Brain01PennyDay => "BRAIN-01",
            Self::Brain02LargeCapDay => "BRAIN-02",
            Self::Brain03ShortSwing => "BRAIN-03",
            Self::Brain04CoreSwing => "BRAIN-04",
        }
    }

    pub const fn is_intraday(self) -> bool {
        matches!(
            self,
            Self::Scalp0Dte1m
                | Self::Scalp0Dte5m
                | Self::Brain01PennyDay
                | Self::Brain02LargeCapDay
        )
    }

    pub const fn seed_base(self) -> u64 {
        match self {
            Self::Scalp0Dte1m => 0x1101_5CA1_0001_0D7E,
            Self::Scalp0Dte5m => 0x1505_5CA1_0005_0D7E,
            Self::Brain01PennyDay => 0x2001_B001_7E11_0001,
            Self::Brain02LargeCapDay => 0x3002_B002_DA90_0002,
            Self::Brain03ShortSwing => 0x4003_B003_5F16_0003,
            Self::Brain04CoreSwing => 0x5004_B004_C0BE_0004,
        }
    }

    pub const fn brain_id(self) -> BrainHorizonId {
        match self {
            Self::Scalp0Dte1m | Self::Scalp0Dte5m | Self::Brain02LargeCapDay => {
                BrainHorizonId::Brain02IntradayDayTrade
            }
            Self::Brain01PennyDay => BrainHorizonId::Brain01PennyAnomaly,
            Self::Brain03ShortSwing => BrainHorizonId::Brain03ShortTermSwing,
            Self::Brain04CoreSwing => BrainHorizonId::Brain04CoreSwing,
        }
    }

    pub const fn tier_name(self) -> &'static str {
        match self {
            Self::Scalp0Dte1m | Self::Scalp0Dte5m => "SCALP_0DTE",
            Self::Brain01PennyDay | Self::Brain02LargeCapDay => "DAY",
            Self::Brain03ShortSwing | Self::Brain04CoreSwing => "SWING",
        }
    }

    pub const fn timeframe_str(self) -> &'static str {
        match self {
            Self::Scalp0Dte1m => "1m",
            Self::Scalp0Dte5m | Self::Brain01PennyDay | Self::Brain02LargeCapDay => "5m",
            Self::Brain03ShortSwing | Self::Brain04CoreSwing => "1h",
        }
    }

    pub const fn bars_per_year(self) -> f64 {
        match self {
            Self::Scalp0Dte1m => 252.0 * 390.0,
            Self::Scalp0Dte5m | Self::Brain01PennyDay | Self::Brain02LargeCapDay => 252.0 * 78.0,
            Self::Brain03ShortSwing | Self::Brain04CoreSwing => 252.0 * 7.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlotterTradeRecord {
    pub run_mode: String,
    pub trade_id: String,
    pub symbol: String,
    pub brain_id: String,
    pub tier: String,
    pub timeframe: String,
    pub bucket: ProofBucket,
    pub signal_time_est: String,
    pub entry_time_est: String,
    pub exit_time_est: String,
    pub direction: String,
    pub structure: String,
    pub entry_underlying_px: f64,
    pub stop_px: f64,
    pub t1_px: f64,
    pub exit_underlying_px: f64,
    pub gross_r: f64,
    pub spread_and_comm_cost_usd: f64,
    pub net_pnl_usd: f64,
    pub net_r: f64,
    pub gross_pnl_usd: f64,
    pub exit_reason: String,
    pub active_gen_id: u32,
    pub scale_gear: String,
    pub vehicle: String,
    pub arm_label: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BucketPerformanceSummary {
    pub bucket_label: String,
    pub run_mode: String,
    pub trade_count: usize,
    pub wins_net: usize,
    pub win_rate_pct: f64,
    pub gross_profit_factor: f64,
    pub net_profit_factor: f64,
    pub gross_expectancy_r: f64,
    pub net_expectancy_r: f64,
    pub net_expectancy_usd: f64,
    pub total_gross_pnl_usd: f64,
    pub total_cost_drag_usd: f64,
    pub total_net_pnl_usd: f64,
    pub max_drawdown_usd: f64,
    pub max_drawdown_pct: f64,
    pub full_stop_rate_pct: f64,
    #[serde(default)]
    pub trail_step_rate_pct: f64,
    pub eod_flatten_rate_pct: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BucketAbComparison {
    pub bucket_label: String,
    pub run_a: BucketPerformanceSummary,
    pub run_b: BucketPerformanceSummary,
    pub delta_trade_count: isize,
    pub delta_win_rate_pct: f64,
    pub delta_net_pf: f64,
    pub delta_net_expectancy_r: f64,
    pub delta_net_pnl_usd: f64,
    pub delta_max_dd_usd: f64,
    pub traps_learned_or_blocked: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VehiclePerformanceSummary {
    pub vehicle_label: String,
    pub run_mode: String,
    pub trade_count: usize,
    pub wins_net: usize,
    pub win_rate_pct: f64,
    pub gross_profit_factor: f64,
    pub net_profit_factor: f64,
    pub net_expectancy_r: f64,
    pub avg_cost_drag_per_trade_usd: f64,
    pub total_gross_pnl_usd: f64,
    pub total_cost_drag_usd: f64,
    pub total_net_pnl_usd: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalkforwardProofReport {
    pub generated_at_utc: String,
    pub dataset1_trading_days_total: usize,
    pub dataset1_oos_days: usize,
    pub dataset1_m1_bars_loaded: usize,
    pub dataset2_universe_symbols_loaded: usize,
    pub dataset2_trading_days_total: usize,
    pub dataset2_warmup_days: usize,
    pub dataset2_oos_days: usize,
    pub blotter_csv_path: String,
    pub report_md_path: String,
    pub oos_trades_run_a: usize,
    pub oos_trades_run_b: usize,
    pub traps_blocked_in_run_b: usize,
    pub traps_quarantined_in_run_b: usize,
    #[serde(default)]
    pub warmup_counterfactuals_harvested: usize,
    #[serde(default)]
    pub oos_counterfactuals_harvested: usize,
    #[serde(default)]
    pub online_rl_grad_updates: u64,
    #[serde(default)]
    pub warmup_telemetry: Vec<WarmupTrainingTelemetry>,
    #[serde(default)]
    pub autopsy_telemetry: Vec<BucketAutopsyTelemetry>,
    #[serde(default)]
    pub promotion_ledger: Vec<GenerationPromotionLedgerRow>,
    #[serde(default)]
    pub dataset1_delta_net_pnl_usd: f64,
    #[serde(default)]
    pub dataset2_delta_net_pnl_usd: f64,
    #[serde(default)]
    pub vehicle_split_run_a: Vec<VehiclePerformanceSummary>,
    #[serde(default)]
    pub vehicle_split_run_b: Vec<VehiclePerformanceSummary>,
    pub bucket_comparisons: Vec<BucketAbComparison>,
    pub portfolio_run_a: BucketPerformanceSummary,
    pub portfolio_run_b: BucketPerformanceSummary,
}

/// Complete self-contained pre-fetched datasets used by the chronological simulator.
#[derive(Debug, Clone, Default)]
pub struct ProofDatasets {
    pub m1_bars: HashMap<String, Vec<RealExchangeOhlcvBar>>,
    pub m5_bars: HashMap<String, Vec<RealExchangeOhlcvBar>>,
    pub h1_bars: HashMap<String, Vec<RealExchangeOhlcvBar>>,
    pub d1_bars: HashMap<String, Vec<RealExchangeOhlcvBar>>,
    pub vix9d_bars: Vec<RealExchangeOhlcvBar>,
    pub vix_bars: Vec<RealExchangeOhlcvBar>,
    pub vix3m_bars: Vec<RealExchangeOhlcvBar>,
    pub breadth_m1: HashMap<String, Vec<RealExchangeOhlcvBar>>,
    pub breadth_m5: HashMap<String, Vec<RealExchangeOhlcvBar>>,
    pub breadth_h1: HashMap<String, Vec<RealExchangeOhlcvBar>>,
}

/// Precomputed strictly causal (`<= bar_idx`) raw tensor & cross-sectional snapshot per `(symbol, bar_idx)`.
#[derive(Debug, Clone)]
pub struct CausalBarPhysicsSnapshot {
    pub valid: bool,
    pub embedding_128: [f32; 128],
    pub instant_32: [f32; FINGERPRINT_FEATURE_DIM],
    pub cross_sect_16: [f32; RL_CROSS_SECT_DIM],
    pub htf_32: [f32; RL_MULTI_SCALE_HTF_DIM],
    pub atr_px: f64,
    pub share_cost_to_atr: f64,
    pub opt_cost_to_atr: f64,
    pub vix9d_to_vix_log_ratio: f64,
    pub vix_spot_dec: f64,
    pub vix9d_spot_dec: f64,
    pub causal_realized_vol: f64,
    pub day_key: i64,
    pub hhmm: u32,
    pub bar_ord: u32,
    pub stream_weights: FourStreamWeights,
    pub primary_driver: PrimaryFingerprintDriver,
}

impl Default for CausalBarPhysicsSnapshot {
    fn default() -> Self {
        Self {
            valid: false,
            embedding_128: [0.0; 128],
            instant_32: [0.0; FINGERPRINT_FEATURE_DIM],
            cross_sect_16: [0.5; RL_CROSS_SECT_DIM],
            htf_32: [0.0; RL_MULTI_SCALE_HTF_DIM],
            atr_px: 0.01,
            share_cost_to_atr: 0.15,
            opt_cost_to_atr: 0.65,
            vix9d_to_vix_log_ratio: 0.0,
            vix_spot_dec: 0.20,
            vix9d_spot_dec: 0.20,
            causal_realized_vol: 0.25,
            day_key: 0,
            hhmm: 1000,
            bar_ord: 0,
            stream_weights: FourStreamWeights {
                chart_tape_physics_weight: 0.25,
                options_gamma_vanna_weight: 0.25,
                short_float_squeeze_weight: 0.25,
                market_internals_macro_weight: 0.25,
            },
            primary_driver: PrimaryFingerprintDriver::VolumeAbsorption,
        }
    }
}

impl CausalBarPhysicsSnapshot {
    #[inline]
    pub fn to_raw_180_vector(
        &self,
        norm_session_remaining: f64,
        book_heat: f64,
        is_option: bool,
    ) -> [f32; RL_STATE_DIM] {
        let cost_atr = if is_option {
            self.opt_cost_to_atr
        } else {
            self.share_cost_to_atr
        };
        build_raw_212_state_vector(
            &self.embedding_128,
            &self.instant_32,
            cost_atr,
            norm_session_remaining,
            self.vix9d_to_vix_log_ratio,
            book_heat,
            &self.cross_sect_16,
            &self.htf_32,
        )
    }

    #[inline]
    pub fn to_raw_212_vector(
        &self,
        norm_session_remaining: f64,
        book_heat: f64,
        is_option: bool,
    ) -> [f32; RL_STATE_DIM] {
        self.to_raw_180_vector(norm_session_remaining, book_heat, is_option)
    }
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct PendingNextBarSignal {
    sym_id: usize,
    symbol: String,
    bucket: ProofBucket,
    brain_id: BrainHorizonId,
    signal_ts_sec: i64,
    signal_day_key: i64,
    signal_bar_ord: u32,
    direction: TradeDirection,
    target_action: f32,
    vehicle_prob: f32,
    tail_stop_dist: f64,
    confidence_pct: f64,
    is_option_routed: bool,
    is_vertical_combo: bool,
    is_penny_tier: bool,
    dte_days: f64,
    kelly_risk_budget_usd: f64,
    embedding_128: Vec<f32>,
    active_gen_id: u32,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct ActiveSimulatedPosition {
    trade_id: Uuid,
    sym_id: usize,
    symbol: String,
    bucket: ProofBucket,
    brain_id: BrainHorizonId,
    signal_ts_sec: i64,
    entry_ts_sec: i64,
    entry_day_key: i64,
    entry_bar_ord: u32,
    direction: TradeDirection,
    current_target_action: f32,
    structure_label: String,
    entry_mid_underlying: f64,
    entry_fill_underlying: f64,
    catastrophic_stop_px: f64,
    realized_partial_gross_usd: f64,
    realized_partial_cost_usd: f64,
    remaining_weight: f64,
    peak_favorable_px: f64,
    adverse_worst_px: f64,
    peak_v_liq_ret: f64,
    bars_held: usize,
    is_option_routed: bool,
    is_vertical_combo: bool,
    #[allow(dead_code)]
    is_penny_tier: bool,
    option_long_strike: f64,
    option_short_strike: Option<f64>,
    option_entry_dte_days: f64,
    option_entry_theo_mid: f64,
    option_contracts: u32,
    option_spread_per_leg: f64,
    equity_shares: f64,
    slip_bps_one_way: f64,
    risk_budget_usd: f64,
    active_gen_id: u32,
    confidence_pct: f64,
    primary_driver: PrimaryFingerprintDriver,
    stream_weights: FourStreamWeights,
    embedding_128: Vec<f32>,
    pending_exit: bool,
    pending_scale_action: Option<f32>,
}

/// Format Unix epoch seconds into compact `YYYY-MM-DD HH:MM:SS ET` using DST-aware NY time.
pub fn format_ny_et_timestamp(ts_sec: i64) -> String {
    let (d, _, h, m, s) = unix_sec_to_ny_et(ts_sec);
    format!(
        "{:04}-{:02}-{:02} {:02}:{:02}:{:02} ET",
        d.year(),
        d.month(),
        d.day(),
        h,
        m,
        s
    )
}

/// Fetches and caches all required Yahoo historical series (`1m` 7d, `5m` 60d, `1h` 730d, `1D`,
/// `^VIX9D`, `^VIX`, `^VIX3M`, and `MACRO_BREADTH_ETFS`). Relies cleanly on `data/ohlcv_cache/`
/// both when live and offline.
pub async fn fetch_all_proof_datasets(adapter: &YahooFinanceLiveAdapter) -> ProofDatasets {
    let _ = adapter.connect().await;

    let mut base_reqs: Vec<(String, Timeframe, usize)> = Vec::new();
    for sym in VIX_TERM_SYMBOLS.iter().chain(MACRO_BREADTH_ETFS.iter()) {
        base_reqs.push(((*sym).to_string(), Timeframe::M1, 8_500));
        base_reqs.push(((*sym).to_string(), Timeframe::M5, 5_000));
        base_reqs.push(((*sym).to_string(), Timeframe::H1, 1_200));
        base_reqs.push(((*sym).to_string(), Timeframe::D1, 300));
    }
    let _ = adapter.prefetch_ohlcv_series_parallel(&base_reqs).await;

    let mut m1_syms: BTreeSet<String> = BTreeSet::new();
    for s in SCALP_0DTE_SYMBOLS.iter().chain(DATASET1_MOVERS.iter()) {
        m1_syms.insert((*s).to_string());
    }
    let m1_reqs: Vec<(String, Timeframe, usize)> =
        m1_syms.iter().map(|s| (s.clone(), Timeframe::M1, 8_500)).collect();
    let _ = adapter.prefetch_ohlcv_series_parallel(&m1_reqs).await;

    let mut d2_reqs: Vec<(String, Timeframe, usize)> =
        Vec::with_capacity(REAL_YAHOO_CORE_UNIVERSE.len() * 3);
    for meta in REAL_YAHOO_CORE_UNIVERSE {
        d2_reqs.push((meta.symbol.to_string(), Timeframe::M5, 5_000));
        d2_reqs.push((meta.symbol.to_string(), Timeframe::H1, 1_000));
        d2_reqs.push((meta.symbol.to_string(), Timeframe::D1, 300));
    }
    let _ = adapter.prefetch_ohlcv_series_parallel(&d2_reqs).await;

    let mut ds = ProofDatasets::default();
    for sym in &m1_syms {
        if let Ok(bars) = adapter
            .session()
            .fetch_real_timeframe_ohlcv_bars(sym, Timeframe::M1, 8_500)
            .await
            && !bars.is_empty()
        {
            ds.m1_bars.insert(sym.clone(), bars);
        }
    }
    for meta in REAL_YAHOO_CORE_UNIVERSE {
        let sym = meta.symbol.to_string();
        if let Ok(bars) = adapter
            .session()
            .fetch_real_timeframe_ohlcv_bars(&sym, Timeframe::M5, 5_000)
            .await
            && !bars.is_empty()
        {
            ds.m5_bars.insert(sym.clone(), bars);
        }
        if let Ok(bars) = adapter
            .session()
            .fetch_real_timeframe_ohlcv_bars(&sym, Timeframe::H1, 1_000)
            .await
            && !bars.is_empty()
        {
            ds.h1_bars.insert(sym.clone(), bars);
        }
        if let Ok(bars) = adapter
            .session()
            .fetch_real_timeframe_ohlcv_bars(&sym, Timeframe::D1, 300)
            .await
            && !bars.is_empty()
        {
            ds.d1_bars.insert(sym.clone(), bars);
        }
    }

    async fn load_best_vix(
        adapter: &YahooFinanceLiveAdapter,
        sym: &str,
    ) -> Vec<RealExchangeOhlcvBar> {
        let mut combined: Vec<RealExchangeOhlcvBar> = Vec::new();
        for (tf, lim) in [
            (Timeframe::H1, 1_200),
            (Timeframe::M5, 5_000),
            (Timeframe::M1, 8_500),
        ] {
            if let Ok(bars) = adapter
                .session()
                .fetch_real_timeframe_ohlcv_bars(sym, tf, lim)
                .await
            {
                combined.extend(bars);
            }
        }
        if combined.is_empty()
            && let Ok(d) = adapter
                .session()
                .fetch_real_timeframe_ohlcv_bars(sym, Timeframe::D1, 300)
                .await
        {
            combined.extend(d);
        }
        combined.sort_by_key(|b| b.timestamp_sec);
        combined.dedup_by_key(|b| b.timestamp_sec);
        combined
    }

    ds.vix9d_bars = load_best_vix(adapter, "^VIX9D").await;
    ds.vix_bars = load_best_vix(adapter, "^VIX").await;
    ds.vix3m_bars = load_best_vix(adapter, "^VIX3M").await;

    for sym in MACRO_BREADTH_ETFS {
        if let Ok(b) = adapter
            .session()
            .fetch_real_timeframe_ohlcv_bars(sym, Timeframe::M1, 8_500)
            .await
            && !b.is_empty()
        {
            ds.breadth_m1.insert((*sym).to_string(), b);
        }
        if let Ok(b) = adapter
            .session()
            .fetch_real_timeframe_ohlcv_bars(sym, Timeframe::M5, 5_000)
            .await
            && !b.is_empty()
        {
            ds.breadth_m5.insert((*sym).to_string(), b);
        }
        if let Ok(b) = adapter
            .session()
            .fetch_real_timeframe_ohlcv_bars(sym, Timeframe::H1, 1_200)
            .await
            && !b.is_empty()
        {
            ds.breadth_h1.insert((*sym).to_string(), b);
        }
    }

    ds
}

fn collect_sorted_timestamps_and_days(
    series: &HashMap<String, Vec<RealExchangeOhlcvBar>>,
    regular_hours_only: bool,
) -> (Vec<i64>, Vec<i64>) {
    let mut ts_set: BTreeSet<i64> = BTreeSet::new();
    let mut day_set: BTreeSet<i64> = BTreeSet::new();
    for bars in series.values() {
        for b in bars {
            let (wd, hhmm) = ny_et_hhmm(b.timestamp_sec);
            if matches!(wd, chrono::Weekday::Sat | chrono::Weekday::Sun) {
                continue;
            }
            if regular_hours_only && !(930..=1600).contains(&hhmm) {
                continue;
            }
            ts_set.insert(b.timestamp_sec);
            day_set.insert(est_trading_day_key(b.timestamp_sec));
        }
    }
    (ts_set.into_iter().collect(), day_set.into_iter().collect())
}

fn causal_realized_vol_ann(bars: &[RawBarRecord], lookback: usize, bars_per_year: f64) -> f64 {
    let n = bars.len().min(lookback);
    if n < 6 {
        return 0.25;
    }
    let slice = &bars[bars.len() - n..];
    let mut rets = Vec::with_capacity(n - 1);
    for w in slice.windows(2) {
        let c0 = w[0].close.to_f64().max(1e-6);
        let c1 = w[1].close.to_f64().max(1e-6);
        rets.push((c1 / c0).ln());
    }
    let m = rets.iter().sum::<f64>() / (rets.len() as f64);
    let var = rets.iter().map(|r| (r - m) * (r - m)).sum::<f64>()
        / (rets.len().saturating_sub(1).max(1) as f64);
    (var.sqrt() * bars_per_year.sqrt()).clamp(0.10, 2.50)
}

#[inline]
fn causal_true_range_atr(bars: &[RealExchangeOhlcvBar], end_idx: usize, window: usize) -> f64 {
    let start = end_idx.saturating_sub(window.saturating_sub(1));
    let mut tr_sum = 0.0_f64;
    let mut count = 0usize;
    for i in start..=end_idx {
        let b = &bars[i];
        let prev_c = if i > 0 { bars[i - 1].close } else { b.open };
        let tr = (b.high - b.low)
            .max((b.high - prev_c).abs())
            .max((b.low - prev_c).abs());
        tr_sum += tr;
        count += 1;
    }
    (tr_sum / (count.max(1) as f64))
        .max(bars[end_idx].close * 0.0008)
        .max(1e-4)
}

#[inline]
fn is_penny_symbol_cand(symbol: &str, close_px: f64) -> bool {
    let meta = lookup_real_universe_meta(symbol);
    close_px < 12.0 || meta.is_otc_penny || PENNY_BRAIN01_SYMBOLS.contains(&symbol)
}

#[inline]
fn symbol_matches_bucket(symbol: &str, close_px: f64, bucket: ProofBucket) -> bool {
    let is_penny = is_penny_symbol_cand(symbol, close_px);
    match bucket {
        ProofBucket::Scalp0Dte1m => {
            !is_penny
                && close_px >= 15.0
                && (SCALP_0DTE_SYMBOLS.contains(&symbol) || DATASET1_MOVERS.contains(&symbol))
        }
        ProofBucket::Scalp0Dte5m => SCALP_0DTE_SYMBOLS.contains(&symbol),
        ProofBucket::Brain01PennyDay => is_penny,
        ProofBucket::Brain02LargeCapDay => close_px >= 15.0 && !is_penny,
        ProofBucket::Brain03ShortSwing | ProofBucket::Brain04CoreSwing => {
            close_px >= 12.0 && !is_penny
        }
    }
}

#[inline]
fn entry_allowed_at_hhmm(bucket: ProofBucket, hhmm: u32) -> bool {
    match bucket {
        ProofBucket::Scalp0Dte1m | ProofBucket::Scalp0Dte5m => (935..=1515).contains(&hhmm),
        ProofBucket::Brain01PennyDay | ProofBucket::Brain02LargeCapDay => {
            (935..=1530).contains(&hhmm)
        }
        ProofBucket::Brain03ShortSwing | ProofBucket::Brain04CoreSwing => {
            (935..=1555).contains(&hhmm)
        }
    }
}

#[inline]
fn session_eod_hhmm(bucket: ProofBucket) -> u32 {
    match bucket {
        ProofBucket::Scalp0Dte1m | ProofBucket::Scalp0Dte5m => 1550,
        ProofBucket::Brain01PennyDay | ProofBucket::Brain02LargeCapDay => 1555,
        ProofBucket::Brain03ShortSwing | ProofBucket::Brain04CoreSwing => 1600,
    }
}

#[inline]
fn normalized_session_remaining(bucket: ProofBucket, hhmm: u32) -> f64 {
    let cur_m = ((hhmm / 100) * 60 + (hhmm % 100)) as f64;
    let end_m = match bucket {
        ProofBucket::Scalp0Dte1m | ProofBucket::Scalp0Dte5m => 15.0 * 60.0 + 50.0,
        ProofBucket::Brain01PennyDay | ProofBucket::Brain02LargeCapDay => 15.0 * 60.0 + 55.0,
        ProofBucket::Brain03ShortSwing | ProofBucket::Brain04CoreSwing => 16.0 * 60.0,
    };
    ((end_m - cur_m).max(0.0) / 385.0).clamp(0.0, 1.0)
}

#[inline]
fn is_option_vehicle_allowed(symbol: &str, close_px: f64) -> bool {
    let meta = lookup_real_universe_meta(symbol);
    meta.is_optionable && !meta.is_otc_penny && close_px >= 5.0
}

#[inline]
fn resolve_vehicle_routing_config(
    symbol: &str,
    close_px: f64,
    bucket: ProofBucket,
    hhmm: u32,
    vix_spot_dec: f64,
    use_option: bool,
) -> (bool, bool, f64) {
    let cur_m = (hhmm / 100) * 60 + (hhmm % 100);
    let mins_to_close = (16 * 60_u32).saturating_sub(cur_m).max(15) as f64;
    let dte_days = match bucket {
        ProofBucket::Scalp0Dte1m | ProofBucket::Scalp0Dte5m => {
            (mins_to_close / 390.0).max(1.0 / 390.0)
        }
        ProofBucket::Brain01PennyDay | ProofBucket::Brain02LargeCapDay => 5.0,
        ProofBucket::Brain03ShortSwing => 21.0,
        ProofBucket::Brain04CoreSwing => 35.0,
    };
    if !use_option || !is_option_vehicle_allowed(symbol, close_px) {
        return (false, false, dte_days);
    }
    let is_vertical = close_px > 240.0
        || (matches!(
            bucket,
            ProofBucket::Brain03ShortSwing | ProofBucket::Brain04CoreSwing
        ) && vix_spot_dec > 0.23);
    (true, is_vertical, dte_days)
}

#[inline]
fn estimate_roundtrip_cost_to_atr(
    symbol: &str,
    spot_px: f64,
    atr_px: f64,
    is_option_routed: bool,
    is_vertical_combo: bool,
    is_penny_tier: bool,
) -> f64 {
    let slip_bps = explicit_one_way_slippage_bps(symbol, spot_px, is_penny_tier);
    let underlying_rt_friction = if is_option_routed {
        let legs = if is_vertical_combo { 2.0 } else { 1.0 };
        let opt_spread = estimate_option_spread_per_leg(symbol, spot_px, is_vertical_combo);
        let comm_per_sh = legs * 2.0 * 0.0065;
        let eff_delta = if is_vertical_combo { 0.16 } else { 0.50 };
        ((legs * opt_spread + comm_per_sh) / eff_delta) + spot_px * (slip_bps / 10_000.0)
    } else {
        spot_px * ((2.0 * slip_bps / 10_000.0) + 0.00003)
    };
    (underlying_rt_friction / atr_px.max(1e-4)).clamp(0.0, 5.0)
}

#[inline]
fn compute_learned_backstop_px(
    anchor_px: f64,
    atr_px: f64,
    learned_tail_dist: f64,
    direction: TradeDirection,
) -> f64 {
    let dist = (anchor_px * learned_tail_dist.clamp(0.015, 0.16))
        .max(2.4 * atr_px)
        .max(anchor_px * 0.015)
        .max(1e-4);
    (anchor_px - direction.sign() * dist).max(0.0001)
}

fn build_causal_synthetic_short_float(
    symbol: &str,
    bars_le_t: &[RealExchangeOhlcvBar],
    snapshot: Option<&TickerSnapshot>,
) -> RawShortFloatSnapshot {
    let latest_ts_ns = bars_le_t
        .last()
        .map(|b| b.timestamp_sec * 1_000_000_000)
        .unwrap_or(0);
    let meta = lookup_real_universe_meta(symbol);
    let cum_vol = snapshot
        .map(|s| s.day_volume)
        .unwrap_or_else(|| bars_le_t.last().map(|b| b.volume).unwrap_or(100_000));
    let default_float = if meta.is_otc_penny {
        50_000_000
    } else {
        150_000_000
    };
    let default_short = if meta.is_otc_penny {
        5_000_000
    } else {
        8_000_000
    };
    let float_sh = snapshot
        .map(|s| s.public_float_shares)
        .unwrap_or(default_float)
        .max(1_000_000);
    let short_sh = snapshot
        .map(|s| s.short_interest_shares)
        .unwrap_or(default_short)
        .max(10_000);
    RawShortFloatSnapshot {
        timestamp_ns: latest_ts_ns,
        public_float_shares: float_sh,
        shares_outstanding: (float_sh as f64 * 1.18) as u64,
        short_interest_shares: short_sh,
        days_to_cover: ((short_sh as f64) / (cum_vol.max(50_000) as f64)).clamp(0.5, 15.0),
        cost_to_borrow_annual_pct: if meta.is_otc_penny { 18.0 } else { 1.2 },
        intraday_short_volume_ratio: 0.45,
        cumulative_session_volume: cum_vol.max(1),
    }
}

fn compute_bsm_structure_price(
    right: OptionRight,
    underlying_spot: f64,
    strike_long: f64,
    strike_short: Option<f64>,
    dte_days: f64,
    atm_iv: f64,
) -> f64 {
    let skew_beta = match right {
        OptionRight::Call => -0.35,
        OptionRight::Put => 0.45,
    };
    let long_leg = compute_bsm_leg_greeks(
        right,
        underlying_spot,
        strike_long,
        dte_days.max(1.0 / 1560.0),
        1.0 / 78.0,
        0.045,
        0.0,
        atm_iv,
        skew_beta,
        0.12,
    );
    if let Some(k2) = strike_short {
        let short_leg = compute_bsm_leg_greeks(
            right,
            underlying_spot,
            k2,
            dte_days.max(1.0 / 1560.0),
            1.0 / 78.0,
            0.045,
            0.0,
            atm_iv,
            skew_beta,
            0.12,
        );
        (long_leg.theoretical_price - short_leg.theoretical_price).max(0.01)
    } else {
        long_leg.theoretical_price.max(0.01)
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn extract_causal_32_htf_features(
    _sym: &str,
    t_sec: i64,
    d_key: i64,
    bars: &[RealExchangeOhlcvBar],
    bar_idx: usize,
    h1_bars: Option<&[RealExchangeOhlcvBar]>,
    daily_bars: Option<&[RealExchangeOhlcvBar]>,
    spy_h1: Option<&[RealExchangeOhlcvBar]>,
    qqq_h1: Option<&[RealExchangeOhlcvBar]>,
    vix9d_to_vix_log: f64,
    vix_spot_dec: f64,
) -> [f32; RL_MULTI_SCALE_HTF_DIM] {
    let mut htf = [0.0_f32; RL_MULTI_SCALE_HTF_DIM];

    let mut closed_h1_bars: Vec<&RealExchangeOhlcvBar> = Vec::new();
    if let Some(h1s) = h1_bars {
        let end_idx = h1s.partition_point(|h| h.timestamp_sec + 3600 <= t_sec);
        if end_idx > 0 {
            for h in &h1s[..end_idx] {
                assert!(
                    h.timestamp_sec + 3600 <= t_sec,
                    "Strict anti-cheat: HTF candle ended > current bar"
                );
                closed_h1_bars.push(h);
            }
        }
    }

    if closed_h1_bars.len() >= 2 {
        let n = closed_h1_bars.len();
        let last = closed_h1_bars[n - 1];
        let prev = closed_h1_bars[n - 2];
        let prev3 = if n >= 4 { closed_h1_bars[n - 4] } else { prev };
        let prev7 = if n >= 8 { closed_h1_bars[n - 8] } else { prev3 };

        let c0 = last.close.max(1e-4);
        let o0 = last.open.max(1e-4);
        let h0 = last.high.max(c0).max(o0);
        let l0 = last.low.min(c0).min(o0).max(1e-4);
        let rng0 = (h0 - l0).max(1e-4);

        let ret1 = (c0 / o0).ln();
        let ret2 = (c0 / prev.close.max(1e-4)).ln();
        let ret4 = (c0 / prev3.close.max(1e-4)).ln();
        let ret8 = (c0 / prev7.close.max(1e-4)).ln();

        let start_4 = n.saturating_sub(4);
        let avg_rng4 = (closed_h1_bars[start_4..n]
            .iter()
            .map(|b| b.high - b.low)
            .sum::<f64>()
            / 4.0)
            .max(1e-4);
        let range_to_c = rng0 / c0;
        let range_ratio = rng0 / avg_rng4;
        let upper_shadow = (h0 - c0.max(o0)) / rng0;
        let lower_shadow = (c0.min(o0) - l0) / rng0;
        let body_ratio = (c0 - o0).abs() / rng0;

        let start_8 = n.saturating_sub(8);
        let avg_vol8 = (closed_h1_bars[start_8..n]
            .iter()
            .map(|b| b.volume as f64)
            .sum::<f64>()
            / 8.0)
            .max(1.0);
        let vol_ratio = (last.volume as f64) / avg_vol8;
        let vol_flow = vol_ratio * ret1;

        let mut up_vol = 1.0_f64;
        let mut dn_vol = 1.0_f64;
        for b in &closed_h1_bars[start_4..n] {
            if b.close >= b.open {
                up_vol += b.volume as f64;
            } else {
                dn_vol += b.volume as f64;
            }
        }
        let up_down_ratio = up_vol / dn_vol;

        let spy_ret = if let Some(spys) = spy_h1 {
            let spy_end = spys.partition_point(|s| s.timestamp_sec + 3600 <= t_sec);
            if spy_end >= 2 {
                (spys[spy_end - 1].close / spys[spy_end - 2].close.max(1e-4)).ln()
            } else {
                0.0
            }
        } else {
            0.0
        };

        let qqq_ret = if let Some(qqqs) = qqq_h1 {
            let qqq_end = qqqs.partition_point(|s| s.timestamp_sec + 3600 <= t_sec);
            if qqq_end >= 2 {
                (qqqs[qqq_end - 1].close / qqqs[qqq_end - 2].close.max(1e-4)).ln()
            } else {
                0.0
            }
        } else {
            0.0
        };

        let spy_rng_ratio = if let Some(spys) = spy_h1 {
            let spy_end = spys.partition_point(|s| s.timestamp_sec + 3600 <= t_sec);
            if spy_end > 0 {
                let srng = spys[spy_end - 1].high - spys[spy_end - 1].low;
                let s_start = spy_end.saturating_sub(4);
                let savg = (spys[s_start..spy_end]
                    .iter()
                    .map(|b| b.high - b.low)
                    .sum::<f64>()
                    / 4.0)
                    .max(1e-4);
                srng / savg
            } else {
                1.0
            }
        } else {
            1.0
        };

        htf[0] = (ret1 as f32).clamp(-3.0, 3.0);
        htf[1] = (ret2 as f32).clamp(-3.0, 3.0);
        htf[2] = (ret4 as f32).clamp(-3.0, 3.0);
        htf[3] = (ret8 as f32).clamp(-3.0, 3.0);
        htf[4] = ((range_to_c * 50.0) as f32).clamp(-3.0, 3.0);
        htf[5] = ((range_ratio - 1.0) as f32).clamp(-3.0, 3.0);
        htf[6] = (upper_shadow as f32).clamp(0.0, 1.0);
        htf[7] = (lower_shadow as f32).clamp(0.0, 1.0);
        htf[8] = (body_ratio as f32).clamp(0.0, 1.0);
        htf[9] = ((vol_ratio - 1.0) as f32).clamp(-3.0, 3.0);
        htf[10] = ((vol_flow * 10.0) as f32).clamp(-3.0, 3.0);
        htf[11] = (up_down_ratio.ln() as f32).clamp(-3.0, 3.0);
        htf[12] = ((spy_ret * 20.0) as f32).clamp(-3.0, 3.0);
        htf[13] = ((qqq_ret * 20.0) as f32).clamp(-3.0, 3.0);
        htf[14] = ((spy_rng_ratio - 1.0) as f32).clamp(-3.0, 3.0);
        htf[15] = (vix9d_to_vix_log as f32).clamp(-2.0, 2.0);
    } else if bar_idx >= 3 {
        let c0 = bars[bar_idx].close.max(1e-4);
        let o0 = bars[bar_idx.saturating_sub(2)].open.max(1e-4);
        let ret1 = (c0 / o0).ln();
        let prev_c = bars[bar_idx.saturating_sub(3)].close.max(1e-4);
        let ret2 = (c0 / prev_c).ln();
        htf[0] = (ret1 as f32).clamp(-3.0, 3.0);
        htf[1] = (ret2 as f32).clamp(-3.0, 3.0);
        htf[15] = (vix9d_to_vix_log as f32).clamp(-2.0, 2.0);
    }

    let mut closed_daily_bars: Vec<&RealExchangeOhlcvBar> = Vec::new();
    if let Some(dbars) = daily_bars {
        let d_end_idx = dbars.partition_point(|d| est_trading_day_key(d.timestamp_sec) < d_key);
        if d_end_idx > 0 {
            for d in &dbars[..d_end_idx] {
                assert!(
                    est_trading_day_key(d.timestamp_sec) < d_key,
                    "Strict anti-cheat: Macro daily bar day >= current day"
                );
                closed_daily_bars.push(d);
            }
        }
    }

    if closed_daily_bars.len() >= 2 {
        let n = closed_daily_bars.len();
        let last = closed_daily_bars[n - 1];
        let prev = closed_daily_bars[n - 2];
        let prev5 = if n >= 6 { closed_daily_bars[n - 6] } else { prev };
        let prev10 = if n >= 11 { closed_daily_bars[n - 11] } else { prev5 };

        let mc0 = last.close.max(1e-4);
        let mo0 = last.open.max(1e-4);
        let mh0 = last.high.max(mc0).max(mo0);
        let ml0 = last.low.min(mc0).min(mo0).max(1e-4);
        let mrng0 = (mh0 - ml0).max(1e-4);

        let mret1 = (mc0 / mo0).ln();
        let mret2 = (mc0 / prev.close.max(1e-4)).ln();
        let mret5 = (mc0 / prev5.close.max(1e-4)).ln();
        let mret10 = (mc0 / prev10.close.max(1e-4)).ln();

        let start_5 = n.saturating_sub(5);
        let avg_mrng5 = (closed_daily_bars[start_5..n]
            .iter()
            .map(|b| b.high - b.low)
            .sum::<f64>()
            / 5.0)
            .max(1e-4);
        let m_range_to_c = mrng0 / mc0;
        let m_range_ratio = mrng0 / avg_mrng5;
        let m_upper_shadow = (mh0 - mc0.max(mo0)) / mrng0;
        let m_lower_shadow = (mc0.min(mo0) - ml0) / mrng0;
        let m_body_ratio = (mc0 - mo0).abs() / mrng0;

        let start_10 = n.saturating_sub(10);
        let avg_mvol10 = (closed_daily_bars[start_10..n]
            .iter()
            .map(|b| b.volume as f64)
            .sum::<f64>()
            / 10.0)
            .max(1.0);
        let m_vol_ratio = (last.volume as f64) / avg_mvol10;
        let m_vol_flow = m_vol_ratio * mret1;

        let mut m_up_vol = 1.0_f64;
        let mut m_dn_vol = 1.0_f64;
        for b in &closed_daily_bars[start_5..n] {
            if b.close >= b.open {
                m_up_vol += b.volume as f64;
            } else {
                m_dn_vol += b.volume as f64;
            }
        }
        let m_up_down_ratio = m_up_vol / m_dn_vol;
        let m_vix_ratio = (vix_spot_dec / 0.20).ln();

        htf[16] = (mret1 as f32).clamp(-3.0, 3.0);
        htf[17] = (mret2 as f32).clamp(-3.0, 3.0);
        htf[18] = (mret5 as f32).clamp(-3.0, 3.0);
        htf[19] = (mret10 as f32).clamp(-3.0, 3.0);
        htf[20] = ((m_range_to_c * 30.0) as f32).clamp(-3.0, 3.0);
        htf[21] = ((m_range_ratio - 1.0) as f32).clamp(-3.0, 3.0);
        htf[22] = (m_upper_shadow as f32).clamp(0.0, 1.0);
        htf[23] = (m_lower_shadow as f32).clamp(0.0, 1.0);
        htf[24] = (m_body_ratio as f32).clamp(0.0, 1.0);
        htf[25] = ((m_vol_ratio - 1.0) as f32).clamp(-3.0, 3.0);
        htf[26] = ((m_vol_flow * 5.0) as f32).clamp(-3.0, 3.0);
        htf[27] = (m_up_down_ratio.ln() as f32).clamp(-3.0, 3.0);
        htf[28] = (mret2 as f32).clamp(-3.0, 3.0);
        htf[29] = (mret5 as f32).clamp(-3.0, 3.0);
        htf[30] = ((mret2 * 10.0) as f32).clamp(-3.0, 3.0);
        htf[31] = (m_vix_ratio as f32).clamp(-3.0, 3.0);
    }

    htf
}

/// Precomputes strictly causal (`<= bar_idx`) raw 4-stream tensors across all symbols on a timeline.
#[allow(clippy::too_many_arguments)]
fn precompute_causal_timeline_physics(
    symbol_bars: &HashMap<String, Vec<RealExchangeOhlcvBar>>,
    h1_bars: &HashMap<String, Vec<RealExchangeOhlcvBar>>,
    daily_bars: &HashMap<String, Vec<RealExchangeOhlcvBar>>,
    vix9d_bars: &[RealExchangeOhlcvBar],
    vix_bars: &[RealExchangeOhlcvBar],
    vix3m_bars: &[RealExchangeOhlcvBar],
    breadth_bars: &HashMap<String, Vec<RealExchangeOhlcvBar>>,
    rep_brain_id: BrainHorizonId,
    bars_per_year: f64,
) -> HashMap<String, Vec<CausalBarPhysicsSnapshot>> {
    let mut all_ts_set: BTreeSet<i64> = BTreeSet::new();
    for bars in symbol_bars.values() {
        for b in bars {
            all_ts_set.insert(b.timestamp_sec);
        }
    }
    let mut internals_by_ts = HashMap::with_capacity(all_ts_set.len());
    for ts in all_ts_set {
        let int = compute_causal_internals_at_timestamp(
            ts,
            breadth_bars,
            vix9d_bars,
            vix_bars,
            vix3m_bars,
        );
        internals_by_ts.insert(ts, int);
    }

    let mut sorted_syms: Vec<&str> = symbol_bars.keys().map(String::as_str).collect();
    sorted_syms.sort_unstable();

    let worker_count = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8)
        .max(1);
    let chunk_size = sorted_syms.len().div_ceil(worker_count).max(1);
    let internals_ref = &internals_by_ts;
    let spy_h1 = h1_bars.get("SPY").map(Vec::as_slice);
    let qqq_h1 = h1_bars.get("QQQ").map(Vec::as_slice);

    let computed_pairs: Vec<(String, Vec<CausalBarPhysicsSnapshot>)> = std::thread::scope(|s| {
        let mut handles = Vec::new();
        for sym_chunk in sorted_syms.chunks(chunk_size) {
            handles.push(s.spawn(move || {
                let mut local_pairs = Vec::with_capacity(sym_chunk.len());
                for &sym in sym_chunk {
                    let bars = &symbol_bars[sym];
                    let sym_h1 = h1_bars.get(sym).map(Vec::as_slice);
                    let sym_daily = daily_bars.get(sym).map(Vec::as_slice);
                    let raw_records: Vec<RawBarRecord> =
                        bars.iter().map(|b| b.to_raw_bar_record()).collect();
                    let meta = lookup_real_universe_meta(sym);
                    let fp_engine = RawFingerprintEngine::new();
                    let mut snaps = vec![CausalBarPhysicsSnapshot::default(); bars.len()];

                    let mut cur_sess_day = i64::MIN;
                    let mut running_sess_vol: u64 = 0;

                    for i in 0..bars.len() {
                        let b = &bars[i];
                        let d_key = est_trading_day_key(b.timestamp_sec);
                        if d_key != cur_sess_day {
                            cur_sess_day = d_key;
                            running_sess_vol = 0;
                        }
                        running_sess_vol = running_sess_vol.saturating_add(b.volume.max(1));

                        if i < 23 {
                            continue;
                        }
                        let start_idx = i.saturating_sub(47);
                        let causal_raw = &raw_records[start_idx..=i];
                        let causal_oh = &bars[start_idx..=i];

                        let Some(cur_int) = internals_ref.get(&b.timestamp_sec) else {
                            continue;
                        };
                        let vix_spot = cur_int.vix_spot.max(10.0);
                        let vix9d = cur_int.vix9d_spot.unwrap_or(vix_spot).max(9.0);
                        let vix3m = cur_int.vix3m_spot.unwrap_or(vix_spot).max(10.0);
                        let vix9d_to_vix_log = (vix9d / vix_spot).ln().clamp(-1.5, 1.5);

                        let real_vol = causal_realized_vol_ann(causal_raw, 36, bars_per_year);
                        let front_iv = ((vix9d / 100.0)
                            * (real_vol / (vix9d / 100.0).max(0.12)).clamp(0.75, 2.40))
                        .clamp(0.10, 2.20);
                        let back_iv = ((vix3m / 100.0)
                            * (real_vol / (vix3m / 100.0).max(0.12)).clamp(0.75, 2.40))
                        .clamp(0.10, 2.20);

                        let prior_adv = daily_bars
                            .get(sym)
                            .map(|dbars| {
                                let end_d = dbars.partition_point(|d| {
                                    est_trading_day_key(d.timestamp_sec) < cur_sess_day
                                });
                                let start_d = end_d.saturating_sub(20);
                                let sl = &dbars[start_d..end_d];
                                if sl.is_empty() {
                                    running_sess_vol.max(200_000)
                                } else {
                                    (sl.iter().map(|x| x.volume).sum::<u64>() / (sl.len() as u64))
                                        .max(50_000)
                                }
                            })
                            .unwrap_or_else(|| running_sess_vol.max(200_000));

                        let mut raw_sf = build_causal_synthetic_short_float(sym, causal_oh, None);
                        raw_sf.cumulative_session_volume = running_sess_vol;
                        raw_sf.days_to_cover = ((raw_sf.short_interest_shares as f64)
                            / (prior_adv as f64))
                        .clamp(0.5, 15.0);

                        let raw_opt = RawOptionsSnapshot {
                            timestamp_ns: b.timestamp_sec * 1_000_000_000,
                            call_volume: (running_sess_vol / 20).max(10),
                            put_volume: (running_sess_vol / 22).max(10),
                            total_open_interest: (prior_adv / 4).max(100),
                            ask_fill_aggression_ratio: 0.50,
                            bid_fill_aggression_ratio: 0.50,
                            net_dealer_gamma_gex: 0.0,
                            net_dealer_vanna: 0.0,
                            zero_dte_volume_share: if SCALP_0DTE_SYMBOLS.contains(&sym) {
                                0.55
                            } else {
                                0.15
                            },
                            front_iv,
                            back_iv,
                            dominant_gamma_wall_strike: SubPennyPrice::from_f64(
                                round_cboe_strike(b.close, b.close).max(0.50),
                            )
                            .unwrap_or(SubPennyPrice::MIN_VALID),
                        };
                        let raw_int = cur_int.to_raw_internals_snapshot(meta.sector_etf);

                        let Ok(tensor) = fp_engine.build_sequence_tensor_for_brain(
                            Some(rep_brain_id),
                            causal_raw,
                            &raw_opt,
                            &raw_sf,
                            &raw_int,
                        ) else {
                            continue;
                        };

                        let mut emb = [0.0_f32; 128];
                        for (k, &v) in tensor.latent_embedding_128.iter().take(128).enumerate() {
                            emb[k] = v;
                        }
                        let mut inst = [0.0_f32; FINGERPRINT_FEATURE_DIM];
                        let last_off = (tensor.seq_len - 1) * FINGERPRINT_FEATURE_DIM;
                        inst.copy_from_slice(
                            &tensor.flat_tensor[last_off..last_off + FINGERPRINT_FEATURE_DIM],
                        );

                        let atr_px = causal_true_range_atr(bars, i, 20);
                        let is_penny = is_penny_symbol_cand(sym, b.close);
                        let share_cost_to_atr = estimate_roundtrip_cost_to_atr(
                            sym, b.close, atr_px, false, false, is_penny,
                        );
                        let opt_cost_to_atr = estimate_roundtrip_cost_to_atr(
                            sym, b.close, atr_px, true, false, is_penny,
                        );

                        let htf_32 = extract_causal_32_htf_features(
                            sym,
                            b.timestamp_sec,
                            d_key,
                            bars,
                            i,
                            sym_h1,
                            sym_daily,
                            spy_h1,
                            qqq_h1,
                            vix9d_to_vix_log,
                            vix_spot / 100.0,
                        );

                        snaps[i] = CausalBarPhysicsSnapshot {
                            valid: true,
                            embedding_128: emb,
                            instant_32: inst,
                            cross_sect_16: [0.5; RL_CROSS_SECT_DIM],
                            htf_32,
                            atr_px,
                            share_cost_to_atr,
                            opt_cost_to_atr,
                            vix9d_to_vix_log_ratio: vix9d_to_vix_log,
                            vix_spot_dec: (vix_spot / 100.0).clamp(0.10, 1.50),
                            vix9d_spot_dec: (vix9d / 100.0).clamp(0.10, 1.50),
                            causal_realized_vol: real_vol,
                            day_key: d_key,
                            hhmm: ny_et_hhmm(b.timestamp_sec).1,
                            bar_ord: i as u32,
                            stream_weights: tensor.stream_weights,
                            primary_driver: tensor.stream_weights.primary_driver(),
                        };
                    }
                    local_pairs.push((sym.to_string(), snaps));
                }
                local_pairs
            }));
        }
        let mut out = Vec::with_capacity(sorted_syms.len());
        for h in handles {
            if let Ok(chunk_pairs) = h.join() {
                out.extend(chunk_pairs);
            }
        }
        out
    });

    let mut out_map: HashMap<String, Vec<CausalBarPhysicsSnapshot>> =
        computed_pairs.into_iter().collect();

    let mut sym_bars_by_ts: HashMap<i64, Vec<(String, usize)>> = HashMap::new();
    for (sym, snaps) in &out_map {
        let bars = &symbol_bars[sym];
        for (i, snap) in snaps.iter().enumerate() {
            if snap.valid && i < bars.len() {
                sym_bars_by_ts
                    .entry(bars[i].timestamp_sec)
                    .or_default()
                    .push((sym.clone(), i));
            }
        }
    }

    for (_ts, mut sym_indices) in sym_bars_by_ts {
        sym_indices.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        let n_syms = sym_indices.len();
        if n_syms == 0 {
            continue;
        }
        if n_syms == 1 {
            let (sym, idx) = &sym_indices[0];
            if let Some(snaps) = out_map.get_mut(sym.as_str()) {
                snaps[*idx].cross_sect_16 = [0.5; RL_CROSS_SECT_DIM];
            }
            continue;
        }

        let mut raw_feats: Vec<[f64; RL_CROSS_SECT_DIM]> = Vec::with_capacity(n_syms);
        for (sym, idx) in &sym_indices {
            let idx = *idx;
            let bars = &symbol_bars[sym.as_str()];
            let snap = &out_map[sym.as_str()][idx];
            let close = bars[idx].close.max(1e-4);
            let ret_1 = if idx >= 1 {
                (close - bars[idx - 1].close) / bars[idx - 1].close.max(1e-4)
            } else {
                0.0
            };
            let ret_3 = if idx >= 3 {
                (close - bars[idx - 3].close) / bars[idx - 3].close.max(1e-4)
            } else {
                0.0
            };
            let ret_12 = if idx >= 12 {
                (close - bars[idx - 12].close) / bars[idx - 12].close.max(1e-4)
            } else {
                0.0
            };
            let prior_vol_sum: f64 = bars[idx.saturating_sub(20)..idx]
                .iter()
                .map(|b| b.volume as f64)
                .sum();
            let vol_surge = (bars[idx].volume as f64) / (prior_vol_sum / 20.0).max(100.0);
            let atr_ratio = snap.atr_px / close;
            let share_cost = snap.share_cost_to_atr;
            let opt_cost = snap.opt_cost_to_atr;
            let hl_span = (bars[idx].high - bars[idx].low) / snap.atr_px.max(1e-4);
            let close_loc = (close - bars[idx].low) / (bars[idx].high - bars[idx].low).max(1e-4);
            let real_vol = snap.causal_realized_vol;
            let inst_mom = snap.instant_32[0] as f64;
            let vix_sens = snap.vix9d_to_vix_log_ratio;
            let str_micro = snap.stream_weights.chart_tape_physics_weight as f64;
            let str_opt = snap.stream_weights.options_gamma_vanna_weight as f64;
            let str_sf = snap.stream_weights.short_float_squeeze_weight as f64;
            let dollar_vol = close * (bars[idx].volume as f64);

            raw_feats.push([
                ret_1, ret_3, ret_12, vol_surge, atr_ratio, share_cost, opt_cost, hl_span,
                close_loc, real_vol, inst_mom, vix_sens, str_micro, str_opt, str_sf, dollar_vol,
            ]);
        }

        let n_f = (n_syms - 1) as f64;
        let mut rank_buf: Vec<(f64, usize)> = Vec::with_capacity(n_syms);
        let mut pct_results = vec![[0.5_f32; RL_CROSS_SECT_DIM]; n_syms];

        for feat_idx in 0..RL_CROSS_SECT_DIM {
            rank_buf.clear();
            for (i, row) in raw_feats.iter().enumerate() {
                rank_buf.push((row[feat_idx], i));
            }
            rank_buf.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            for (rank, &(_, sym_i)) in rank_buf.iter().enumerate() {
                pct_results[sym_i][feat_idx] = (rank as f64 / n_f) as f32;
            }
        }

        for (i, (sym, idx)) in sym_indices.iter().enumerate() {
            if let Some(snaps) = out_map.get_mut(sym.as_str()) {
                snaps[*idx].cross_sect_16 = pct_results[i];
            }
        }
    }

    out_map
}

#[allow(clippy::too_many_arguments)]
fn open_simulated_position_at_next_bar(
    sym_id: usize,
    symbol: &str,
    bucket: ProofBucket,
    brain_id: BrainHorizonId,
    signal_ts_sec: i64,
    signal_day_key: i64,
    direction: TradeDirection,
    target_action: f32,
    learned_tail_stop_dist: f64,
    is_option_routed: bool,
    is_vertical_combo: bool,
    is_penny_tier: bool,
    dte_days: f64,
    desired_risk_budget_usd: f64,
    account_equity_usd: f64,
    entry_bar: &RealExchangeOhlcvBar,
    entry_snap: &CausalBarPhysicsSnapshot,
    embedding_128: Vec<f32>,
    confidence_pct: f64,
    active_gen_id: u32,
    trade_id: Uuid,
) -> Option<ActiveSimulatedPosition> {
    let cur_day = est_trading_day_key(entry_bar.timestamp_sec);
    let (_, hhmm) = ny_et_hhmm(entry_bar.timestamp_sec);
    let is_intraday = bucket.is_intraday();
    if is_intraday && (cur_day != signal_day_key || hhmm >= 1535) {
        return None;
    }
    if matches!(bucket, ProofBucket::Scalp0Dte1m | ProofBucket::Scalp0Dte5m) && hhmm >= 1520 {
        return None;
    }

    let slip_bps = explicit_one_way_slippage_bps(symbol, entry_bar.open, is_penny_tier);
    let entry_fill_px = match direction {
        TradeDirection::Long => entry_bar.open * (1.0 + slip_bps / 10_000.0),
        TradeDirection::Short => entry_bar.open * (1.0 - slip_bps / 10_000.0),
    };
    let catastrophic_stop_px = compute_learned_backstop_px(
        entry_fill_px,
        entry_snap.atr_px,
        learned_tail_stop_dist,
        direction,
    );
    let unit_risk_u = (entry_fill_px - catastrophic_stop_px)
        .abs()
        .max(entry_bar.open * 0.008)
        .max(1e-5);

    let scaled_budget = (desired_risk_budget_usd
        * (0.55 + 0.45 * (target_action.abs() as f64).clamp(0.0, 1.0)))
    .max(250.0);
    let right = OptionRight::from_direction(direction);
    let base_vix = if is_intraday {
        entry_snap.vix9d_spot_dec
    } else {
        entry_snap.vix_spot_dec
    };
    let eff_iv = (base_vix * (entry_snap.causal_realized_vol / base_vix.max(0.12)).clamp(0.75, 2.40))
        .clamp(0.12, 1.85);

    struct VehicleEntrySizing {
        k1: f64,
        k2: Option<f64>,
        theo_entry_mid: f64,
        spread_per_leg: f64,
        contracts: u32,
        shares: f64,
        actual_risk_usd: f64,
        effective_vertical: bool,
        structure_label: String,
    }

    let mut use_option = is_option_routed;
    let mut opt_sizing: Option<VehicleEntrySizing> = None;
    if use_option {
        let strike = round_cboe_strike(entry_bar.open, entry_bar.open);
        let step = if entry_bar.open < 15.0 {
            0.50
        } else if entry_bar.open < 100.0 {
            1.0
        } else if entry_bar.open < 300.0 {
            2.5
        } else {
            5.0
        };
        let mut use_vert = is_vertical_combo;
        let mut wing = if use_vert {
            Some((strike + direction.sign() * step * 2.0).max(0.50))
        } else {
            None
        };
        let mut mid_px =
            compute_bsm_structure_price(right, entry_bar.open, strike, wing, dte_days, eff_iv);
        let mut spread_leg = estimate_option_spread_per_leg(symbol, entry_bar.open, use_vert);
        let mut theo_at_stop = compute_bsm_structure_price(
            right,
            catastrophic_stop_px,
            strike,
            wing,
            (dte_days * 0.92).max(1.0 / 390.0),
            eff_iv,
        );
        let mut legs = if use_vert { 2.0 } else { 1.0 };
        let mut risk_per_contract =
            ((mid_px - theo_at_stop).abs().max(0.20 * mid_px) + legs * spread_leg + legs * 0.013)
                * 100.0;

        if !use_vert && risk_per_contract > scaled_budget * 1.8 {
            use_vert = true;
            wing = Some((strike + direction.sign() * step * 2.0).max(0.50));
            mid_px =
                compute_bsm_structure_price(right, entry_bar.open, strike, wing, dte_days, eff_iv);
            spread_leg = estimate_option_spread_per_leg(symbol, entry_bar.open, true);
            theo_at_stop = compute_bsm_structure_price(
                right,
                catastrophic_stop_px,
                strike,
                wing,
                (dte_days * 0.92).max(1.0 / 390.0),
                eff_iv,
            );
            legs = 2.0;
            risk_per_contract = ((mid_px - theo_at_stop).abs().max(0.20 * mid_px)
                + legs * spread_leg
                + legs * 0.013)
                * 100.0;
        }
        if risk_per_contract > scaled_budget * 1.45 && !is_penny_tier {
            use_option = false;
        } else if risk_per_contract > account_equity_usd * 0.035 {
            return None;
        } else {
            let qty_contracts = (scaled_budget / risk_per_contract.max(25.0))
                .round()
                .clamp(1.0, 25.0) as u32;
            let pos_risk_usd = ((qty_contracts as f64) * risk_per_contract).max(50.0);
            let label = if let Some(kw) = wing {
                format!("BSM_VERTICAL_${:.1}/${:.1}", strike, kw)
            } else {
                format!(
                    "BSM_{}_${:.1}{}",
                    if matches!(bucket, ProofBucket::Scalp0Dte1m | ProofBucket::Scalp0Dte5m) {
                        "0DTE"
                    } else {
                        "OPT"
                    },
                    strike,
                    if right == OptionRight::Call { "C" } else { "P" }
                )
            };
            opt_sizing = Some(VehicleEntrySizing {
                k1: strike,
                k2: wing,
                theo_entry_mid: mid_px,
                spread_per_leg: spread_leg,
                contracts: qty_contracts,
                shares: (qty_contracts as f64) * 100.0,
                actual_risk_usd: pos_risk_usd,
                effective_vertical: use_vert,
                structure_label: label,
            });
        }
    }

    let VehicleEntrySizing {
        k1,
        k2,
        theo_entry_mid,
        spread_per_leg,
        contracts,
        shares,
        actual_risk_usd,
        effective_vertical,
        structure_label,
    } = if let Some(opt_sz) = opt_sizing {
        opt_sz
    } else {
        let rt_friction_per_sh = entry_bar.open * ((2.0 * slip_bps / 10_000.0) + 0.00003);
        let total_unit_risk = (unit_risk_u + rt_friction_per_sh).max(1e-4);
        let max_bp_frac = if is_penny_tier { 0.12 } else { 0.20 };
        let max_bp_shares = (max_bp_frac * account_equity_usd) / entry_fill_px.max(0.01);
        let qty_shares = (scaled_budget / total_unit_risk)
            .min(max_bp_shares)
            .floor()
            .max(1.0);
        let pos_risk_usd = (qty_shares * total_unit_risk).max(25.0);
        VehicleEntrySizing {
            k1: 0.0,
            k2: None,
            theo_entry_mid: 0.0,
            spread_per_leg: 0.0,
            contracts: 0,
            shares: qty_shares,
            actual_risk_usd: pos_risk_usd,
            effective_vertical: false,
            structure_label: "SHARES_ONLY".to_string(),
        }
    };

    Some(ActiveSimulatedPosition {
        trade_id,
        sym_id,
        symbol: symbol.to_string(),
        bucket,
        brain_id,
        signal_ts_sec,
        entry_ts_sec: entry_bar.timestamp_sec,
        entry_day_key: cur_day,
        entry_bar_ord: entry_snap.bar_ord,
        direction,
        current_target_action: target_action,
        structure_label,
        entry_mid_underlying: entry_bar.open,
        entry_fill_underlying: entry_fill_px,
        catastrophic_stop_px,
        realized_partial_gross_usd: 0.0,
        realized_partial_cost_usd: 0.0,
        remaining_weight: 1.0,
        peak_favorable_px: entry_fill_px,
        adverse_worst_px: entry_fill_px,
        peak_v_liq_ret: -(2.0 * slip_bps / 10_000.0).max(0.0004),
        bars_held: 1,
        is_option_routed: use_option,
        is_vertical_combo: effective_vertical,
        is_penny_tier,
        option_long_strike: k1,
        option_short_strike: k2,
        option_entry_dte_days: dte_days,
        option_entry_theo_mid: theo_entry_mid,
        option_contracts: contracts,
        option_spread_per_leg: spread_per_leg,
        equity_shares: shares,
        slip_bps_one_way: slip_bps,
        risk_budget_usd: actual_risk_usd,
        active_gen_id,
        confidence_pct,
        primary_driver: entry_snap.primary_driver,
        stream_weights: entry_snap.stream_weights,
        embedding_128,
        pending_exit: false,
        pending_scale_action: None,
    })
}

/// Evaluates catastrophic stop backstop, neural-policy exits/scales at `bar_t.open`, and
/// mandatory EOD forced flattens. Guarantees pessimistic STOP-FIRST execution on ambiguous bars.
fn step_active_position_on_bar(
    pos: &mut ActiveSimulatedPosition,
    bar_t: &RealExchangeOhlcvBar,
    current_vix9d_or_vix: f64,
    is_newly_opened_this_bar: bool,
) -> Option<(f64, String, bool)> {
    if !is_newly_opened_this_bar {
        pos.bars_held = pos.bars_held.saturating_add(1);
    }

    match pos.direction {
        TradeDirection::Long => {
            pos.peak_favorable_px = pos.peak_favorable_px.max(bar_t.high);
            pos.adverse_worst_px = pos.adverse_worst_px.min(bar_t.low);
        }
        TradeDirection::Short => {
            pos.peak_favorable_px = pos.peak_favorable_px.min(bar_t.low);
            pos.adverse_worst_px = pos.adverse_worst_px.max(bar_t.high);
        }
    }

    let (_, hhmm) = ny_et_hhmm(bar_t.timestamp_sec);
    let is_scalp = matches!(pos.bucket, ProofBucket::Scalp0Dte1m | ProofBucket::Scalp0Dte5m);
    let is_day = matches!(
        pos.bucket,
        ProofBucket::Brain01PennyDay | ProofBucket::Brain02LargeCapDay
    );

    // 1. PESSIMISTIC CATASTROPHIC BACKSTOP CHECK
    let (gap_through_stop, hit_stop_intrabar) = match pos.direction {
        TradeDirection::Long => (
            bar_t.open <= pos.catastrophic_stop_px,
            bar_t.low <= pos.catastrophic_stop_px,
        ),
        TradeDirection::Short => (
            bar_t.open >= pos.catastrophic_stop_px,
            bar_t.high >= pos.catastrophic_stop_px,
        ),
    };
    if gap_through_stop {
        return Some((bar_t.open, "CATASTROPHIC_GAP_BACKSTOP".to_string(), true));
    }

    // 2. NEURAL POLICY EXIT AT `bar_{t+1}.open`
    if !is_newly_opened_this_bar && pos.pending_exit {
        return Some((bar_t.open, "NEURAL_POLICY_EXIT".to_string(), false));
    }

    if hit_stop_intrabar {
        return Some((
            pos.catastrophic_stop_px,
            "CATASTROPHIC_INTRABAR_BACKSTOP".to_string(),
            true,
        ));
    }

    // 3. NEURAL POLICY PARTIAL SCALE-OUT AT `bar_{t+1}.open`
    if !is_newly_opened_this_bar
        && let Some(new_act) = pos.pending_scale_action.take()
    {
        let prev_mag = pos.current_target_action.abs().max(0.05);
        let new_mag = new_act.abs();
        if new_mag < prev_mag * 0.75 && pos.remaining_weight > 0.20 {
            let trim_frac = ((prev_mag - new_mag) / prev_mag) as f64;
            let trim_w = (pos.remaining_weight * trim_frac).clamp(0.10, pos.remaining_weight - 0.10);
            let (part_gross, part_cost) = compute_leg_pnl_components(
                pos,
                bar_t.timestamp_sec,
                bar_t.open,
                trim_w,
                current_vix9d_or_vix,
            );
            pos.realized_partial_gross_usd += part_gross;
            pos.realized_partial_cost_usd += part_cost;
            pos.remaining_weight -= trim_w;
            pos.current_target_action = new_act;
        }
    }

    // 4. STRICT INTRADAY SESSION FORCED FLATTENS
    let bar_day_key = est_trading_day_key(bar_t.timestamp_sec);
    if (is_scalp || is_day) && bar_day_key > pos.entry_day_key {
        return Some((
            bar_t.open,
            "EOD_MISSED_BAR_NEXT_OPEN_FLATTEN".to_string(),
            false,
        ));
    }
    if is_scalp && hhmm >= 1550 {
        return Some((bar_t.close, "EOD_FORCED_FLATTEN_1550_ET".to_string(), false));
    }
    if is_day && hhmm >= 1555 {
        return Some((bar_t.close, "EOD_FORCED_FLATTEN_1555_ET".to_string(), false));
    }

    None
}

fn compute_leg_pnl_components(
    pos: &ActiveSimulatedPosition,
    exit_ts_sec: i64,
    exit_underlying_px: f64,
    weight: f64,
    current_vix_iv: f64,
) -> (f64, f64) {
    if pos.is_option_routed && pos.option_contracts > 0 {
        let elapsed_trading_days = if pos.option_entry_dte_days <= 1.0 {
            ((exit_ts_sec - pos.entry_ts_sec).max(0) as f64) / (390.0 * 60.0)
        } else {
            (((exit_ts_sec - pos.entry_ts_sec).max(0) as f64) / 86_400.0) * (252.0 / 365.25)
        };
        let rem_dte = (pos.option_entry_dte_days - elapsed_trading_days).max(1.0 / (390.0 * 4.0));
        let right = OptionRight::from_direction(pos.direction);
        let theo_exit = compute_bsm_structure_price(
            right,
            exit_underlying_px,
            pos.option_long_strike,
            pos.option_short_strike,
            rem_dte,
            current_vix_iv,
        );
        let contracts_w = (pos.option_contracts as f64) * weight;
        let gross_usd = (theo_exit - pos.option_entry_theo_mid) * 100.0 * contracts_w;
        let legs = if pos.is_vertical_combo { 2.0 } else { 1.0 };
        let half_spread_exit = (0.5 * pos.option_spread_per_leg * legs) * 100.0 * contracts_w;
        let comm_exit = 0.65 * legs * contracts_w;
        (gross_usd, half_spread_exit + comm_exit)
    } else {
        let shares_w = pos.equity_shares * weight;
        let gross_usd =
            pos.direction.sign() * (exit_underlying_px - pos.entry_mid_underlying) * shares_w;
        let exit_notional = exit_underlying_px * shares_w;
        let exit_slip_usd = exit_notional * (pos.slip_bps_one_way / 10_000.0);
        let reg_fee = if pos.direction == TradeDirection::Long {
            (exit_notional * 0.0000278 + shares_w * 0.000166).clamp(0.01, 8.30)
        } else {
            (exit_notional * 0.00003).max(0.01)
        };
        (gross_usd, exit_slip_usd + reg_fee)
    }
}

/// Builds a unified underlying-equivalent Mark-to-Liquidation `SelfAwarenessState`
/// (`V_liq` units matching `PureAutonomousRlBrain` BPTT training) and updates `peak_v_liq_ret`.
#[inline]
fn build_active_position_self_awareness(
    pos: &mut ActiveSimulatedPosition,
    ts_sec: i64,
    mark_px: f64,
    current_vix_iv: f64,
    norm_rem: f64,
    ema_turnover: f32,
) -> (SelfAwarenessState, f64) {
    let (rem_gross, rem_exit_cost) =
        compute_leg_pnl_components(pos, ts_sec, mark_px, pos.remaining_weight, current_vix_iv);
    let (entry_cost_usd, eff_notional, eff_one_way_bps) =
        if pos.is_option_routed && pos.option_contracts > 0 {
            let legs = if pos.is_vertical_combo { 2.0 } else { 1.0 };
            let half_spread_entry =
                (0.5 * pos.option_spread_per_leg * legs) * 100.0 * (pos.option_contracts as f64);
            let comm_entry = 0.65 * legs * (pos.option_contracts as f64);
            let eff_delta = if pos.is_vertical_combo { 0.34 } else { 0.50 };
            let eq_base =
                (pos.entry_mid_underlying * 100.0 * (pos.option_contracts as f64) * eff_delta)
                    .max(100.0);
            let one_way_opt_bps = (((half_spread_entry + comm_entry) / eq_base) * 10_000.0)
                .clamp(4.0, 250.0);
            (half_spread_entry + comm_entry, eq_base, one_way_opt_bps)
        } else {
            let entry_notional = (pos.entry_mid_underlying * pos.equity_shares).max(100.0);
            let ec = entry_notional * (pos.slip_bps_one_way / 10_000.0);
            (ec, entry_notional, pos.slip_bps_one_way.max(1.0))
        };
    let gross_pnl = pos.realized_partial_gross_usd + rem_gross;
    let total_cost = entry_cost_usd + pos.realized_partial_cost_usd + rem_exit_cost;
    let v_liq_ret = ((gross_pnl - total_cost) / eff_notional).clamp(-0.50, 0.50);
    if pos.bars_held <= 2 && pos.peak_v_liq_ret <= -(2.0 * pos.slip_bps_one_way / 10_000.0) + 1e-9 {
        let init_liq = -(2.0 * eff_one_way_bps / 10_000.0);
        pos.peak_v_liq_ret = init_liq.max(v_liq_ret);
    } else {
        pos.peak_v_liq_ret = pos.peak_v_liq_ret.max(v_liq_ret);
    }
    let peak_mfe = (pos.peak_v_liq_ret as f32).max(0.0);
    let unreal = v_liq_ret as f32;
    let dd_peak = (pos.peak_v_liq_ret as f32 - unreal).max(0.0);
    let state = SelfAwarenessState {
        current_pos: pos.current_target_action,
        unrealized_net_ret: unreal,
        peak_excursion: peak_mfe,
        drawdown_from_peak: dd_peak,
        log_bars_held: (1.0 + pos.bars_held as f32).ln(),
        one_way_cost_bps: eff_one_way_bps as f32,
        session_time_left: norm_rem as f32,
        ema_turnover,
    };
    (state, v_liq_ret)
}

fn finalize_settled_position(
    pos: &ActiveSimulatedPosition,
    exit_ts_sec: i64,
    exit_underlying_px: f64,
    exit_reason: String,
    is_full_stop: bool,
    current_vix_iv: f64,
    run_mode: ProofRunMode,
) -> (BlotterTradeRecord, ClosedTradeRow) {
    let (rem_gross, rem_exit_cost) = compute_leg_pnl_components(
        pos,
        exit_ts_sec,
        exit_underlying_px,
        pos.remaining_weight,
        current_vix_iv,
    );
    let entry_cost_usd = if pos.is_option_routed && pos.option_contracts > 0 {
        let legs = if pos.is_vertical_combo { 2.0 } else { 1.0 };
        let half_spread_entry =
            (0.5 * pos.option_spread_per_leg * legs) * 100.0 * (pos.option_contracts as f64);
        let comm_entry = 0.65 * legs * (pos.option_contracts as f64);
        half_spread_entry + comm_entry
    } else {
        let entry_notional = pos.entry_mid_underlying * pos.equity_shares;
        entry_notional * (pos.slip_bps_one_way / 10_000.0)
    };

    let gross_pnl_usd = pos.realized_partial_gross_usd + rem_gross;
    let total_cost_usd = entry_cost_usd + pos.realized_partial_cost_usd + rem_exit_cost;
    let net_pnl_usd = gross_pnl_usd - total_cost_usd;
    let r_denom = pos.risk_budget_usd.max(25.0);
    let gross_r = gross_pnl_usd / r_denom;
    let net_r = net_pnl_usd / r_denom;

    let vehicle = if pos.is_option_routed {
        "V_OPTION"
    } else {
        "V_SHARES"
    }
    .to_string();
    let arm_label = format!(
        "{}_A{:.2}_{}",
        pos.direction.as_str(),
        pos.current_target_action.abs(),
        if pos.is_option_routed { "OPT" } else { "SH" }
    );

    let blotter = BlotterTradeRecord {
        run_mode: run_mode.as_str().to_string(),
        trade_id: pos.trade_id.to_string(),
        symbol: pos.symbol.clone(),
        brain_id: pos.brain_id.code().to_string(),
        tier: pos.bucket.tier_name().to_string(),
        timeframe: pos.bucket.timeframe_str().to_string(),
        bucket: pos.bucket,
        signal_time_est: format_ny_et_timestamp(pos.signal_ts_sec),
        entry_time_est: format_ny_et_timestamp(pos.entry_ts_sec),
        exit_time_est: format_ny_et_timestamp(exit_ts_sec),
        direction: pos.direction.as_str().to_string(),
        structure: pos.structure_label.clone(),
        entry_underlying_px: pos.entry_fill_underlying,
        stop_px: pos.catastrophic_stop_px,
        t1_px: pos.entry_fill_underlying,
        exit_underlying_px,
        gross_r,
        spread_and_comm_cost_usd: total_cost_usd,
        net_pnl_usd,
        net_r,
        gross_pnl_usd,
        exit_reason,
        active_gen_id: pos.active_gen_id,
        scale_gear: "CONTINUOUS_BPTT".to_string(),
        vehicle,
        arm_label,
    };

    let closed_ts_ns = exit_ts_sec * 1_000_000_000;
    let opened_ts_ns = pos.entry_ts_sec * 1_000_000_000;
    let to_sp = |v: f64| SubPennyPrice::from_f64(v.max(0.0001)).unwrap_or(SubPennyPrice::MIN_VALID);
    let badge = if is_full_stop {
        ClosedOutcomeBadge::StoppedOut
    } else if net_r >= 0.8 {
        ClosedOutcomeBadge::TargetHit
    } else {
        ClosedOutcomeBadge::AiDynamicExit
    };
    let mae_pct = ((pos.adverse_worst_px - pos.entry_fill_underlying).abs()
        / pos.entry_fill_underlying.max(1e-4))
        * 100.0;
    let mfe_pct = ((pos.peak_favorable_px - pos.entry_fill_underlying).abs()
        / pos.entry_fill_underlying.max(1e-4))
        * 100.0;
    let meta = lookup_real_universe_meta(&pos.symbol);

    let closed_row = ClosedTradeRow {
        trade_id: pos.trade_id,
        position_id: pos.trade_id,
        outcome_badge: badge,
        brain_id: pos.brain_id,
        brain_name: pos.brain_id.display_name().to_string(),
        symbol: pos.symbol.clone(),
        exchange_index: meta.exchange_index.to_string(),
        direction: pos.direction,
        opened_at_iso: format_ny_et_timestamp(pos.entry_ts_sec),
        closed_at_iso: format_ny_et_timestamp(exit_ts_sec),
        opened_timestamp_ns: opened_ts_ns,
        closed_timestamp_ns: closed_ts_ns,
        hold_duration_ms: (exit_ts_sec - pos.entry_ts_sec).max(0) * 1_000,
        entry_price: to_sp(pos.entry_fill_underlying),
        initial_structural_stop: to_sp(pos.catastrophic_stop_px),
        target_1_price: to_sp(pos.entry_fill_underlying),
        target_2_price: to_sp(pos.entry_fill_underlying),
        actual_exit_price: to_sp(exit_underlying_px),
        quantity_shares: pos.equity_shares,
        kelly_fraction_pct: (pos.risk_budget_usd / 100_000.0) * 100.0,
        risk_budget_usd: pos.risk_budget_usd,
        notional_allocated_usd: pos.equity_shares * pos.entry_fill_underlying,
        tranche_count: 1,
        scale_in_history: Vec::new(),
        option_structure: pos.structure_label.clone(),
        matched_option_contract: pos.structure_label.clone(),
        option_entry_premium: to_sp(pos.option_entry_theo_mid),
        option_stop_premium: SubPennyPrice::ZERO,
        option_exit_premium: SubPennyPrice::ZERO,
        option_contracts_count: pos.option_contracts,
        contract_delta: if pos.direction == TradeDirection::Long {
            0.50
        } else {
            -0.50
        },
        contract_gamma: 0.04,
        contract_vanna: 0.01,
        contract_theta: -0.05,
        contract_vega: 0.10,
        contract_iv: 0.30,
        gross_pnl_usd,
        slippage_commissions_usd: total_cost_usd,
        net_pnl_usd,
        kelly_net_pnl_usd: net_pnl_usd,
        option_net_pnl_usd: if pos.is_option_routed {
            net_pnl_usd
        } else {
            0.0
        },
        stock_return_pct: pos.direction.sign()
            * ((exit_underlying_px - pos.entry_fill_underlying)
                / pos.entry_fill_underlying.max(1e-4))
            * 100.0,
        option_return_pct: 0.0,
        realized_r_multiple: net_r,
        mae_pct,
        mfe_pct,
        is_winner: net_r >= 0.0,
        cumulative_net_profit_usd: 0.0,
        ai_confidence_pct: pos.confidence_pct,
        primary_driver: pos.primary_driver,
        precedent_matches_count: 1,
        precedent_win_rate_pct: 50.0,
        stream_weights: pos.stream_weights,
        price_sparkline: vec![pos.entry_fill_underlying, exit_underlying_px],
        entry_bars: Vec::new(),
        entry_bar_idx: 0,
        exit_bar_idx: 0,
        engine_mode: EngineExecutionMode::PureMlSelfLearned,
        strategy_archetype: StrategyArchetype::PureMlLatentAnomaly,
        loss_forensic_autopsy: None,
        entry_embedding_128: pos.embedding_128.clone(),
        was_vetoed_by_loss_autopsy: false,
        vetoed_by_loss_trade_id: None,
        entry_trap_similarity_score: 0.0,
        was_scratched_by_loss_autopsy: false,
        scratch_bar_offset: 0,
        adaptive_outcome_badge: String::new(),
        adaptive_exit_price: SubPennyPrice::ZERO,
        adaptive_scratch_exit_r: 0.0,
        adaptive_net_pnl_usd: net_pnl_usd,
        adaptive_kelly_net_pnl_usd: net_pnl_usd,
        adaptive_loss_saved_usd: 0.0,
        adaptive_kelly_loss_saved_usd: 0.0,
        loss_affinity_posterior_pct: 0.0,
        post_entry_bar1_features_32: Vec::new(),
        culprit_anomaly_weights_32: Vec::new(),
        was_protected_from_giveback: false,
        giveback_ratchet_bar_offset: 0,
        giveback_r_saved: 0.0,
        was_scaled_with_house_money: false,
        house_money_scale_bar_offset: None,
        house_money_additional_r_gained: 0.0,
        recycled_risk_multiplier_gamma: 1.0,
        recycled_extra_r_compounded: 0.0,
        almgren_chriss_urgency_kappa: 0.0,
        realized_execution_slippage_bps: pos.slip_bps_one_way,
        slippage_saved_usd: 0.0,
        tail_hedge_ratio_theta: 0.0,
        tail_hedge_convexity_payout_usd: 0.0,
        sor_selected_venue: "HISTORICAL_PESSIMISTIC".to_string(),
        sor_markout_toxicity_bps: pos.slip_bps_one_way,
        charm_vanna_unpinning_impulse: 0.0,
        charm_vanna_unpin_extra_r: 0.0,
        occ_tims_pm_relief_ratio_pct: 0.0,
    };

    (blotter, closed_row)
}

#[inline]
fn exit_iv_for_bar(bucket: ProofBucket, snap: &CausalBarPhysicsSnapshot) -> f64 {
    let base = match bucket {
        ProofBucket::Scalp0Dte1m | ProofBucket::Scalp0Dte5m | ProofBucket::Brain02LargeCapDay => {
            snap.vix9d_spot_dec
        }
        _ => snap.vix_spot_dec,
    };
    (base * (snap.causal_realized_vol / base.max(0.12)).clamp(0.75, 2.40)).clamp(0.12, 1.85)
}

/// Builds a single strictly causal `CausalTrajectoryBar` at index `t` (`bars[t + 1]` must be `<= max_ts_sec`).
#[inline]
fn build_single_causal_bar(
    sym: &str,
    bucket: ProofBucket,
    bars: &[RealExchangeOhlcvBar],
    snaps: &[CausalBarPhysicsSnapshot],
    t: usize,
    max_ts_sec: i64,
    force_eod_override: bool,
) -> Option<CausalTrajectoryBar> {
    if t + 1 >= bars.len() {
        return None;
    }
    let snap = &snaps[t];
    if !snap.valid {
        return None;
    }
    let bar_t = &bars[t];
    if !symbol_matches_bucket(sym, bar_t.close, bucket) {
        return None;
    }
    let is_scalp = matches!(bucket, ProofBucket::Scalp0Dte1m | ProofBucket::Scalp0Dte5m);
    let is_day = matches!(
        bucket,
        ProofBucket::Brain01PennyDay | ProofBucket::Brain02LargeCapDay
    );
    let force_flat = force_eod_override
        || (is_scalp && snap.hhmm >= 1550)
        || (is_day && snap.hhmm >= 1555);

    // If not force_flat, require `bars[t + 2].timestamp_sec <= max_ts_sec` so open-to-open return is resolved!
    // On the final EOD flatten bar (`force_flat == true`), `a_{T_eod} = 0.0`, so `ret_oo` is unused (`0 * ret_oo`),
    // only `one_way_cost` on `bars[t]`/`bars[t+1]` is needed to charge exact EOD liquidation friction!
    let open_1 = bars[t + 1].open.max(1e-4);
    let (ret_oo_shares, intrabar_mae) = if force_flat {
        (0.0_f32, 0.0_f32)
    } else if t + 2 < bars.len() && bars[t + 2].timestamp_sec <= max_ts_sec {
        let open_2 = bars[t + 2].open.max(1e-4);
        let r = ((open_2 - open_1) / open_1).clamp(-0.25, 0.25) as f32;
        let ib_mae = (((open_1 - bars[t + 1].low).max(bars[t + 1].high - open_1)) / open_1)
            .clamp(0.0, 0.15) as f32;
        (r, ib_mae)
    } else {
        return None;
    };

    let is_penny = is_penny_symbol_cand(sym, bar_t.close);
    let slip_bps = explicit_one_way_slippage_bps(sym, open_1, is_penny);
    let one_way_cost_shares = ((slip_bps / 10_000.0) + 0.000015) as f32;
    let rt_cost_shares = (2.0 * one_way_cost_shares as f64).max(4e-4);

    let option_allowed = !is_penny && is_option_vehicle_allowed(sym, open_1);
    // Risk-normalized underlying-equivalent option return & friction so BPTT compares `V_SHARES` vs `V_OPTION` apples-to-apples!
    let (
        one_way_cost_option,
        option_gamma_boost,
        option_theta_per_bar,
        ret_oo_option,
        is_vert_bar,
    ) = if option_allowed {
        let (_, is_vert, dte_days) = resolve_vehicle_routing_config(
            sym,
            open_1,
            bucket,
            snap.hhmm,
            snap.vix_spot_dec,
            true,
        );
        let legs = if is_vert { 2.0 } else { 1.0 };
        let opt_spread = estimate_option_spread_per_leg(sym, open_1, is_vert);
        let eff_delta = if is_vert { 0.22 } else { 0.48 };
        let underlying_equiv_one_way_opt =
            (((0.5 * legs * opt_spread + legs * 0.0065) / eff_delta) / open_1)
                .clamp(0.0004, 0.025) as f32;
        let dt_bars = if is_scalp {
            if bucket == ProofBucket::Scalp0Dte1m {
                390.0 * dte_days.max(0.08)
            } else {
                78.0 * dte_days.max(0.08)
            }
        } else if is_day {
            78.0 * dte_days.max(1.0)
        } else {
            7.0 * dte_days.max(5.0)
        };
        let approx_prem_frac =
            (snap.vix_spot_dec * (dte_days / 252.0).sqrt() * 0.40).max(0.004);
        let theta_underlying_eq =
            ((approx_prem_frac / dt_bars.max(4.0)) / eff_delta).clamp(0.00005, 0.008) as f32;
        let gamma_boost = if is_vert {
            0.88_f32
        } else if ret_oo_shares.abs() > (2.0 * underlying_equiv_one_way_opt) {
            1.14_f32
        } else {
            0.96_f32
        };
        (
            underlying_equiv_one_way_opt,
            gamma_boost,
            theta_underlying_eq,
            if force_flat {
                0.0
            } else {
                (ret_oo_shares * gamma_boost - theta_underlying_eq).clamp(-0.25, 0.25)
            },
            is_vert,
        )
    } else {
        (0.02, 1.0, 0.0, ret_oo_shares, false)
    };

    let norm_rem = normalized_session_remaining(bucket, snap.hhmm);
    let x_212 = snap.to_raw_180_vector(norm_rem, 0.0, false);
    let entry_allowed = !force_flat && entry_allowed_at_hhmm(bucket, snap.hhmm);

    // Strictly causal 4-channel counterfactual net returns (`<= max_ts_sec`, same session when intraday):
    // `[LongSharesNet, ShortSharesNet, LongCallNet, LongPutNet]`
    let mut cf_net_returns = [0.0_f32; 4];
    let mut cf_valid = false;
    if !force_flat {
        let rt_opt = (2.0 * one_way_cost_option).max(8e-4);
        let rt_sh = rt_cost_shares as f32;
        let mut mae_l_sh = 0.0_f32;
        let mut mae_s_sh = 0.0_f32;
        let mut mae_l_opt = 0.0_f32;
        let mut mae_s_opt = 0.0_f32;
        let mut acc_l_sh = 0.0_f32;
        let mut acc_s_sh = 0.0_f32;
        let mut acc_l_opt = 0.0_f32;
        let mut acc_s_opt = 0.0_f32;
        let mut best_l_sh = f32::NEG_INFINITY;
        let mut best_s_sh = f32::NEG_INFINITY;
        let mut best_l_opt = f32::NEG_INFINITY;
        let mut best_s_opt = f32::NEG_INFINITY;
        let mut w_sum = 0.0_f32;
        let horizons: [usize; 4] = if bucket == ProofBucket::Scalp0Dte1m {
            [10, 25, 50, 90]
        } else if is_scalp || is_day {
            [4, 9, 18, 32]
        } else {
            [2, 4, 8, 16]
        };
        let max_h = horizons[3];
        for step_k in 1..=max_h {
            let end_idx = t + 1 + step_k;
            if end_idx >= bars.len() || bars[end_idx].timestamp_sec > max_ts_sec {
                break;
            }
            if (is_scalp || is_day) && snaps[end_idx - 1].day_key != snap.day_key {
                break;
            }
            let open_end = bars[end_idx].open.max(1e-4);
            let gross_sh = ((open_end - open_1) / open_1).clamp(-0.30, 0.30) as f32;
            let theta_cum = (option_theta_per_bar * (step_k as f32)).min(0.15);
            mae_l_sh = mae_l_sh.max((-gross_sh).max(0.0));
            mae_s_sh = mae_s_sh.max(gross_sh.max(0.0));
            let mult = if is_vert_bar {
                0.86_f32
            } else if gross_sh.abs() > rt_opt {
                1.08_f32
            } else {
                0.96_f32
            };
            let gross_l_opt = gross_sh * mult - theta_cum;
            let gross_s_opt = -gross_sh * mult - theta_cum;
            mae_l_opt = mae_l_opt.max((-gross_l_opt).max(0.0));
            mae_s_opt = mae_s_opt.max((-gross_s_opt).max(0.0));

            let is_eod_bar = (is_scalp && snaps[end_idx].hhmm >= 1550)
                || (is_day && snaps[end_idx].hhmm >= 1555);
            let w = if step_k == horizons[0] {
                0.16
            } else if step_k == horizons[1] {
                0.28
            } else if step_k == horizons[2] {
                0.32
            } else if step_k == horizons[3] {
                0.24
            } else if is_eod_bar {
                0.20
            } else {
                0.0
            };
            if w > 0.0 {
                let v_l_sh = gross_sh - 0.18 * mae_l_sh - rt_sh;
                let v_s_sh = -gross_sh - 0.18 * mae_s_sh - rt_sh;
                w_sum += w;
                acc_l_sh += w * v_l_sh;
                acc_s_sh += w * v_s_sh;
                best_l_sh = best_l_sh.max(v_l_sh);
                best_s_sh = best_s_sh.max(v_s_sh);
                if option_allowed {
                    let v_l_opt = gross_l_opt - 0.18 * mae_l_opt - rt_opt;
                    let v_s_opt = gross_s_opt - 0.18 * mae_s_opt - rt_opt;
                    acc_l_opt += w * v_l_opt;
                    acc_s_opt += w * v_s_opt;
                    best_l_opt = best_l_opt.max(v_l_opt);
                    best_s_opt = best_s_opt.max(v_s_opt);
                }
                cf_valid = true;
            }
            if is_eod_bar {
                break;
            }
        }
        if cf_valid && w_sum > 0.0 {
            let m_l_sh = acc_l_sh / w_sum;
            let m_s_sh = acc_s_sh / w_sum;
            cf_net_returns[0] = (0.64 * m_l_sh + 0.36 * best_l_sh).clamp(-0.35, 0.35);
            cf_net_returns[1] = (0.64 * m_s_sh + 0.36 * best_s_sh).clamp(-0.35, 0.35);
            cf_net_returns[2] = if option_allowed {
                let m_l_opt = acc_l_opt / w_sum;
                (0.64 * m_l_opt + 0.36 * best_l_opt).clamp(-0.35, 0.35)
            } else {
                cf_net_returns[0]
            };
            cf_net_returns[3] = if option_allowed {
                let m_s_opt = acc_s_opt / w_sum;
                (0.64 * m_s_opt + 0.36 * best_s_opt).clamp(-0.35, 0.35)
            } else {
                cf_net_returns[1]
            };
        }
    }

    // Strictly causal multi-horizon physics targets (`[5, 15, 40, 100]` on `1m` matching `[1, 3, 8, 24]` on `5m`/`1h`):
    // `[q25_ret, q50_ret, q75_ret, ln(Range / RT_Cost)]` populated ONLY when `t + 1 + delta <= max_ts_sec`!
    let mut wm_targets = [0.0_f32; WM_OUT_DIM];
    let mut wm_valid = [false; WM_OUT_DIM];
    let wm_deltas: [usize; 4] = if bucket == ProofBucket::Scalp0Dte1m {
        [5, 15, 40, 100]
    } else {
        WM_HORIZONS
    };
    for (h_idx, &delta) in wm_deltas.iter().enumerate() {
        let end_idx = t + 1 + delta;
        if end_idx < bars.len()
            && bars[end_idx].timestamp_sec <= max_ts_sec
            && (!(is_scalp || is_day) || snaps[end_idx].day_key == snap.day_key)
        {
            // For intraday buckets, stay within same session or up to session close
            let open_end = bars[end_idx].open.max(1e-4);
            let ret_delta = ((open_end - open_1) / open_1).clamp(-0.35, 0.35) as f32;
            let mut hi_max = bars[t + 1].high;
            let mut lo_min = bars[t + 1].low;
            for b in &bars[(t + 1)..=end_idx] {
                hi_max = hi_max.max(b.high);
                lo_min = lo_min.min(b.low);
            }
            let range_frac = ((hi_max - lo_min) / open_1).max(1e-5);
            let rc_log = (range_frac / rt_cost_shares).max(0.05).ln().clamp(-3.0, 5.0) as f32;
            let base = h_idx * 4;
            wm_targets[base] = ret_delta;
            wm_targets[base + 1] = ret_delta;
            wm_targets[base + 2] = ret_delta;
            wm_targets[base + 3] = rc_log;
            wm_valid[base] = true;
            wm_valid[base + 1] = true;
            wm_valid[base + 2] = true;
            wm_valid[base + 3] = true;
        }
    }

    Some(CausalTrajectoryBar {
        ts_sec: bar_t.timestamp_sec,
        x_212,
        ret_oo_shares,
        ret_oo_option,
        one_way_cost_shares,
        one_way_cost_option,
        one_way_cost_bps: slip_bps as f32,
        session_time_left: norm_rem as f32,
        option_allowed,
        entry_allowed,
        force_flat,
        option_gamma_boost,
        option_theta_per_bar,
        cf_net_returns,
        cf_valid,
        intrabar_mae,
        wm_targets,
        wm_valid,
    })
}

/// FIX #2: Slices ALL intraday brains (`SCALP_0DTE 1m`, `SCALP_0DTE 5m`, `BRAIN-01`, `BRAIN-02`)
/// into individual `(Symbol, TradingDay)` Session Episodes (`[09:35 ET .. 15:50/15:55 ET]`),
/// carrying ONLY causal cross-day regime context `prior_regime_z` across session boundaries, and
/// hard-enforcing `force_flat = true` (`a_{T_eod} = 0.0` + exact EOD flatten friction) on the
/// final bar of EVERY daily episode!
#[allow(clippy::too_many_arguments)]
fn build_causal_session_episodes(
    bucket: ProofBucket,
    symbol_bars: &HashMap<String, Vec<RealExchangeOhlcvBar>>,
    precomputed: &HashMap<String, Vec<CausalBarPhysicsSnapshot>>,
    min_day: i64,
    max_ts_sec: i64,
    world_model: Option<&PhysicsWorldModel>,
    feat_mean: &[f32],
    feat_var: &[f32],
) -> Vec<CausalSessionEpisode> {
    let mut syms: Vec<String> = symbol_bars.keys().cloned().collect();
    syms.sort();

    let is_intraday = bucket.is_intraday();
    let eod_hhmm = session_eod_hhmm(bucket);
    let mut episodes = Vec::new();

    for sym in syms {
        let Some(bars) = symbol_bars.get(&sym) else {
            continue;
        };
        let Some(snaps) = precomputed.get(&sym) else {
            continue;
        };
        if bars.len() < 8 || snaps.len() != bars.len() {
            continue;
        }

        if is_intraday {
            // Group indices by `day_key` within `[09:35 ET ..= eod_hhmm]` strictly `<= max_ts_sec`
            let mut day_groups: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
            for t in 0..bars.len().saturating_sub(1) {
                if bars[t + 1].timestamp_sec > max_ts_sec {
                    break;
                }
                let snap = &snaps[t];
                if !snap.valid || snap.day_key < min_day {
                    continue;
                }
                if !(935..=eod_hhmm).contains(&snap.hhmm) {
                    continue;
                }
                if !symbol_matches_bucket(&sym, bars[t].close, bucket) {
                    continue;
                }
                day_groups.entry(snap.day_key).or_default().push(t);
            }

            let mut cross_day_regime_z = [0.0_f32; WM_DIM];
            for (d_key, indices) in day_groups {
                if indices.len() < 6 {
                    continue;
                }
                let last_i = indices.len() - 1;
                let mut ep_bars = Vec::with_capacity(indices.len());
                for (pos_i, &bar_i) in indices.iter().enumerate() {
                    let is_last_session_bar = pos_i == last_i;
                    if let Some(mut cbar) = build_single_causal_bar(
                        &sym,
                        bucket,
                        bars,
                        snaps,
                        bar_i,
                        max_ts_sec,
                        is_last_session_bar,
                    ) {
                        if is_last_session_bar {
                            cbar.force_flat = true;
                            cbar.entry_allowed = false;
                        }
                        ep_bars.push(cbar);
                    }
                }
                if ep_bars.len() >= 6 {
                    // Ensure final bar of session episode ALWAYS enforces `a_{T_eod} = 0.0`!
                    if let Some(last_b) = ep_bars.last_mut() {
                        last_b.force_flat = true;
                        last_b.entry_allowed = false;
                    }
                    let prior_z = cross_day_regime_z;
                    if let Some(wm) = world_model
                        && feat_mean.len() == RL_STATE_DIM
                        && feat_var.len() == RL_STATE_DIM
                    {
                        let mut z_step = prior_z;
                        for b in &ep_bars {
                            let xn = normalize_212_with_stats(&b.x_212, feat_mean, feat_var);
                            z_step = wm.forward_step(&xn, &z_step).0;
                        }
                        for j in 0..WM_DIM {
                            cross_day_regime_z[j] = 0.45 * z_step[j];
                        }
                    }

                    if let (Some(first_b), Some(last_b)) = (ep_bars.first(), ep_bars.last()) {
                        episodes.push(CausalSessionEpisode {
                            symbol: sym.clone(),
                            day_key: d_key,
                            start_ts_sec: first_b.ts_sec,
                            end_ts_sec: last_b.ts_sec,
                            is_intraday_session: true,
                            prior_regime_z: prior_z,
                            bars: ep_bars,
                        });
                    }
                }
            }
        } else {
            // Swing (`1h`): slice into causal multi-week episodes (`<= 70` hourly bars = ~10 trading days)
            let mut all_swing_bars = Vec::new();
            for t in 0..bars.len().saturating_sub(2) {
                if bars[t + 2].timestamp_sec > max_ts_sec {
                    break;
                }
                if snaps[t].day_key < min_day {
                    continue;
                }
                if let Some(cbar) =
                    build_single_causal_bar(&sym, bucket, bars, snaps, t, max_ts_sec, false)
                {
                    all_swing_bars.push(cbar);
                }
            }
            let mut start = 0usize;
            let mut z_carry = [0.0_f32; WM_DIM];
            while start + 12 <= all_swing_bars.len() {
                let end = (start + 70).min(all_swing_bars.len());
                let slice = all_swing_bars[start..end].to_vec();
                let prior_z = z_carry;
                if let Some(wm) = world_model
                    && feat_mean.len() == RL_STATE_DIM
                    && feat_var.len() == RL_STATE_DIM
                {
                    let mut z_step = prior_z;
                    for b in &slice {
                        let xn = normalize_212_with_stats(&b.x_212, feat_mean, feat_var);
                        z_step = wm.forward_step(&xn, &z_step).0;
                    }
                    z_carry = z_step;
                }
                if let (Some(first_b), Some(last_b)) = (slice.first(), slice.last()) {
                    episodes.push(CausalSessionEpisode {
                        symbol: sym.clone(),
                        day_key: est_trading_day_key(first_b.ts_sec),
                        start_ts_sec: first_b.ts_sec,
                        end_ts_sec: last_b.ts_sec,
                        is_intraday_session: false,
                        prior_regime_z: prior_z,
                        bars: slice,
                    });
                }
                start += 70;
            }
        }
    }

    episodes.sort_by_key(|ep| (ep.end_ts_sec, ep.symbol.clone()));
    episodes
}

type WmSymbolCache = Vec<([f32; RL_STATE_DIM], [f32; WM_DIM], [f32; WM_OUT_DIM])>;
type PretrainM5Context<'a> = (
    &'a HashMap<String, Vec<RealExchangeOhlcvBar>>,
    &'a HashMap<String, Vec<CausalBarPhysicsSnapshot>>,
    i64,
);

/// Fast zero-allocation pre-indexed timeline structure for a specific `ProofBucket`.
struct BucketTimelineIndex<'a> {
    bucket: ProofBucket,
    syms: Vec<String>,
    bars_by_sym: Vec<&'a [RealExchangeOhlcvBar]>,
    snaps_by_sym: Vec<&'a [CausalBarPhysicsSnapshot]>,
    all_ts: Vec<i64>,
    /// For each `ts_idx` in `0..all_ts.len()`, list of `(sym_id, bar_idx)`.
    at_ts: Vec<Vec<(u16, u32)>>,
}

impl<'a> BucketTimelineIndex<'a> {
    fn build(
        bucket: ProofBucket,
        symbol_bars: &'a HashMap<String, Vec<RealExchangeOhlcvBar>>,
        precomputed: &'a HashMap<String, Vec<CausalBarPhysicsSnapshot>>,
    ) -> Self {
        let mut cand_syms: Vec<String> = symbol_bars
            .keys()
            .filter(|sym| {
                symbol_bars
                    .get(*sym)
                    .is_some_and(|bs| bs.iter().any(|b| symbol_matches_bucket(sym, b.close, bucket)))
            })
            .cloned()
            .collect();
        cand_syms.sort();

        let mut bars_by_sym = Vec::with_capacity(cand_syms.len());
        let mut snaps_by_sym = Vec::with_capacity(cand_syms.len());
        let mut ts_set: BTreeSet<i64> = BTreeSet::new();

        for sym in &cand_syms {
            let bs = symbol_bars[sym].as_slice();
            let sn = precomputed[sym].as_slice();
            bars_by_sym.push(bs);
            snaps_by_sym.push(sn);
            for b in bs {
                let (wd, hhmm) = ny_et_hhmm(b.timestamp_sec);
                if !matches!(wd, chrono::Weekday::Sat | chrono::Weekday::Sun)
                    && (930..=1600).contains(&hhmm)
                {
                    ts_set.insert(b.timestamp_sec);
                }
            }
        }

        let all_ts: Vec<i64> = ts_set.into_iter().collect();
        let mut ts_to_pos: HashMap<i64, usize> = HashMap::with_capacity(all_ts.len());
        for (ti, &ts) in all_ts.iter().enumerate() {
            ts_to_pos.insert(ts, ti);
        }

        let mut at_ts = vec![Vec::new(); all_ts.len()];
        for (sym_id, bs) in bars_by_sym.iter().enumerate() {
            for (bar_idx, b) in bs.iter().enumerate() {
                if let Some(&ti) = ts_to_pos.get(&b.timestamp_sec) {
                    at_ts[ti].push((sym_id as u16, bar_idx as u32));
                }
            }
        }

        Self {
            bucket,
            syms: cand_syms,
            bars_by_sym,
            snaps_by_sym,
            all_ts,
            at_ts,
        }
    }

    /// Precomputes normalized `x_norm` and World-Model `(z_wm, wm_pred)` across all symbols up to `cutoff_ts`
    /// in parallel via `rayon` so all 24 candidates share identical causal latents in O(1) lookup!
    fn precompute_wm_cache(
        &self,
        world_model: &PhysicsWorldModel,
        feat_mean: &[f32],
        feat_var: &[f32],
        cutoff_ts: i64,
    ) -> Vec<WmSymbolCache> {
        let is_intraday = self.bucket.is_intraday();
        self.bars_by_sym
            .par_iter()
            .zip(self.snaps_by_sym.par_iter())
            .map(|(bars, snaps)| {
                let end_b = bars.partition_point(|b| b.timestamp_sec <= cutoff_ts);
                let mut vec_out =
                    vec![([0.0_f32; RL_STATE_DIM], [0.0_f32; WM_DIM], [0.0_f32; WM_OUT_DIM]); end_b];
                let mut prev_z = [0.0_f32; WM_DIM];
                let mut prev_day = i64::MIN;
                for i in 0..end_b {
                    let snap = &snaps[i];
                    if !snap.valid {
                        continue;
                    }
                    if is_intraday && snap.day_key != prev_day {
                        if prev_day != i64::MIN {
                            for z_v in &mut prev_z {
                                *z_v *= 0.45;
                            }
                        }
                        prev_day = snap.day_key;
                    }
                    let norm_rem = normalized_session_remaining(self.bucket, snap.hhmm);
                    let raw_x = snap.to_raw_180_vector(norm_rem, 0.0, false);
                    let x_n = normalize_212_with_stats(&raw_x, feat_mean, feat_var);
                    let (z_wm, wm_pred) = world_model.forward_step(&x_n, &prev_z);
                    prev_z = z_wm;
                    vec_out[i] = (x_n, z_wm, wm_pred);
                }
                vec_out
            })
            .collect()
    }
}

#[derive(Debug, Clone)]
struct SettledCausalTrade {
    sym_id: usize,
    symbol: String,
    entry_bar_ord: usize,
    exit_bar_ord: usize,
    exit_ts_sec: i64,
    direction: TradeDirection,
    is_option_routed: bool,
    gross_pnl_usd: f64,
    net_pnl_usd: f64,
    net_r: f64,
    exit_reason: String,
    peak_v_liq_ret: f64,
}

#[derive(Debug, Clone, Default)]
struct DiscreteCausalEvalSummary {
    score: f64,
    sharpe: f64,
    net_pnl_usd: f64,
    net_pf: f64,
    red_loss_usd: f64,
    win_r_sum: f64,
    winners_count: usize,
    max_dd_pct: f64,
    trades: Vec<SettledCausalTrade>,
}

/// Evaluates a single `PopulationMember` using exact discrete `proof_harness` physics
/// on `[eval_start_ts ..= eval_cutoff_ts]` using pre-indexed `BucketTimelineIndex` + `wm_cache`.
fn evaluate_single_member_indexed(
    member: &mut PopulationMember,
    idx: &BucketTimelineIndex<'_>,
    wm_cache: &[WmSymbolCache],
    eval_start_ts: i64,
    eval_cutoff_ts: i64,
    starting_equity: f64,
) -> DiscreteCausalEvalSummary {
    let n_syms = idx.syms.len();
    let bucket = idx.bucket;
    let is_intraday = bucket.is_intraday();
    let max_slots = if is_intraday { 2usize } else { 3usize };

    let mut equity = starting_equity.max(50_000.0);
    let mut peak_eq = equity;
    let mut max_dd_pct = 0.0_f64;
    let mut hiddens = vec![[0.0_f32; HIDDEN_DIM]; n_syms];
    let mut sym_last_day = vec![i64::MIN; n_syms];
    let mut sym_ema_turn = vec![0.0_f32; n_syms];
    let mut sym_prev_act = vec![0.0_f32; n_syms];
    let mut open_positions: Vec<ActiveSimulatedPosition> = Vec::with_capacity(max_slots);
    let mut pending_signals: Vec<PendingNextBarSignal> = Vec::with_capacity(max_slots);
    let mut settled: Vec<SettledCausalTrade> = Vec::new();
    let mut trade_seq = 1u128;
    let mut flat_steps = 0usize;
    let mut sum_dz_margin = 0.0_f64;

    let end_ti = idx.all_ts.partition_point(|&ts| ts <= eval_cutoff_ts);
    let start_ti = idx.all_ts[..end_ti].partition_point(|&ts| ts < eval_start_ts);
    let prewarm_ti = start_ti.saturating_sub(48);

    for ti in prewarm_ti..start_ti {
        let t_sec = idx.all_ts[ti];
        let cur_day = est_trading_day_key(t_sec);
        let (_, hhmm) = ny_et_hhmm(t_sec);
        let norm_rem = normalized_session_remaining(bucket, hhmm);
        for &(sid_u16, bidx_u32) in &idx.at_ts[ti] {
            let sid = sid_u16 as usize;
            let bi = bidx_u32 as usize;
            if sid >= 64 || bi >= wm_cache[sid].len() {
                continue;
            }
            if is_intraday && sym_last_day[sid] != cur_day {
                hiddens[sid] = [0.0; HIDDEN_DIM];
                sym_ema_turn[sid] *= 0.25;
                sym_prev_act[sid] = 0.0;
                sym_last_day[sid] = cur_day;
            }
            let snap = &idx.snaps_by_sym[sid][bi];
            if !snap.valid {
                continue;
            }
            let bar_t = &idx.bars_by_sym[sid][bi];
            let sym = &idx.syms[sid];
            let is_penny = is_penny_symbol_cand(sym, bar_t.close);
            let opt_allowed = !is_penny && is_option_vehicle_allowed(sym, bar_t.close);
            let slip_bps = explicit_one_way_slippage_bps(sym, bar_t.close, is_penny) as f32;
            let (ref x_n, ref z_wm, ref wm_pred) = wm_cache[sid][bi];
            let self_s = SelfAwarenessState {
                current_pos: 0.0,
                unrealized_net_ret: 0.0,
                peak_excursion: 0.0,
                drawdown_from_peak: 0.0,
                log_bars_held: 0.0,
                one_way_cost_bps: slip_bps,
                session_time_left: norm_rem as f32,
                ema_turnover: sym_ema_turn[sid],
            };
            let step = member.net.forward_step_with_wm_cal(
                x_n,
                z_wm,
                wm_pred,
                self_s,
                &hiddens[sid],
                *z_wm,
                member.hparams.cost_sensitivity,
                member.hparams.downside_aversion,
                member.hparams.conformal_q_bps,
                opt_allowed,
            );
            hiddens[sid] = step.next_hidden;
        }
    }

    for ti in start_ti..end_ti {
        let t_sec = idx.all_ts[ti];
        let cur_day = est_trading_day_key(t_sec);
        let active_pairs = &idx.at_ts[ti];
        if active_pairs.is_empty() {
            continue;
        }

        // Fast lookup of bar_idx for symbols active at `ti`
        // (n_syms <= 59, stack array lookup in L1 cache!)
        let mut bar_at_t = [u32::MAX; 64];
        for &(sid, bidx) in active_pairs {
            if (sid as usize) < 64 {
                bar_at_t[sid as usize] = bidx;
                if is_intraday && sym_last_day[sid as usize] != cur_day {
                    hiddens[sid as usize] = [0.0; HIDDEN_DIM];
                    sym_ema_turn[sid as usize] *= 0.25;
                    sym_prev_act[sid as usize] = 0.0;
                    sym_last_day[sid as usize] = cur_day;
                }
            }
        }

        // 1. Fill pending t+1 open signals
        let mut newly_opened: [Uuid; 8] = [Uuid::nil(); 8];
        let mut newly_len = 0usize;
        for sig in pending_signals.drain(..) {
            let bidx = bar_at_t[sig.sym_id];
            if bidx == u32::MAX {
                continue;
            }
            let bi = bidx as usize;
            let entry_bar = &idx.bars_by_sym[sig.sym_id][bi];
            let entry_snap = &idx.snaps_by_sym[sig.sym_id][bi];
            let tid = Uuid::from_u128(0xA000_0000_0000_0000_u128 | trade_seq);
            if let Some(pos) = open_simulated_position_at_next_bar(
                sig.sym_id,
                &sig.symbol,
                sig.bucket,
                sig.brain_id,
                sig.signal_ts_sec,
                sig.signal_day_key,
                sig.direction,
                sig.target_action,
                sig.tail_stop_dist,
                sig.is_option_routed,
                sig.is_vertical_combo,
                sig.is_penny_tier,
                sig.dte_days,
                sig.kelly_risk_budget_usd,
                equity,
                entry_bar,
                entry_snap,
                sig.embedding_128,
                sig.confidence_pct,
                sig.active_gen_id,
                tid,
            ) {
                trade_seq += 1;
                if newly_len < newly_opened.len() {
                    newly_opened[newly_len] = tid;
                    newly_len += 1;
                }
                open_positions.push(pos);
            }
        }

        // 2. Step open positions
        let mut i_pos = 0usize;
        while i_pos < open_positions.len() {
            let sid = open_positions[i_pos].sym_id;
            let bidx = bar_at_t[sid];
            if bidx == u32::MAX {
                i_pos += 1;
                continue;
            }
            let bi = bidx as usize;
            let bar_t = &idx.bars_by_sym[sid][bi];
            let snap_t = &idx.snaps_by_sym[sid][bi];
            let is_new = newly_opened[..newly_len].contains(&open_positions[i_pos].trade_id);
            let exit_iv = exit_iv_for_bar(bucket, snap_t);

            if let Some((exit_px, reason, is_full_stop)) =
                step_active_position_on_bar(&mut open_positions[i_pos], bar_t, exit_iv, is_new)
            {
                let pos = open_positions.swap_remove(i_pos);
                sym_ema_turn[sid] =
                    (0.90 * sym_ema_turn[sid] + 0.10 * sym_prev_act[sid].abs()).clamp(0.0, 2.0);
                sym_prev_act[sid] = 0.0;
                let (blotter, _) = finalize_settled_position(
                    &pos,
                    t_sec,
                    exit_px,
                    reason,
                    is_full_stop,
                    exit_iv,
                    ProofRunMode::RunBOnlineRl,
                );
                equity += blotter.net_pnl_usd;
                peak_eq = peak_eq.max(equity);
                max_dd_pct = max_dd_pct.max(((peak_eq - equity) / peak_eq.max(1.0)) * 100.0);
                settled.push(SettledCausalTrade {
                    sym_id: pos.sym_id,
                    symbol: pos.symbol,
                    entry_bar_ord: pos.entry_bar_ord as usize,
                    exit_bar_ord: bi,
                    exit_ts_sec: t_sec,
                    direction: pos.direction,
                    is_option_routed: pos.is_option_routed,
                    gross_pnl_usd: blotter.gross_pnl_usd,
                    net_pnl_usd: blotter.net_pnl_usd,
                    net_r: blotter.net_r,
                    exit_reason: blotter.exit_reason,
                    peak_v_liq_ret: pos.peak_v_liq_ret,
                });
            } else {
                if !is_new && snap_t.valid && bi < wm_cache[sid].len() {
                    let pos = &mut open_positions[i_pos];
                    let (ref x_n, ref z_wm, ref wm_pred) = wm_cache[sid][bi];
                    let norm_rem = normalized_session_remaining(bucket, snap_t.hhmm);
                    let (self_s, _) = build_active_position_self_awareness(
                        pos,
                        t_sec,
                        bar_t.close,
                        exit_iv,
                        norm_rem,
                        sym_ema_turn[sid],
                    );
                    let step = member.net.forward_step_with_wm_cal(
                        x_n,
                        z_wm,
                        wm_pred,
                        self_s,
                        &hiddens[sid],
                        *z_wm,
                        member.hparams.cost_sensitivity,
                        member.hparams.downside_aversion,
                        member.hparams.conformal_q_bps,
                        !pos.is_penny_tier && is_option_vehicle_allowed(&pos.symbol, bar_t.close),
                    );
                    hiddens[sid] = step.next_hidden;
                    sym_ema_turn[sid] = (0.90 * sym_ema_turn[sid]
                        + 0.10 * (step.action - sym_prev_act[sid]).abs())
                    .clamp(0.0, 2.0);
                    sym_prev_act[sid] = step.action;

                    let pos_sign = pos.direction.sign() as f32;
                    let exit_snap = (0.22 * pos.current_target_action.abs()).clamp(0.028, 0.075);
                    if step.action.abs() < exit_snap || step.action.signum() != pos_sign {
                        pos.pending_exit = true;
                    } else if (step.action - pos.current_target_action).abs() >= 0.30 {
                        pos.pending_scale_action = Some(step.action);
                    }
                }
                i_pos += 1;
            }
        }

        // 3. Evaluate flat symbols
        let (_, hhmm) = ny_et_hhmm(t_sec);
        if !entry_allowed_at_hhmm(bucket, hhmm) || open_positions.len() >= max_slots {
            continue;
        }
        let norm_rem = normalized_session_remaining(bucket, hhmm);
        let mut candidates: Vec<(f64, PendingNextBarSignal)> = Vec::new();

        for &(sid_u16, bidx_u32) in active_pairs {
            let sid = sid_u16 as usize;
            let bi = bidx_u32 as usize;
            if bi >= wm_cache[sid].len() {
                continue;
            }
            let snap = &idx.snaps_by_sym[sid][bi];
            if !snap.valid {
                continue;
            }
            let bar_t = &idx.bars_by_sym[sid][bi];
            let sym = &idx.syms[sid];
            if !symbol_matches_bucket(sym, bar_t.close, bucket) {
                continue;
            }
            if open_positions.iter().any(|p| p.sym_id == sid)
                || pending_signals.iter().any(|s| s.sym_id == sid)
            {
                continue;
            }
            let is_penny = is_penny_symbol_cand(sym, bar_t.close);
            let opt_allowed = !is_penny && is_option_vehicle_allowed(sym, bar_t.close);
            let slip_bps = explicit_one_way_slippage_bps(sym, bar_t.close, is_penny) as f32;
            let (ref x_n, ref z_wm, ref wm_pred) = wm_cache[sid][bi];
            let self_s = SelfAwarenessState {
                current_pos: 0.0,
                unrealized_net_ret: 0.0,
                peak_excursion: 0.0,
                drawdown_from_peak: 0.0,
                log_bars_held: 0.0,
                one_way_cost_bps: slip_bps,
                session_time_left: norm_rem as f32,
                ema_turnover: sym_ema_turn[sid],
            };
            let step = member.net.forward_step_with_wm_cal(
                x_n,
                z_wm,
                wm_pred,
                self_s,
                &hiddens[sid],
                *z_wm,
                member.hparams.cost_sensitivity,
                member.hparams.downside_aversion,
                member.hparams.conformal_q_bps,
                opt_allowed,
            );
            hiddens[sid] = step.next_hidden;
            sym_ema_turn[sid] =
                (0.90 * sym_ema_turn[sid] + 0.10 * (step.action - sym_prev_act[sid]).abs())
                    .clamp(0.0, 2.0);
            sym_prev_act[sid] = if step.action.abs() >= 0.070 {
                step.action
            } else {
                0.0
            };
            flat_steps += 1;
            sum_dz_margin += (step.raw_z.abs() - step.deadzone) as f64;

            if step.action.abs() >= 0.070 {
                let dir = if step.action > 0.0 {
                    TradeDirection::Long
                } else {
                    TradeDirection::Short
                };
                let (is_opt, is_vert, dte_days) = resolve_vehicle_routing_config(
                    sym,
                    bar_t.close,
                    bucket,
                    hhmm,
                    snap.vix_spot_dec,
                    step.vehicle_prob >= 0.5,
                );
                let risk_budget_usd =
                    (equity * (0.0055 + 0.0065 * step.action.abs() as f64)).clamp(420.0, 1_250.0);
                candidates.push((
                    step.action.abs() as f64,
                    PendingNextBarSignal {
                        sym_id: sid,
                        symbol: sym.clone(),
                        bucket,
                        brain_id: bucket.brain_id(),
                        signal_ts_sec: t_sec,
                        signal_day_key: cur_day,
                        signal_bar_ord: snap.bar_ord,
                        direction: dir,
                        target_action: step.action,
                        vehicle_prob: step.vehicle_prob,
                        tail_stop_dist: step.tail_stop_dist as f64,
                        confidence_pct: (55.0 + (step.action.abs() as f64) * 40.0)
                            .clamp(55.0, 98.5),
                        is_option_routed: is_opt,
                        is_vertical_combo: is_vert,
                        is_penny_tier: is_penny,
                        dte_days,
                        kelly_risk_budget_usd: risk_budget_usd,
                        embedding_128: Vec::new(),
                        active_gen_id: 0,
                    },
                ));
            }
        }
        candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let slots_left = max_slots.saturating_sub(open_positions.len());
        for (_, sig) in candidates.into_iter().take(slots_left.min(2)) {
            pending_signals.push(sig);
        }
    }

    // Settle any open positions Mark-to-Liquidation at `eval_cutoff_ts` (`<= eval_cutoff_ts` causal bar!)
    // so multi-day Swing windows never hide open unrealized drawdowns or leak across embargo boundaries.
    for pos in open_positions {
        let bars = idx.bars_by_sym[pos.sym_id];
        let snaps = idx.snaps_by_sym[pos.sym_id];
        let cut_ord = bars
            .partition_point(|b| b.timestamp_sec <= eval_cutoff_ts)
            .saturating_sub(1);
        if let Some(last_b) = bars.get(cut_ord) {
            let last_iv = snaps
                .get(cut_ord)
                .map(|s| exit_iv_for_bar(bucket, s))
                .unwrap_or(0.20);
            let (blotter, _) = finalize_settled_position(
                &pos,
                last_b.timestamp_sec,
                last_b.close,
                "EVAL_CUTOFF_MTM_LIQ".to_string(),
                false,
                last_iv,
                ProofRunMode::RunBOnlineRl,
            );
            equity += blotter.net_pnl_usd;
            peak_eq = peak_eq.max(equity);
            max_dd_pct = max_dd_pct.max(((peak_eq - equity) / peak_eq.max(1.0)) * 100.0);
            settled.push(SettledCausalTrade {
                sym_id: pos.sym_id,
                symbol: pos.symbol,
                entry_bar_ord: pos.entry_bar_ord as usize,
                exit_bar_ord: cut_ord,
                exit_ts_sec: last_b.timestamp_sec,
                direction: pos.direction,
                is_option_routed: pos.is_option_routed,
                gross_pnl_usd: blotter.gross_pnl_usd,
                net_pnl_usd: blotter.net_pnl_usd,
                net_r: blotter.net_r,
                exit_reason: blotter.exit_reason,
                peak_v_liq_ret: pos.peak_v_liq_ret,
            });
        }
    }

    let mut summary = summarize_settled_causal_trades(&settled, max_dd_pct);
    if summary.trades.is_empty() {
        summary.score = -1.5 + 0.40 * (sum_dz_margin / (flat_steps.max(1) as f64)).clamp(-0.9, 0.0);
    }
    member.causal_score = summary.score;
    member.causal_sharpe = summary.sharpe;
    member.causal_net_pnl = summary.net_pnl_usd;
    member.causal_trades = summary.trades.len();
    member.causal_pf = summary.net_pf;
    member.causal_red_loss_usd = summary.red_loss_usd;
    member.causal_win_r_sum = summary.win_r_sum;
    summary
}

/// Evaluates the FULL `PureAutonomousRlBrain` ensemble (`infer_ensemble_step_with_precomputed_wm`)
/// using the exact discrete non-differentiable post-cost `proof_harness` backtest across
/// `[eval_start_ts ..= eval_cutoff_ts]` for Step 3 Mandatory Full Causal Re-Test Before Promotion!
fn evaluate_brain_ensemble_retest_indexed(
    brain: &PureAutonomousRlBrain,
    idx: &BucketTimelineIndex<'_>,
    wm_cache: &[WmSymbolCache],
    eval_start_ts: i64,
    eval_cutoff_ts: i64,
    starting_equity: f64,
) -> DiscreteCausalEvalSummary {
    let n_syms = idx.syms.len();
    let bucket = idx.bucket;
    let is_intraday = bucket.is_intraday();
    let max_slots = if is_intraday { 2usize } else { 3usize };

    let mut equity = starting_equity.max(50_000.0);
    let mut peak_eq = equity;
    let mut max_dd_pct = 0.0_f64;
    let mut hiddens: Vec<Vec<[f32; HIDDEN_DIM]>> = vec![Vec::new(); n_syms];
    let mut sym_last_day = vec![i64::MIN; n_syms];
    let mut sym_ema_turn = vec![0.0_f32; n_syms];
    let mut sym_prev_act = vec![0.0_f32; n_syms];
    let turn_alpha: f32 = if is_intraday { 0.14 } else { 0.18 };
    let mut open_positions: Vec<ActiveSimulatedPosition> = Vec::with_capacity(max_slots);
    let mut pending_signals: Vec<PendingNextBarSignal> = Vec::with_capacity(max_slots);
    let mut settled: Vec<SettledCausalTrade> = Vec::new();
    let mut trade_seq = 1u128;

    let end_ti = idx.all_ts.partition_point(|&ts| ts <= eval_cutoff_ts);
    let start_ti = idx.all_ts[..end_ti].partition_point(|&ts| ts < eval_start_ts);
    let prewarm_ti = start_ti.saturating_sub(48);

    for ti in prewarm_ti..start_ti {
        let t_sec = idx.all_ts[ti];
        let cur_day = est_trading_day_key(t_sec);
        let (_, hhmm) = ny_et_hhmm(t_sec);
        let norm_rem = normalized_session_remaining(bucket, hhmm);
        for &(sid_u16, bidx_u32) in &idx.at_ts[ti] {
            let sid = sid_u16 as usize;
            let bi = bidx_u32 as usize;
            if sid >= 64 || bi >= wm_cache[sid].len() {
                continue;
            }
            if is_intraday && sym_last_day[sid] != cur_day {
                hiddens[sid].clear();
                sym_ema_turn[sid] *= 0.25;
                sym_prev_act[sid] = 0.0;
                sym_last_day[sid] = cur_day;
            }
            let snap = &idx.snaps_by_sym[sid][bi];
            if !snap.valid {
                continue;
            }
            let bar_t = &idx.bars_by_sym[sid][bi];
            let sym = &idx.syms[sid];
            let is_penny = is_penny_symbol_cand(sym, bar_t.close);
            let opt_allowed = !is_penny && is_option_vehicle_allowed(sym, bar_t.close);
            let slip_bps = explicit_one_way_slippage_bps(sym, bar_t.close, is_penny) as f32;
            let (ref x_n, ref z_wm, ref wm_pred) = wm_cache[sid][bi];
            let self_s = SelfAwarenessState {
                current_pos: 0.0,
                unrealized_net_ret: 0.0,
                peak_excursion: 0.0,
                drawdown_from_peak: 0.0,
                log_bars_held: 0.0,
                one_way_cost_bps: slip_bps,
                session_time_left: norm_rem as f32,
                ema_turnover: sym_ema_turn[sid],
            };
            let inf = brain.infer_ensemble_step_with_precomputed_wm(
                x_n,
                z_wm,
                wm_pred,
                self_s,
                &hiddens[sid],
                opt_allowed,
            );
            hiddens[sid] = inf.next_hiddens;
            sym_ema_turn[sid] = (1.0 - turn_alpha) * sym_ema_turn[sid]
                + turn_alpha * (inf.action - sym_prev_act[sid]).abs().min(2.0);
            sym_prev_act[sid] = inf.action;
        }
    }

    for ti in start_ti..end_ti {
        let t_sec = idx.all_ts[ti];
        let cur_day = est_trading_day_key(t_sec);
        let active_pairs = &idx.at_ts[ti];
        if active_pairs.is_empty() {
            continue;
        }

        let mut bar_at_t = [u32::MAX; 64];
        for &(sid, bidx) in active_pairs {
            if (sid as usize) < 64 {
                bar_at_t[sid as usize] = bidx;
                if is_intraday && sym_last_day[sid as usize] != cur_day {
                    hiddens[sid as usize].clear();
                    sym_ema_turn[sid as usize] *= 0.25;
                    sym_prev_act[sid as usize] = 0.0;
                    sym_last_day[sid as usize] = cur_day;
                }
            }
        }

        let mut newly_opened: [Uuid; 8] = [Uuid::nil(); 8];
        let mut newly_len = 0usize;
        for sig in pending_signals.drain(..) {
            let bidx = bar_at_t[sig.sym_id];
            if bidx == u32::MAX {
                continue;
            }
            let bi = bidx as usize;
            let entry_bar = &idx.bars_by_sym[sig.sym_id][bi];
            let entry_snap = &idx.snaps_by_sym[sig.sym_id][bi];
            let tid = Uuid::from_u128(0xB000_0000_0000_0000_u128 | trade_seq);
            if let Some(pos) = open_simulated_position_at_next_bar(
                sig.sym_id,
                &sig.symbol,
                sig.bucket,
                sig.brain_id,
                sig.signal_ts_sec,
                sig.signal_day_key,
                sig.direction,
                sig.target_action,
                sig.tail_stop_dist,
                sig.is_option_routed,
                sig.is_vertical_combo,
                sig.is_penny_tier,
                sig.dte_days,
                sig.kelly_risk_budget_usd,
                equity,
                entry_bar,
                entry_snap,
                sig.embedding_128,
                sig.confidence_pct,
                sig.active_gen_id,
                tid,
            ) {
                trade_seq += 1;
                if newly_len < newly_opened.len() {
                    newly_opened[newly_len] = tid;
                    newly_len += 1;
                }
                open_positions.push(pos);
            }
        }

        let mut i_pos = 0usize;
        while i_pos < open_positions.len() {
            let sid = open_positions[i_pos].sym_id;
            let bidx = bar_at_t[sid];
            if bidx == u32::MAX {
                i_pos += 1;
                continue;
            }
            let bi = bidx as usize;
            let bar_t = &idx.bars_by_sym[sid][bi];
            let snap_t = &idx.snaps_by_sym[sid][bi];
            let is_new = newly_opened[..newly_len].contains(&open_positions[i_pos].trade_id);
            let exit_iv = exit_iv_for_bar(bucket, snap_t);

            if let Some((exit_px, reason, is_full_stop)) =
                step_active_position_on_bar(&mut open_positions[i_pos], bar_t, exit_iv, is_new)
            {
                let pos = open_positions.swap_remove(i_pos);
                let (blotter, _) = finalize_settled_position(
                    &pos,
                    t_sec,
                    exit_px,
                    reason,
                    is_full_stop,
                    exit_iv,
                    ProofRunMode::RunBOnlineRl,
                );
                equity += blotter.net_pnl_usd;
                peak_eq = peak_eq.max(equity);
                max_dd_pct = max_dd_pct.max(((peak_eq - equity) / peak_eq.max(1.0)) * 100.0);
                sym_prev_act[pos.sym_id] = 0.0;
                settled.push(SettledCausalTrade {
                    sym_id: pos.sym_id,
                    symbol: pos.symbol,
                    entry_bar_ord: pos.entry_bar_ord as usize,
                    exit_bar_ord: bi,
                    exit_ts_sec: t_sec,
                    direction: pos.direction,
                    is_option_routed: pos.is_option_routed,
                    gross_pnl_usd: blotter.gross_pnl_usd,
                    net_pnl_usd: blotter.net_pnl_usd,
                    net_r: blotter.net_r,
                    exit_reason: blotter.exit_reason,
                    peak_v_liq_ret: pos.peak_v_liq_ret,
                });
            } else {
                if !is_new && snap_t.valid && bi < wm_cache[sid].len() {
                    let pos = &mut open_positions[i_pos];
                    let (ref x_n, ref z_wm, ref wm_pred) = wm_cache[sid][bi];
                    let norm_rem = normalized_session_remaining(bucket, snap_t.hhmm);
                    let (self_s, _) = build_active_position_self_awareness(
                        pos,
                        t_sec,
                        bar_t.close,
                        exit_iv,
                        norm_rem,
                        sym_ema_turn[sid],
                    );
                    let inf = brain.infer_ensemble_step_with_precomputed_wm(
                        x_n,
                        z_wm,
                        wm_pred,
                        self_s,
                        &hiddens[sid],
                        !pos.is_penny_tier && is_option_vehicle_allowed(&pos.symbol, bar_t.close),
                    );
                    hiddens[sid] = inf.next_hiddens;
                    sym_ema_turn[sid] = (1.0 - turn_alpha) * sym_ema_turn[sid]
                        + turn_alpha * (inf.action - sym_prev_act[sid]).abs().min(2.0);
                    sym_prev_act[sid] = inf.action;

                    let pos_sign = pos.direction.sign() as f32;
                    let exit_snap = (0.22 * pos.current_target_action.abs()).clamp(0.028, 0.075);
                    if inf.action.abs() < exit_snap || inf.action.signum() != pos_sign {
                        pos.pending_exit = true;
                    } else if (inf.action - pos.current_target_action).abs() >= 0.30 {
                        pos.pending_scale_action = Some(inf.action);
                    }
                }
                i_pos += 1;
            }
        }

        let (_, hhmm) = ny_et_hhmm(t_sec);
        if !entry_allowed_at_hhmm(bucket, hhmm)
            || open_positions.len() + pending_signals.len() >= max_slots
        {
            continue;
        }
        let norm_rem = normalized_session_remaining(bucket, hhmm);
        let mut candidates: Vec<(f64, PendingNextBarSignal)> = Vec::new();

        for &(sid_u16, bidx_u32) in active_pairs {
            let sid = sid_u16 as usize;
            let bi = bidx_u32 as usize;
            if bi >= wm_cache[sid].len() {
                continue;
            }
            let snap = &idx.snaps_by_sym[sid][bi];
            if !snap.valid {
                continue;
            }
            let bar_t = &idx.bars_by_sym[sid][bi];
            let sym = &idx.syms[sid];
            if !symbol_matches_bucket(sym, bar_t.close, bucket) {
                continue;
            }
            if open_positions.iter().any(|p| p.sym_id == sid)
                || pending_signals.iter().any(|s| s.sym_id == sid)
            {
                continue;
            }
            let is_penny = is_penny_symbol_cand(sym, bar_t.close);
            let opt_allowed = !is_penny && is_option_vehicle_allowed(sym, bar_t.close);
            let slip_bps = explicit_one_way_slippage_bps(sym, bar_t.close, is_penny) as f32;
            let (ref x_n, ref z_wm, ref wm_pred) = wm_cache[sid][bi];
            let self_s = SelfAwarenessState {
                current_pos: 0.0,
                unrealized_net_ret: 0.0,
                peak_excursion: 0.0,
                drawdown_from_peak: 0.0,
                log_bars_held: 0.0,
                one_way_cost_bps: slip_bps,
                session_time_left: norm_rem as f32,
                ema_turnover: sym_ema_turn[sid],
            };
            let inf = brain.infer_ensemble_step_with_precomputed_wm(
                x_n,
                z_wm,
                wm_pred,
                self_s,
                &hiddens[sid],
                opt_allowed,
            );
            hiddens[sid] = inf.next_hiddens;
            sym_ema_turn[sid] = (1.0 - turn_alpha) * sym_ema_turn[sid]
                + turn_alpha * (inf.action - sym_prev_act[sid]).abs().min(2.0);
            sym_prev_act[sid] = inf.action;

            if inf.action.abs() >= 0.070 {
                let dir = if inf.action > 0.0 {
                    TradeDirection::Long
                } else {
                    TradeDirection::Short
                };
                let (is_opt, is_vert, dte_days) = resolve_vehicle_routing_config(
                    sym,
                    bar_t.close,
                    bucket,
                    hhmm,
                    snap.vix_spot_dec,
                    inf.vehicle_prob >= 0.5,
                );
                let risk_budget_usd =
                    (equity * (0.0055 + 0.0065 * inf.action.abs() as f64)).clamp(420.0, 1_250.0);
                candidates.push((
                    inf.action.abs() as f64,
                    PendingNextBarSignal {
                        sym_id: sid,
                        symbol: sym.clone(),
                        bucket,
                        brain_id: bucket.brain_id(),
                        signal_ts_sec: t_sec,
                        signal_day_key: cur_day,
                        signal_bar_ord: snap.bar_ord,
                        direction: dir,
                        target_action: inf.action,
                        vehicle_prob: inf.vehicle_prob,
                        tail_stop_dist: inf.tail_stop_dist as f64,
                        confidence_pct: (55.0 + (inf.action.abs() as f64) * 40.0)
                            .clamp(55.0, 98.5),
                        is_option_routed: is_opt,
                        is_vertical_combo: is_vert,
                        is_penny_tier: is_penny,
                        dte_days,
                        kelly_risk_budget_usd: risk_budget_usd,
                        embedding_128: Vec::new(),
                        active_gen_id: brain.promoted_generation_id,
                    },
                ));
            }
        }
        candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let slots_left = max_slots.saturating_sub(open_positions.len() + pending_signals.len());
        for (_, sig) in candidates.into_iter().take(slots_left.min(2)) {
            pending_signals.push(sig);
        }
    }

    // Settle any open positions Mark-to-Liquidation at `eval_cutoff_ts` (`<= eval_cutoff_ts` causal bar!)
    // so multi-day Swing windows (`BRAIN-03`/`BRAIN-04`) never hide open drawdowns or leak across embargo boundaries.
    for pos in open_positions {
        let bars = idx.bars_by_sym[pos.sym_id];
        let snaps = idx.snaps_by_sym[pos.sym_id];
        let cut_ord = bars
            .partition_point(|b| b.timestamp_sec <= eval_cutoff_ts)
            .saturating_sub(1);
        if let Some(last_b) = bars.get(cut_ord) {
            let last_iv = snaps
                .get(cut_ord)
                .map(|s| exit_iv_for_bar(bucket, s))
                .unwrap_or(0.20);
            let (blotter, _) = finalize_settled_position(
                &pos,
                last_b.timestamp_sec,
                last_b.close,
                "EVAL_CUTOFF_MTM_LIQ".to_string(),
                false,
                last_iv,
                ProofRunMode::RunBOnlineRl,
            );
            equity += blotter.net_pnl_usd;
            peak_eq = peak_eq.max(equity);
            max_dd_pct = max_dd_pct.max(((peak_eq - equity) / peak_eq.max(1.0)) * 100.0);
            settled.push(SettledCausalTrade {
                sym_id: pos.sym_id,
                symbol: pos.symbol,
                entry_bar_ord: pos.entry_bar_ord as usize,
                exit_bar_ord: cut_ord,
                exit_ts_sec: last_b.timestamp_sec,
                direction: pos.direction,
                is_option_routed: pos.is_option_routed,
                gross_pnl_usd: blotter.gross_pnl_usd,
                net_pnl_usd: blotter.net_pnl_usd,
                net_r: blotter.net_r,
                exit_reason: blotter.exit_reason,
                peak_v_liq_ret: pos.peak_v_liq_ret,
            });
        }
    }

    summarize_settled_causal_trades(&settled, max_dd_pct)
}

fn summarize_settled_causal_trades(
    settled: &[SettledCausalTrade],
    max_dd_pct: f64,
) -> DiscreteCausalEvalSummary {
    let n = settled.len();
    if n == 0 {
        return DiscreteCausalEvalSummary {
            score: -1.5,
            ..Default::default()
        };
    }
    let mut win_pnl = 0.0_f64;
    let mut red_loss = 0.0_f64;
    let mut win_r_sum = 0.0_f64;
    let mut winners_count = 0usize;
    for tr in settled {
        if tr.net_pnl_usd > 0.0 {
            win_pnl += tr.net_pnl_usd;
            win_r_sum += tr.net_r.max(0.0);
            winners_count += 1;
        } else if tr.net_pnl_usd < 0.0 {
            red_loss += tr.net_pnl_usd.abs();
        }
    }
    let net_pnl = win_pnl - red_loss;
    let pf = if red_loss > 1e-6 {
        win_pnl / red_loss
    } else if win_pnl > 0.0 {
        12.0
    } else {
        0.0
    };
    let mean = net_pnl / (n as f64);
    let var = settled
        .iter()
        .map(|t| (t.net_pnl_usd - mean) * (t.net_pnl_usd - mean))
        .sum::<f64>()
        / (n.max(2) as f64 - 1.0).max(1.0);
    let sharpe = if var > 1e-6 {
        (mean / var.sqrt()) * (n as f64).sqrt()
    } else {
        0.0
    };
    let trade_bonus = if n < 6 {
        -1.6 + 0.22 * (n as f64)
    } else if pf >= 1.15 {
        (n as f64).min(24.0).sqrt() * 0.10
    } else {
        -0.04 * (n.saturating_sub(8) as f64)
    };
    let mean_r = settled.iter().map(|t| t.net_r).sum::<f64>() / (n as f64);
    let n_trust = (n as f64 / 15.0).min(1.0);
    let pf_adj = 1.0 + (pf.min(4.0) - 1.0) * n_trust;
    let score = (pf_adj - 1.0) * 1.8
        + sharpe * 0.35
        + (net_pnl / 1200.0).clamp(-3.0, 3.5)
        - max_dd_pct * 0.05
        + mean_r.clamp(-1.5, 2.0) * 0.90
        + trade_bonus;

    DiscreteCausalEvalSummary {
        score,
        sharpe,
        net_pnl_usd: net_pnl,
        net_pf: pf,
        red_loss_usd: red_loss,
        win_r_sum,
        winners_count,
        max_dd_pct,
        trades: settled.to_vec(),
    }
}

/// Evaluates all `M = 24` population members in parallel via `rayon` on `[eval_start_ts ..= eval_cutoff_ts]`.
fn evaluate_population_indexed_parallel(
    brain: &mut PureAutonomousRlBrain,
    idx: &BucketTimelineIndex<'_>,
    wm_cache: &[WmSymbolCache],
    eval_start_ts: i64,
    eval_cutoff_ts: i64,
    starting_equity: f64,
) -> DiscreteCausalEvalSummary {
    let summaries: Vec<DiscreteCausalEvalSummary> = brain
        .population
        .par_iter_mut()
        .map(|member| {
            evaluate_single_member_indexed(
                member,
                idx,
                wm_cache,
                eval_start_ts,
                eval_cutoff_ts,
                starting_equity,
            )
        })
        .collect();

    brain.update_hedge_weights();
    summaries
        .into_iter()
        .max_by(|a, b| a.score.total_cmp(&b.score))
        .unwrap_or_default()
}

/// Extracts the strictly causal trajectory slice (`<= exit_ts_sec <= cutoff_ts`) around a settled trade
/// for Step 1 Counterfactual & Saliency Loss Audit or Winner Fisher EWC computation.
fn extract_settled_trade_causal_bars(
    tr: &SettledCausalTrade,
    idx: &BucketTimelineIndex<'_>,
    cutoff_ts: i64,
) -> Option<(usize, Vec<CausalTrajectoryBar>)> {
    let bars = idx.bars_by_sym[tr.sym_id];
    let snaps = idx.snaps_by_sym[tr.sym_id];
    let start_b = tr.entry_bar_ord.saturating_sub(8);
    let end_b = tr.exit_bar_ord.min(bars.len().saturating_sub(2));
    if end_b <= start_b + 2 {
        return None;
    }
    let mut out = Vec::with_capacity(end_b - start_b + 1);
    let mut entry_off = 0usize;
    for b_i in start_b..=end_b {
        if bars[b_i].timestamp_sec > cutoff_ts {
            break;
        }
        if b_i == tr.entry_bar_ord {
            entry_off = out.len();
        }
        let is_last = b_i == end_b;
        if let Some(cbar) =
            build_single_causal_bar(&tr.symbol, idx.bucket, bars, snaps, b_i, cutoff_ts, is_last)
        {
            out.push(cbar);
        }
    }
    if out.len() >= 3 {
        Some((entry_off.min(out.len() - 1), out))
    } else {
        None
    }
}

fn compute_recency_weighted_causal_score(
    ev: &DiscreteCausalEvalSummary,
    cutoff_ts: i64,
    is_intraday: bool,
) -> f64 {
    if ev.trades.is_empty() {
        return -1e6;
    }
    let tau_sec = if is_intraday {
        7.0 * 86_400.0
    } else {
        18.0 * 86_400.0
    };
    let mut w_sum = 0.0_f64;
    let mut w_win = 0.0_f64;
    let mut w_loss = 0.0_f64;
    let mut w_net = 0.0_f64;
    let mut w_win_r = 0.0_f64;
    for tr in &ev.trades {
        let age = (cutoff_ts - tr.exit_ts_sec).max(0) as f64;
        let w = (-age / tau_sec).exp().max(0.05);
        w_sum += w;
        w_net += w * tr.net_pnl_usd;
        if tr.net_pnl_usd > 0.0 {
            w_win += w * tr.net_pnl_usd;
            w_win_r += w * tr.net_r.max(0.0);
        } else if tr.net_pnl_usd < 0.0 {
            w_loss += w * tr.net_pnl_usd.abs();
        }
    }
    let w_sum = w_sum.max(1e-6);
    let w_mean = w_net / w_sum;
    let mut w_var = 0.0_f64;
    for tr in &ev.trades {
        let age = (cutoff_ts - tr.exit_ts_sec).max(0) as f64;
        let w = (-age / tau_sec).exp().max(0.05);
        let d = tr.net_pnl_usd - w_mean;
        w_var += w * d * d;
    }
    let rw_sharpe = if w_var > 1e-6 {
        (w_mean / (w_var / w_sum).sqrt()) * (ev.trades.len() as f64).sqrt()
    } else {
        0.0
    };
    let rw_pf = if w_loss > 1e-6 {
        (w_win / w_loss).min(5.0)
    } else if w_win > 0.0 {
        5.0
    } else {
        0.0
    };
    let avg_scale = (ev.trades.len() as f64) / w_sum;
    let rw_net_scaled = w_net * avg_scale;
    let rw_loss_scaled = w_loss * avg_scale;
    let n_tr = ev.trades.len() as f64;
    let participation = if rw_pf >= 1.18 {
        8.0 * n_tr.min(16.0)
    } else if rw_pf < 1.0 {
        -12.0 * n_tr.min(25.0)
    } else {
        0.0
    };
    rw_net_scaled
        + 180.0 * (rw_pf - 1.0)
        + 60.0 * rw_sharpe.clamp(-3.0, 5.0)
        - 0.15 * rw_loss_scaled
        - 25.0 * ev.max_dd_pct
        + 20.0 * (w_win_r * avg_scale).min(20.0)
        + participation
}

/// Unified Purged-Holdout Post-Cost Utility Lower-Confidence-Bound Objective (`U_lcb`):
/// `U_lcb(pi) = Net_PnL - lambda_dd * MaxDD_pct - z_alpha * SE_semi(Net_Returns)`
/// where `0` trades yields exact `U_lcb = 0.0`, and `SE_semi` measures downside semi-deviation + mild cross-sectional
/// dispersion (`max(SE_trade_semi, SE_daily_semi)`) so right-tail convex winners are NEVER punished as risk while
/// correlated same-day multi-symbol bets and downside volatility are strictly penalized.
fn compute_utility_lcb(ev: &DiscreteCausalEvalSummary) -> f64 {
    if ev.trades.is_empty() {
        return 0.0;
    }
    let n = ev.trades.len();
    let total_pnl = ev.net_pnl_usd;
    let se_trade_total = if n < 2 {
        let down = (-total_pnl).max(0.0);
        down + 0.30 * total_pnl.abs().max(35.0)
    } else {
        let mu = total_pnl / (n as f64);
        let semi_var = ev
            .trades
            .iter()
            .map(|t| {
                let r = t.net_pnl_usd;
                let neg = r.min(0.0);
                let d = r - mu;
                neg * neg + 0.15 * d * d
            })
            .sum::<f64>()
            / ((n - 1) as f64);
        (n as f64).sqrt() * semi_var.max(0.0).sqrt()
    };

    let mut daily_pnl: BTreeMap<i64, f64> = BTreeMap::new();
    for tr in &ev.trades {
        let dk = est_trading_day_key(tr.exit_ts_sec);
        *daily_pnl.entry(dk).or_insert(0.0) += tr.net_pnl_usd;
    }
    let k = daily_pnl.len();
    let se_day_total = if k < 2 {
        let down = (-total_pnl).max(0.0);
        down + 0.30 * total_pnl.abs().max(35.0)
    } else {
        let kf = k as f64;
        let mu_d = total_pnl / kf;
        let semi_var_d = daily_pnl
            .values()
            .map(|&p| {
                let neg = p.min(0.0);
                let d = p - mu_d;
                neg * neg + 0.15 * d * d
            })
            .sum::<f64>()
            / ((k - 1) as f64);
        kf.sqrt() * semi_var_d.max(0.0).sqrt()
    };

    let se_eff = se_trade_total.max(se_day_total);
    let lambda_dd = 24.0_f64;
    let z_alpha = 0.48_f64;
    total_pnl - lambda_dd * ev.max_dd_pct - z_alpha * se_eff
}

#[inline]
fn eval_expectancy_metrics(ev: &DiscreteCausalEvalSummary) -> (f64, f64, f64) {
    if ev.trades.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let total_net_r: f64 = ev.trades.iter().map(|t| t.net_r).sum();
    let mean_win_r = if ev.winners_count > 0 {
        ev.win_r_sum / (ev.winners_count as f64)
    } else {
        0.0
    };
    let net_exp_usd = ev.net_pnl_usd / (ev.trades.len() as f64);
    (total_net_r, mean_win_r, net_exp_usd)
}

/// Unified Purged `T_holdout` Post-Cost `U_lcb` Promotion Gate (Zero-Baseline & Friction-Pruning Safe):
/// - `0`-trade incumbent has `U_lcb = 0.0`: promotes any candidate with `U_lcb > 0.0` and `Net PF > 1.0`.
/// - Underwater incumbent (`U_lcb <= 0.0` or `Net PF <= 1.0`): promotes candidates raising `U_lcb` that
///   achieve `Net PF > 1.0 && U_lcb > 0.0`, OR strictly prune losing churn (`trades < inc && red_loss <= 0.80 * inc && PF >= 0.94`)
///   — NEVER promoting sub-1.0 PF trade-bloat expansions.
/// - Profitable incumbent: promotes candidates strictly raising `U_lcb` (`> incumbent + eps`) with `Net PF > 1.0`
///   and preserved Net Expectancy / Total Net R / Mean Winner R (NEVER raw integer win/trade count floors).
fn is_purged_holdout_lcb_promotion(
    champ_holdout: &DiscreteCausalEvalSummary,
    cand_holdout: &DiscreteCausalEvalSummary,
    _is_paralyzed: bool,
) -> bool {
    if cand_holdout.trades.is_empty() {
        return false;
    }
    let dd_ok = cand_holdout.max_dd_pct <= champ_holdout.max_dd_pct.max(3.5) + 0.85
        && cand_holdout.max_dd_pct <= 7.5;
    if !dd_ok {
        return false;
    }

    let u_inc = compute_utility_lcb(champ_holdout);
    let u_cand = compute_utility_lcb(cand_holdout);

    // 1. Dormant `0`-trade incumbent (`U_lcb == 0.0`): promote any net-profitable wakeup (`U_lcb > 0`, `Net PF > 1.0`)
    if champ_holdout.trades.is_empty() {
        return u_cand > 0.0 && cand_holdout.net_pf > 1.0 && cand_holdout.net_pnl_usd > 0.0;
    }

    // 2. Underwater incumbent (`U_lcb <= 0.0` or `Net PF <= 1.0` or `Net PnL <= 0.0`):
    if u_inc <= 0.0 || champ_holdout.net_pf <= 1.0 || champ_holdout.net_pnl_usd <= 0.0 {
        let min_flip_pf = if cand_holdout.trades.len() > champ_holdout.trades.len() {
            1.12
        } else {
            1.02
        };
        let flips_profitable = u_cand > 0.0
            && cand_holdout.net_pnl_usd > 0.0
            && cand_holdout.net_pf > min_flip_pf;
        let strict_loss_pruning = u_cand > u_inc + 1e-6
            && cand_holdout.trades.len() < champ_holdout.trades.len()
            && cand_holdout.net_pf >= 0.96
            && cand_holdout.net_pf > champ_holdout.net_pf + 0.05
            && cand_holdout.red_loss_usd <= champ_holdout.red_loss_usd * 0.80;
        return flips_profitable || strict_loss_pruning;
    }

    // 3. Profitable incumbent (`U_lcb > 0`, `Net PF > 1.0`):
    //    Promote strictly via `U_lcb(cand) > U_lcb(inc) + eps` + `Net PF > 1.05` + Expectancy Preservation
    //    (ZERO raw gross-loss veto, ZERO integer win-count or trade-count floor veto!)
    if u_cand <= u_inc + 1e-6 || cand_holdout.net_pf <= 1.05 || cand_holdout.net_pnl_usd <= 0.0 {
        return false;
    }
    let (inc_net_r, inc_mean_wr, inc_exp) = eval_expectancy_metrics(champ_holdout);
    let (cand_net_r, cand_mean_wr, cand_exp) = eval_expectancy_metrics(cand_holdout);
    let expectancy_preserved = cand_net_r >= inc_net_r - 1e-6
        || (cand_exp >= inc_exp - 1e-6 && cand_mean_wr >= inc_mean_wr * 0.90 - 1e-6);
    let no_dilutive_bloat = if cand_holdout.trades.len() > champ_holdout.trades.len() + 2
        && cand_exp < inc_exp * 0.85
    {
        cand_holdout.net_pf >= champ_holdout.net_pf.min(1.60)
    } else {
        true
    };
    expectancy_preserved && no_dilutive_bloat
}

fn is_causal_promotion(
    champ_eval: &DiscreteCausalEvalSummary,
    cand_eval: &DiscreteCausalEvalSummary,
    _cutoff_ts: i64,
    _is_intraday: bool,
) -> bool {
    if cand_eval.trades.len() < 2 {
        return false;
    }
    let dd_safe = cand_eval.max_dd_pct <= champ_eval.max_dd_pct.max(4.0) + 0.85
        && cand_eval.max_dd_pct <= 8.5;
    if !dd_safe {
        return false;
    }

    let u_inc = compute_utility_lcb(champ_eval);
    let u_cand = compute_utility_lcb(cand_eval);

    // 1. Dormant `0`-trade incumbent (`U_lcb == 0.0`): wake up into a genuinely net-profitable policy!
    if champ_eval.trades.is_empty() {
        return u_cand > 0.0 && cand_eval.net_pf >= 1.12 && cand_eval.net_pnl_usd > 0.0;
    }

    // 2. Underwater incumbent (`U_lcb <= 0.0` or `Net PF <= 1.0` or `Net PnL <= 0.0`):
    if u_inc <= 0.0 || champ_eval.net_pf <= 1.0 || champ_eval.net_pnl_usd <= 0.0 {
        let min_flip_pf = if cand_eval.trades.len() > champ_eval.trades.len() {
            1.15
        } else {
            1.05
        };
        let flips_profitable =
            u_cand > 0.0 && cand_eval.net_pnl_usd > 0.0 && cand_eval.net_pf >= min_flip_pf;
        let strict_loss_pruning = u_cand > u_inc + 1e-6
            && cand_eval.trades.len() < champ_eval.trades.len()
            && cand_eval.net_pf >= 0.96
            && cand_eval.net_pf > champ_eval.net_pf + 0.05
            && cand_eval.red_loss_usd <= champ_eval.red_loss_usd * 0.80;
        return flips_profitable || strict_loss_pruning;
    }

    // 3. Profitable incumbent (`U_lcb > 0.0`, `Net PF > 1.0`):
    //    Unified `U_lcb` improvement + Winner Expectancy Preservation (ZERO `red_loss_shrunk` or `winners_count` vetoes!)
    if u_cand <= u_inc + 1e-6 || cand_eval.net_pf < 1.10 || cand_eval.net_pnl_usd <= 0.0 {
        return false;
    }
    let (inc_net_r, inc_mean_wr, inc_exp) = eval_expectancy_metrics(champ_eval);
    let (cand_net_r, cand_mean_wr, cand_exp) = eval_expectancy_metrics(cand_eval);
    let expectancy_preserved = cand_net_r >= inc_net_r - 1e-6
        || (cand_exp >= inc_exp - 1e-6 && cand_mean_wr >= inc_mean_wr * 0.90 - 1e-6);
    let no_dilutive_bloat =
        if cand_eval.trades.len() > champ_eval.trades.len() + 3 && cand_exp < inc_exp * 0.85 {
            cand_eval.net_pf >= champ_eval.net_pf.min(1.65)
        } else {
            cand_eval.net_pf >= champ_eval.net_pf.min(1.35) - 0.02
        };
    expectancy_preserved && no_dilutive_bloat
}

fn restore_champion_weights_keeping_audit_memory(
    brain: &mut PureAutonomousRlBrain,
    backup: PureAutonomousRlBrain,
) {
    let keep_red = std::mem::take(&mut brain.red_replay_buffer);
    let keep_win = std::mem::take(&mut brain.winner_fisher_buffer);
    let keep_tel = brain.autopsy_telemetry.clone();
    let keep_upd = brain
        .online_updates_count
        .max(backup.online_updates_count.saturating_add(1));
    *brain = backup;
    brain.red_replay_buffer = keep_red;
    brain.winner_fisher_buffer = keep_win;
    brain.autopsy_telemetry = keep_tel;
    brain.autopsy_telemetry.cost_sensitivity_after = brain
        .population
        .iter()
        .map(|m| m.hparams.cost_sensitivity)
        .sum::<f32>()
        / (brain.population.len().max(1) as f32);
    brain.online_updates_count = keep_upd;
}

/// PART 2: Mandatory 4-Step Closed-Loop Red-Trade Loss Audit -> RL Update -> Full Causal Re-Test -> Promotion!
#[allow(clippy::too_many_arguments)]
fn execute_closed_loop_red_trade_autopsy_and_promotion(
    brain: &mut PureAutonomousRlBrain,
    idx: &BucketTimelineIndex<'_>,
    causal_episodes: &[CausalSessionEpisode],
    retest_start_ts: i64,
    cutoff_ts: i64,
    starting_equity: f64,
    phase_label: &str,
    ledger_out: &mut Vec<GenerationPromotionLedgerRow>,
) -> bool {
    let is_intraday = idx.bucket.is_intraday();
    let wm_cache = idx.precompute_wm_cache(
        &brain.world_model,
        &brain.feat_mean,
        &brain.feat_var,
        cutoff_ts,
    );
    let champ_eval = evaluate_brain_ensemble_retest_indexed(
        brain,
        idx,
        &wm_cache,
        retest_start_ts,
        cutoff_ts,
        starting_equity,
    );
    let immutable_champ = brain.clone();

    // Step 1: Automated 4-Vector Counterfactual & Gradient Saliency Loss Audit across settled trades `<= cutoff_ts`
    // (Accumulate deduplicated buffers rather than wiping historical lessons, including >55% EOD-flatten decays!)
    let mut red_audited = 0usize;
    let mut entry_faults = 0usize;
    let mut exit_faults = 0usize;
    let mut new_vehicle_regrets = 0usize;
    let mut sum_sal_dx = 0.0_f32;
    let mut sum_sal_dh = 0.0_f32;

    for tr in &champ_eval.trades {
        let eod_decay_giveback = tr.exit_reason.contains("EOD_")
            && tr.peak_v_liq_ret > 0.005
            && tr.net_r < 0.45 * (tr.peak_v_liq_ret / 0.012);
        if let Some((entry_off, bar_slice)) = extract_settled_trade_causal_bars(tr, idx, cutoff_ts)
        {
            if tr.net_pnl_usd < 0.0 || eod_decay_giveback {
                if let Some(existing) = brain
                    .red_replay_buffer
                    .iter()
                    .find(|r| r.symbol == tr.symbol && r.exit_ts_sec == tr.exit_ts_sec)
                {
                    red_audited += 1;
                    match existing.fault_kind {
                        LossFaultKind::FrictionInversion
                        | LossFaultKind::VehicleSelectionRegret => {
                            entry_faults += 1;
                        }
                        LossFaultKind::HoldingEodDecay
                        | LossFaultKind::DeadzoneParalysisRegret => {
                            exit_faults += 1;
                        }
                    }
                    sum_sal_dx += existing.saliency_dx;
                    sum_sal_dh += existing.saliency_dh;
                } else {
                    let audit = brain.audit_and_insert_loss_event(
                        tr.symbol.clone(),
                        tr.exit_ts_sec,
                        tr.net_pnl_usd,
                        tr.gross_pnl_usd,
                        tr.net_r as f32,
                        tr.direction.sign() as f32,
                        tr.is_option_routed,
                        entry_off,
                        [0.0; WM_DIM],
                        bar_slice,
                    );
                    red_audited += 1;
                    if audit.fault_kind == LossFaultKind::VehicleSelectionRegret {
                        new_vehicle_regrets += 1;
                    }
                    match audit.fault_kind {
                        LossFaultKind::FrictionInversion
                        | LossFaultKind::VehicleSelectionRegret => {
                            entry_faults += 1;
                        }
                        LossFaultKind::HoldingEodDecay
                        | LossFaultKind::DeadzoneParalysisRegret => {
                            exit_faults += 1;
                        }
                    }
                    sum_sal_dx += audit.saliency_dx;
                    sum_sal_dh += audit.saliency_dh;
                }
            } else if tr.net_pnl_usd > 0.0 {
                let dup = brain.winner_fisher_buffer.iter().any(|w| {
                    w.exit_ts_sec == tr.exit_ts_sec && w.entry_offset == entry_off
                });
                if !dup {
                    brain.insert_winner_trajectory(SettledWinnerTrajectory {
                        exit_ts_sec: tr.exit_ts_sec,
                        net_r: tr.net_r as f32,
                        direction_sign: tr.direction.sign() as f32,
                        entry_offset: entry_off,
                        prior_regime_z: [0.0; WM_DIM],
                        bars: bar_slice,
                    });
                }
            }
        }
    }

    let is_paralyzed = champ_eval.trades.len() < 3;
    if causal_episodes.is_empty() && brain.red_replay_buffer.is_empty() {
        return false;
    }

    // Purged Chronological Split (`T_train` -> `T_embargo` -> `T_holdout`) with strict causal embargo
    // `Delta_t_embargo >= mean_episode_horizon` so multi-day Swing holds NEVER overlap `T_train` and `T_holdout`!
    let span_sec = (cutoff_ts - retest_start_ts).max(0);
    let mut hold_secs_sum = 0i64;
    let mut hold_cnt = 0usize;
    for tr in &champ_eval.trades {
        if let Some(eb) = idx
            .bars_by_sym
            .get(tr.sym_id)
            .and_then(|bs| bs.get(tr.entry_bar_ord))
        {
            hold_secs_sum += (tr.exit_ts_sec - eb.timestamp_sec).max(0);
            hold_cnt += 1;
        }
    }
    let default_horizon_sec = if is_intraday { 7_200 } else { 4 * 86_400 };
    let empirical_mean_hold_sec = if hold_cnt > 0 {
        (hold_secs_sum / (hold_cnt as i64)).max(if is_intraday { 3_600 } else { 2 * 86_400 })
    } else {
        default_horizon_sec
    };
    let min_embargo_sec = if is_intraday { 14_400 } else { 3 * 86_400 };
    let (train_split_ts, holdout_start_ts) = if span_sec > 14_400 {
        let t_split = retest_start_ts + (span_sec * 54) / 100;
        let max_emb = (span_sec * 16) / 100;
        let emb = empirical_mean_hold_sec.max(min_embargo_sec).min(max_emb);
        let h_start = (t_split + emb).min(cutoff_ts - (span_sec * 28) / 100);
        (t_split, h_start.max(t_split))
    } else {
        (cutoff_ts, cutoff_ts)
    };
    let train_oof_start_ts = retest_start_ts + ((train_split_ts - retest_start_ts).max(0) * 52) / 100;

    // Active Counterfactual Paralysis Regret Audit (`<= train_split_ts`) so 0-trade / dormant periods
    // compute exact analytical `(|dL/dx|, |dL/dh|)` saliency and record `Paralysis Regret` telemetry!
    let mut paralysis_saliency_n = 0usize;
    if is_paralyzed || red_audited == 0 {
        let (par_ev, par_dx, par_dh) =
            brain.audit_and_train_paralysis_regret(causal_episodes, train_split_ts, 0, 0.0);
        if par_ev > 0 && red_audited == 0 {
            sum_sal_dx += par_dx;
            sum_sal_dh += par_dh;
            paralysis_saliency_n = 1;
        }
    }

    let full_reds = brain.red_replay_buffer.clone();
    let full_wins = brain.winner_fisher_buffer.clone();
    let mut audited_tel = brain.autopsy_telemetry.clone();

    let mut train_template = brain.clone();
    if train_split_ts < cutoff_ts {
        train_template
            .red_replay_buffer
            .retain(|r| r.exit_ts_sec <= train_split_ts);
        train_template
            .winner_fisher_buffer
            .retain(|w| w.exit_ts_sec <= train_split_ts);
    }
    let train_reds_f = train_template
        .red_replay_buffer
        .iter()
        .filter(|r| r.fault_kind == LossFaultKind::FrictionInversion)
        .count();
    let train_reds_h = train_template
        .red_replay_buffer
        .iter()
        .filter(|r| r.fault_kind == LossFaultKind::HoldingEodDecay)
        .count();
    let train_settled_n = champ_eval
        .trades
        .iter()
        .filter(|t| t.exit_ts_sec <= train_split_ts)
        .count()
        .max(train_reds_f + train_reds_h);
    // Unconstrained log-space `log_cost_sens` adaptation driven strictly by `<= train_split_ts` friction prevalence
    train_template.adapt_cost_sensitivity_from_autopsy_cycle(
        train_reds_f,
        train_reds_h,
        train_settled_n,
    );
    train_template.compute_fisher_ewc_from_winners_le(train_split_ts);

    let recent_vehicle_regrets = train_template
        .red_replay_buffer
        .iter()
        .filter(|r| r.fault_kind == LossFaultKind::VehicleSelectionRegret)
        .count();
    let has_vehicle_regret = new_vehicle_regrets >= 2 || recent_vehicle_regrets >= 3;

    let mut train_sub_episodes: Vec<CausalSessionEpisode> = Vec::with_capacity(causal_episodes.len());
    for ep in causal_episodes {
        let sb: Vec<CausalTrajectoryBar> = ep
            .bars
            .iter()
            .filter(|b| b.ts_sec <= train_split_ts)
            .cloned()
            .collect();
        if sb.len() >= 4 {
            train_sub_episodes.push(CausalSessionEpisode {
                symbol: ep.symbol.clone(),
                day_key: ep.day_key,
                start_ts_sec: sb[0].ts_sec,
                end_ts_sec: sb[sb.len() - 1].ts_sec,
                is_intraday_session: ep.is_intraday_session,
                prior_regime_z: ep.prior_regime_z,
                bars: sb,
            });
        }
    }
    let train_eps_ref: &[CausalSessionEpisode] = if !train_sub_episodes.is_empty() {
        &train_sub_episodes
    } else {
        causal_episodes
    };

    // Causal Adaptive Conformal Inference (`q_conformal`) strictly on `<= train_split_ts` bars & trades!
    let train_trades_slice: Vec<&SettledCausalTrade> = champ_eval
        .trades
        .iter()
        .filter(|t| t.exit_ts_sec <= train_split_ts)
        .collect();
    let (tr_gross_win, tr_gross_loss) =
        train_trades_slice
            .iter()
            .fold((0.0_f64, 0.0_f64), |(w, l), t| {
                if t.net_pnl_usd > 0.0 {
                    (w + t.net_pnl_usd, l)
                } else {
                    (w, l + (-t.net_pnl_usd))
                }
            });
    let train_emp_pf = if tr_gross_loss > 1e-6 {
        (tr_gross_win / tr_gross_loss).clamp(0.15, 5.0)
    } else if tr_gross_win > 0.0 {
        2.5
    } else {
        0.85
    };
    let _ = train_template.calibrate_causal_conformal_residual(
        train_eps_ref,
        train_split_ts,
        train_emp_pf,
        train_trades_slice.len(),
    );

    let champ_val_eval = if holdout_start_ts < cutoff_ts {
        Some(evaluate_brain_ensemble_retest_indexed(
            &immutable_champ,
            idx,
            &wm_cache,
            holdout_start_ts,
            cutoff_ts,
            starting_equity,
        ))
    } else {
        None
    };

    let max_oof_span = match idx.bucket {
        ProofBucket::Scalp0Dte1m => 7 * 86_400,
        ProofBucket::Scalp0Dte5m
        | ProofBucket::Brain01PennyDay
        | ProofBucket::Brain02LargeCapDay => 12 * 86_400,
        ProofBucket::Brain03ShortSwing | ProofBucket::Brain04CoreSwing => 24 * 86_400,
    };
    let min_oof_span = if is_intraday { 2 * 86_400 } else { 10 * 86_400 };
    let score_member_train_window = |cand: &mut PureAutonomousRlBrain, wm: &[WmSymbolCache]| {
        // Strictly score `M = 24` specialists ONCE on `<= train_split_ts` so `T_embargo` and `T_holdout`
        // remain 100% blind to both weight updates AND `hedge_weight` routing selection!
        let mut win_start = if train_split_ts - train_oof_start_ts >= min_oof_span {
            train_oof_start_ts
        } else {
            retest_start_ts
        };
        let capped_start = (train_split_ts - max_oof_span).max(retest_start_ts);
        if win_start < capped_start {
            win_start = capped_start;
        }
        let _ = evaluate_population_indexed_parallel(
            cand,
            idx,
            wm,
            win_start,
            train_split_ts,
            starting_equity,
        );
    };

    let eval_cand_gate = |cand: &PureAutonomousRlBrain, wm: &[WmSymbolCache]| {
        let full_ev = evaluate_brain_ensemble_retest_indexed(
            cand,
            idx,
            wm,
            retest_start_ts,
            cutoff_ts,
            starting_equity,
        );
        let full_rank = 0.65 * compute_utility_lcb(&full_ev)
            + 0.35 * compute_recency_weighted_causal_score(&full_ev, cutoff_ts, is_intraday);
        let full_ok = is_causal_promotion(&champ_eval, &full_ev, cutoff_ts, is_intraday);
        if !full_ok {
            return (full_ev, full_rank, false);
        }
        if let Some(ref cv) = champ_val_eval {
            let ve = evaluate_brain_ensemble_retest_indexed(
                cand,
                idx,
                wm,
                holdout_start_ts,
                cutoff_ts,
                starting_equity,
            );
            let val_ok = if cv.trades.is_empty() && ve.trades.is_empty() {
                full_ev.net_pf > 1.18 && compute_utility_lcb(&full_ev) > 0.0
            } else {
                is_purged_holdout_lcb_promotion(cv, &ve, is_paralyzed)
            };
            let ve_ulcb = compute_utility_lcb(&ve);
            let val_rank = 0.70 * ve_ulcb
                + 0.30 * compute_recency_weighted_causal_score(&ve, cutoff_ts, is_intraday);
            let comb_rank = 0.72 * val_rank + 0.28 * full_rank;
            (full_ev, comb_rank, val_ok)
        } else {
            (full_ev, full_rank, true)
        }
    };

    // Candidates 1..=6: Parallel `rayon` Multiplicative-Weights + Cost-Scale Mutation Sweep across `M = 24` specialists (`<= train_split_ts`)
    let mut scored_base = immutable_champ.clone();
    scored_base.conformal_q_bps = train_template.conformal_q_bps;
    for m in &mut scored_base.population {
        m.hparams.conformal_q_bps = train_template.conformal_q_bps;
    }
    score_member_train_window(&mut scored_base, &wm_cache);
    let routing_specs: [(f64, f32); 6] = [
        (2.4, 1.00),
        (6.5, 1.00),
        (14.0, 1.00),
        (5.5, 1.08),
        (9.5, 1.16),
        (4.5, 0.93),
    ];
    let mut cand_results: Vec<(PureAutonomousRlBrain, DiscreteCausalEvalSummary, f64, bool)> =
        routing_specs
            .par_iter()
            .map(|&(eta, cs_mult)| {
                let mut cand = scored_base.clone();
                if (cs_mult - 1.0).abs() > 1e-4 {
                    for m in &mut cand.population {
                        let new_log = (m.hparams.cost_sensitivity.max(0.05) * cs_mult).ln();
                        m.hparams.set_log_cost_sens(new_log);
                    }
                }
                cand.update_hedge_weights_with_eta(eta);
                let (full_ev, rank_score, passes) = eval_cand_gate(&cand, &wm_cache);
                (cand, full_ev, rank_score, passes)
            })
            .collect();

    // Candidates 7..=9: Trust-region RL + Fisher EWC candidates trained strictly on `<= train_split_ts`
    let mut rl_specs: Vec<(usize, f32, f32, f64)> = vec![
        (
            4,
            0.075,
            if is_paralyzed {
                0.08
            } else if champ_eval.net_pf < 1.15 {
                0.45
            } else {
                1.35
            },
            4.2,
        ),
        (
            3,
            0.050,
            if is_paralyzed { 0.12 } else { 1.60 },
            8.0,
        ),
    ];
    if champ_eval.net_pf < 1.15 || is_paralyzed {
        rl_specs.push((
            5,
            0.105,
            if is_paralyzed { 0.05 } else { 0.40 },
            6.5,
        ));
    }
    let rl_train_end_ts = if train_split_ts < cutoff_ts {
        train_split_ts
    } else {
        cutoff_ts
    };

    let mut max_flat_replayed = audited_tel.flat_regret_windows_replayed;
    let mut rl_scored: Vec<(PureAutonomousRlBrain, DiscreteCausalEvalSummary, f64, bool, f64)> =
        Vec::with_capacity(rl_specs.len());
    for &(steps, lr_sc, ewc_lam, eta) in &rl_specs {
        let mut cand = train_template.clone();
        cand.train_red_autopsy_and_causal_batch_parallel_le(
            train_eps_ref,
            train_split_ts,
            steps,
            lr_sc,
            ewc_lam,
        );
        if is_intraday && (exit_faults > 0 || champ_eval.net_pf < 1.15) {
            let _ = cand.train_intraday_exit_timing_episodes(
                train_eps_ref,
                2,
                (lr_sc * 2.5).clamp(0.18, 0.45),
            );
        }
        max_flat_replayed =
            max_flat_replayed.max(cand.autopsy_telemetry.flat_regret_windows_replayed);
        if champ_eval.trades.len() < 4 {
            let _ = cand.audit_and_train_paralysis_regret(
                train_eps_ref,
                train_split_ts,
                (steps / 2).max(2),
                lr_sc.max(0.38),
            );
        }
        if steps >= 3 && (is_paralyzed || (red_audited > 0 && has_vehicle_regret)) {
            let _ = cand.train_vehicle_regret_episodes_ext(
                train_eps_ref,
                2,
                (lr_sc * 0.8).max(0.20),
                false,
            );
        }
        score_member_train_window(&mut cand, &wm_cache);
        cand.update_hedge_weights_with_eta(eta);
        let train_ev = evaluate_brain_ensemble_retest_indexed(
            &cand,
            idx,
            &wm_cache,
            retest_start_ts,
            rl_train_end_ts,
            starting_equity,
        );
        let train_rank = 0.65 * compute_utility_lcb(&train_ev)
            + 0.35 * compute_recency_weighted_causal_score(&train_ev, rl_train_end_ts, is_intraday);
        let (full_ev, rank_score, passes) = eval_cand_gate(&cand, &wm_cache);
        rl_scored.push((cand, full_ev, rank_score, passes, train_rank));
    }

    // Causal World-Model TTT + EWC Loss-Autopsy candidate trained strictly `<= train_split_ts` (when paralyzed or net_pf < 1.05)
    if is_paralyzed || champ_eval.net_pf < 1.05 {
        let mut ttt_cand = train_template.clone();
        let _ = ttt_cand.train_world_model_episodes(train_eps_ref, 1, 0.0008);
        let _ = ttt_cand.calibrate_causal_conformal_residual(
            train_eps_ref,
            train_split_ts,
            train_emp_pf,
            train_trades_slice.len(),
        );
        ttt_cand.train_red_autopsy_and_causal_batch_parallel_le(
            train_eps_ref,
            train_split_ts,
            2,
            0.055,
            if champ_eval.net_pf < 1.05 { 0.20 } else { 1.30 },
        );
        max_flat_replayed =
            max_flat_replayed.max(ttt_cand.autopsy_telemetry.flat_regret_windows_replayed);
        let ttt_wm = idx.precompute_wm_cache(
            &ttt_cand.world_model,
            &ttt_cand.feat_mean,
            &ttt_cand.feat_var,
            cutoff_ts,
        );
        score_member_train_window(&mut ttt_cand, &ttt_wm);
        ttt_cand.update_hedge_weights_with_eta(3.6);
        let (ttt_full_ev, ttt_rank_score, ttt_passes) = eval_cand_gate(&ttt_cand, &ttt_wm);
        cand_results.push((ttt_cand, ttt_full_ev, ttt_rank_score, ttt_passes));
    }
    audited_tel.flat_regret_windows_replayed = max_flat_replayed;
    if audited_tel.cost_sensitivity_before <= 0.0 {
        audited_tel.cost_sensitivity_before =
            train_template.autopsy_telemetry.cost_sensitivity_before;
    }

    // Gentle Polyak trust-region blend (`beta = 0.50`) of top RL candidate (inherits blended `hedge_weight` & `conformal_q_bps`)
    if let Some((best_rl_cand, _, _, _, _)) = rl_scored.iter().max_by(|a, b| a.4.total_cmp(&b.4)) {
        let mut blended = immutable_champ.clone();
        blended.polyak_blend_from(best_rl_cand, 0.50);
        let (b_ev, b_rank, b_pass) = eval_cand_gate(&blended, &wm_cache);
        cand_results.push((blended, b_ev, b_rank, b_pass));
    }
    for (cand, ev, rank, pass, _) in rl_scored {
        cand_results.push((cand, ev, rank, pass));
    }

    // Step 3: Mandatory Full + Unseen Validation Promotion Gate
    let best_passing = cand_results
        .iter()
        .filter(|(_, _, _, passes)| *passes)
        .max_by(|a, b| a.2.total_cmp(&b.2));

    let sal_denom = (red_audited + paralysis_saliency_n).max(1) as f32;
    let mean_dx = sum_sal_dx / sal_denom;
    let mean_dh = sum_sal_dh / sal_denom;

    if let Some((best_cand, best_ev, _, _)) = best_passing {
        let cand_tel = best_cand.autopsy_telemetry.clone();
        let next_gen = immutable_champ.promoted_generation_id.saturating_add(1);
        let red_saved = champ_eval.red_loss_usd - best_ev.red_loss_usd;
        let (inc_nr, _, _) = eval_expectancy_metrics(&champ_eval);
        let (cnd_nr, _, _) = eval_expectancy_metrics(best_ev);
        let win_desc = format!(
            "{}->{} wins (Net {:+.2}R -> {:+.2}R)",
            champ_eval.winners_count, best_ev.winners_count, inc_nr, cnd_nr
        );
        *brain = best_cand.clone();
        brain.red_replay_buffer = full_reds;
        brain.winner_fisher_buffer = full_wins;
        brain.autopsy_telemetry.friction_inversion_events = cand_tel
            .friction_inversion_events
            .max(audited_tel.friction_inversion_events);
        brain.autopsy_telemetry.holding_eod_decay_events = cand_tel
            .holding_eod_decay_events
            .max(audited_tel.holding_eod_decay_events);
        brain.autopsy_telemetry.vehicle_regret_events = cand_tel
            .vehicle_regret_events
            .max(audited_tel.vehicle_regret_events);
        brain.autopsy_telemetry.deadzone_paralysis_events = cand_tel
            .deadzone_paralysis_events
            .max(audited_tel.deadzone_paralysis_events);
        brain.autopsy_telemetry.flat_regret_windows_replayed = cand_tel
            .flat_regret_windows_replayed
            .max(audited_tel.flat_regret_windows_replayed);
        brain.autopsy_telemetry.cost_sensitivity_before =
            if audited_tel.cost_sensitivity_before > 0.0 {
                audited_tel.cost_sensitivity_before
            } else {
                cand_tel.cost_sensitivity_before
            };
        brain.autopsy_telemetry.cost_sensitivity_after = brain
            .population
            .iter()
            .map(|m| m.hparams.cost_sensitivity)
            .sum::<f32>()
            / (brain.population.len().max(1) as f32);
        brain.promoted_generation_id = next_gen;
        brain.autopsy_telemetry.promoted_generations = next_gen;
        brain.online_updates_count = immutable_champ.online_updates_count.saturating_add(1);
        brain.snapshot_ewc_anchors();

        ledger_out.push(GenerationPromotionLedgerRow {
            gen_id: next_gen,
            phase_label: phase_label.to_string(),
            brain_label: idx.bucket.label().to_string(),
            red_trades_audited: red_audited,
            entry_faults,
            exit_faults,
            mean_saliency_dx: mean_dx,
            mean_saliency_dh: mean_dh,
            pf_before: champ_eval.net_pf,
            pf_after: best_ev.net_pf,
            pnl_before_usd: champ_eval.net_pnl_usd,
            pnl_after_usd: best_ev.net_pnl_usd,
            red_loss_reduced_usd: red_saved,
            winners_preserved_desc: win_desc,
            promoted: true,
        });
        true
    } else {
        // Best rejected candidate diagnostics for audit transparency while preserving `immutable_champ`
        let best_attempt = cand_results.iter().max_by(|a, b| a.2.total_cmp(&b.2));
        let (pf_att, pnl_att, red_diff, win_desc) = if let Some((_, ev, _, _)) = best_attempt {
            (
                ev.net_pf,
                ev.net_pnl_usd,
                champ_eval.red_loss_usd - ev.red_loss_usd,
                format!(
                    "{}->{} wins ({:+.2}R -> {:+.2}R)",
                    champ_eval.winners_count, ev.winners_count, champ_eval.win_r_sum, ev.win_r_sum
                ),
            )
        } else {
            (
                champ_eval.net_pf,
                champ_eval.net_pnl_usd,
                0.0,
                format!("{} wins ({:+.2}R)", champ_eval.winners_count, champ_eval.win_r_sum),
            )
        };
        restore_champion_weights_keeping_audit_memory(brain, immutable_champ);
        brain.red_replay_buffer = full_reds;
        brain.winner_fisher_buffer = full_wins;
        brain.autopsy_telemetry = audited_tel;
        brain.autopsy_telemetry.cost_sensitivity_after = brain
            .population
            .iter()
            .map(|m| m.hparams.cost_sensitivity)
            .sum::<f32>()
            / (brain.population.len().max(1) as f32);
        ledger_out.push(GenerationPromotionLedgerRow {
            gen_id: brain.promoted_generation_id,
            phase_label: phase_label.to_string(),
            brain_label: idx.bucket.label().to_string(),
            red_trades_audited: red_audited,
            entry_faults,
            exit_faults,
            mean_saliency_dx: mean_dx,
            mean_saliency_dh: mean_dh,
            pf_before: champ_eval.net_pf,
            pf_after: pf_att,
            pnl_before_usd: champ_eval.net_pnl_usd,
            pnl_after_usd: pnl_att,
            red_loss_reduced_usd: red_diff,
            winners_preserved_desc: win_desc,
            promoted: false,
        });
        false
    }
}

/// Causal Walk-Forward Inner PBT + World Model + Closed-Loop Red-Trade Loss-Autopsy Calibration (`<= T_oos_start`).
#[allow(clippy::too_many_arguments)]
fn train_warmup_with_causal_pbt(
    brain: &mut PureAutonomousRlBrain,
    bucket: ProofBucket,
    symbol_bars: &HashMap<String, Vec<RealExchangeOhlcvBar>>,
    precomputed: &HashMap<String, Vec<CausalBarPhysicsSnapshot>>,
    pretrain_m5_for_1m: Option<PretrainM5Context<'_>>,
    warmup_start_day: i64,
    warmup_cutoff_ts: i64,
    starting_equity: f64,
    promotion_ledger: &mut Vec<GenerationPromotionLedgerRow>,
) -> (WarmupTrainingTelemetry, usize) {
    let mut pretrain_bars_count = 0usize;

    // FIX #2b: Pre-train `SCALP_0DTE (1m)` encoder/trunk on causal `5m` session episodes
    // strictly `t_end <= d1_warmup_cutoff_ts` (`<= T_1m_oos_start`) before `1m` fine-tuning!
    if let Some((m5_bars, pre_m5, m5_min_day)) = pretrain_m5_for_1m {
        let m5_eps_raw = build_causal_session_episodes(
            ProofBucket::Scalp0Dte5m,
            m5_bars,
            pre_m5,
            m5_min_day,
            warmup_cutoff_ts,
            None,
            &[],
            &[],
        );
        let mut xs_m5 = Vec::new();
        for ep in &m5_eps_raw {
            for b in &ep.bars {
                xs_m5.push(b.x_212);
            }
        }
        pretrain_bars_count += xs_m5.len();
        if !xs_m5.is_empty() {
            brain.update_welford_batch(&xs_m5);
            let _ = brain.train_world_model_episodes(&m5_eps_raw, 6, 0.004);
            let m5_eps = build_causal_session_episodes(
                ProofBucket::Scalp0Dte5m,
                m5_bars,
                pre_m5,
                m5_min_day,
                warmup_cutoff_ts,
                Some(&brain.world_model),
                &brain.feat_mean,
                &brain.feat_var,
            );
            brain.train_population_episodes_bptt_parallel(&m5_eps, 3, 0.85, 0.0);
            // Reset Welford feature standardization so `1m` fine-tuning normalizes exact `1m` scale!
            brain.reset_welford();
        }
    }

    let idx = BucketTimelineIndex::build(bucket, symbol_bars, precomputed);
    let warmup_ts: Vec<i64> = idx
        .all_ts
        .iter()
        .copied()
        .filter(|&ts| {
            let d = est_trading_day_key(ts);
            d >= warmup_start_day && ts <= warmup_cutoff_ts
        })
        .collect();
    let mut warmup_days_set = BTreeSet::new();
    for &ts in &warmup_ts {
        warmup_days_set.insert(est_trading_day_key(ts));
    }

    let init_eps = build_causal_session_episodes(
        bucket,
        symbol_bars,
        precomputed,
        warmup_start_day,
        warmup_cutoff_ts,
        None,
        &[],
        &[],
    );
    let mut xs = Vec::new();
    for ep in &init_eps {
        for b in &ep.bars {
            xs.push(b.x_212);
        }
    }
    brain.update_welford_batch(&xs);

    // Stage A: Self-Supervised Multi-Horizon Physics World-Model Pre-Training (`1, 3, 8, 24`-bar Pinball Loss)
    let wm_epochs = if warmup_days_set.len() <= 3 { 8usize } else { 10usize };
    let _ = brain.train_world_model_episodes(&init_eps, wm_epochs, 0.0045);

    // Rebuild episodes with trained World-Model cross-day causal regime context `prior_regime_z`
    let full_eps = build_causal_session_episodes(
        bucket,
        symbol_bars,
        precomputed,
        warmup_start_day,
        warmup_cutoff_ts,
        Some(&brain.world_model),
        &brain.feat_mean,
        &brain.feat_var,
    );
    let total_bars = pretrain_bars_count + full_eps.iter().map(|e| e.bars.len()).sum::<usize>();

    // Stage B: Purged Expanding Inner Walk-Forward PBT Calibration (`<= warmup_cutoff_ts`)
    // Strictly cap fold BPTT training at `<= 0.70 * n_ts` so the final 30% (`[0.71 .. 1.00]`) of warmup
    // remains 100% Out-Of-Fold (unseen by policy weights!) for specialist `hedge_weight` selection!
    let (n_folds, epochs_per_gen) = (4usize, 3usize);
    let n_ts = warmup_ts.len();
    let mut best_pf = 0.0_f64;
    let mut best_pnl = 0.0_f64;
    let mut best_trades = 0usize;
    let fold_specs: [(f64, f64, f64); 4] = [
        (0.38, 0.40, 0.64),
        (0.54, 0.56, 0.80),
        (0.68, 0.70, 1.00),
        (0.68, 0.70, 1.00),
    ];
    let heldout_val_start_i = ((n_ts as f64) * 0.70).floor() as usize;
    let last_val_start_ts = warmup_ts
        .get(heldout_val_start_i)
        .copied()
        .unwrap_or(warmup_cutoff_ts);
    let max_train_end_i = ((n_ts as f64) * 0.68).floor() as usize;
    let max_train_cutoff_ts = warmup_ts
        .get(max_train_end_i)
        .copied()
        .unwrap_or(warmup_cutoff_ts);

    let slice_eps_le = |cutoff_ts: i64| -> Vec<CausalSessionEpisode> {
        let mut out = Vec::new();
        for ep in &full_eps {
            let sliced_bars: Vec<CausalTrajectoryBar> = ep
                .bars
                .iter()
                .filter(|b| b.ts_sec <= cutoff_ts)
                .cloned()
                .collect();
            if sliced_bars.len() >= 4 {
                out.push(CausalSessionEpisode {
                    symbol: ep.symbol.clone(),
                    day_key: ep.day_key,
                    start_ts_sec: sliced_bars[0].ts_sec,
                    end_ts_sec: sliced_bars[sliced_bars.len() - 1].ts_sec,
                    is_intraday_session: ep.is_intraday_session,
                    prior_regime_z: ep.prior_regime_z,
                    bars: sliced_bars,
                });
            }
        }
        out
    };

    for (fold_idx, &(train_frac, val_start_frac, val_end_frac)) in fold_specs.iter().enumerate() {
        let train_end_i = ((n_ts as f64) * train_frac).floor() as usize;
        let val_start_i = ((n_ts as f64) * val_start_frac).floor() as usize;
        let val_end_i = (((n_ts as f64) * val_end_frac).ceil() as usize).min(n_ts.saturating_sub(1));

        let train_cutoff_ts = warmup_ts.get(train_end_i).copied().unwrap_or(max_train_cutoff_ts);
        let val_start_ts = warmup_ts.get(val_start_i).copied().unwrap_or(last_val_start_ts);
        let val_end_ts = warmup_ts.get(val_end_i).copied().unwrap_or(warmup_cutoff_ts);

        let fold_eps = slice_eps_le(train_cutoff_ts);
        let train_slice = if !fold_eps.is_empty() {
            &fold_eps[..]
        } else {
            &full_eps[..]
        };

        let fold_epochs = if fold_idx + 1 == n_folds {
            2
        } else {
            epochs_per_gen
        };
        let fold_lr = if fold_idx + 1 == n_folds { 0.55 } else { 1.0 };
        brain.train_population_episodes_bptt_parallel(train_slice, fold_epochs, fold_lr, 0.0);
        if bucket.is_intraday() {
            let _ = brain.train_intraday_exit_timing_episodes(train_slice, 2, 0.42 * fold_lr);
        }
        let _ = brain.calibrate_causal_conformal_residual(
            train_slice,
            train_cutoff_ts,
            if best_trades > 0 { best_pf } else { 1.08 },
            best_trades,
        );
        let wm_cache = idx.precompute_wm_cache(
            &brain.world_model,
            &brain.feat_mean,
            &brain.feat_var,
            val_end_ts,
        );
        let mut best_ev = evaluate_population_indexed_parallel(
            brain,
            &idx,
            &wm_cache,
            val_start_ts,
            val_end_ts,
            starting_equity,
        );
        if best_ev.trades.len() < 3 {
            let pre_wake = brain.clone();
            let _ = brain.audit_and_train_paralysis_regret(
                train_slice,
                train_cutoff_ts,
                3,
                0.55,
            );
            let _ = brain.train_vehicle_regret_episodes(train_slice, 2, 0.45);
            let post_ev = evaluate_population_indexed_parallel(
                brain,
                &idx,
                &wm_cache,
                val_start_ts,
                val_end_ts,
                starting_equity,
            );
            // Unified U_lcb wake check: accept wakeup whenever `U_lcb` strictly improves and `Net PF > 1.0`!
            if post_ev.trades.len() > best_ev.trades.len()
                && post_ev.net_pf > 1.0
                && compute_utility_lcb(&post_ev) >= compute_utility_lcb(&best_ev).max(0.0)
            {
                best_ev = post_ev;
            } else {
                let keep_par = brain.autopsy_telemetry.deadzone_paralysis_events;
                let keep_flat = brain.autopsy_telemetry.flat_regret_windows_replayed;
                *brain = pre_wake;
                brain.autopsy_telemetry.deadzone_paralysis_events = keep_par;
                brain.autopsy_telemetry.flat_regret_windows_replayed = keep_flat;
            }
        }
        best_pf = best_ev.net_pf;
        best_pnl = best_ev.net_pnl_usd;
        best_trades = best_ev.trades.len();

        // Run PBT exploit & explore after folds 0..2; fold 3 evaluates post-PBT specialists strictly OOF!
        if fold_idx + 1 < n_folds {
            brain.pbt_exploit_and_explore();
        }
    }

    // Final strictly Out-Of-Fold specialist scoring on `[last_val_start_ts .. warmup_cutoff_ts]`
    // (zero BPTT training ever touched bars `> max_train_cutoff_ts`!).
    let _ = brain.calibrate_causal_conformal_residual(
        &full_eps,
        max_train_cutoff_ts,
        if best_trades > 0 { best_pf } else { 1.05 },
        best_trades,
    );
    let wm_cache_full = idx.precompute_wm_cache(
        &brain.world_model,
        &brain.feat_mean,
        &brain.feat_var,
        warmup_cutoff_ts,
    );
    let mut best_held_out = evaluate_population_indexed_parallel(
        brain,
        &idx,
        &wm_cache_full,
        last_val_start_ts,
        warmup_cutoff_ts,
        starting_equity,
    );
    if best_held_out.trades.len() < 3 {
        let pre_wake = brain.clone();
        let cap_eps = slice_eps_le(max_train_cutoff_ts);
        let wake_ref = if !cap_eps.is_empty() {
            &cap_eps[..]
        } else {
            &full_eps[..]
        };
        let _ = brain.audit_and_train_paralysis_regret(wake_ref, max_train_cutoff_ts, 3, 0.55);
        let _ = brain.train_vehicle_regret_episodes(wake_ref, 2, 0.45);
        let post_held_out = evaluate_population_indexed_parallel(
            brain,
            &idx,
            &wm_cache_full,
            last_val_start_ts,
            warmup_cutoff_ts,
            starting_equity,
        );
        if post_held_out.trades.len() > best_held_out.trades.len()
            && post_held_out.net_pf > 1.0
            && compute_utility_lcb(&post_held_out) >= compute_utility_lcb(&best_held_out).max(0.0)
        {
            best_held_out = post_held_out;
        } else {
            let keep_par = brain.autopsy_telemetry.deadzone_paralysis_events;
            let keep_flat = brain.autopsy_telemetry.flat_regret_windows_replayed;
            *brain = pre_wake;
            brain.autopsy_telemetry.deadzone_paralysis_events = keep_par;
            brain.autopsy_telemetry.flat_regret_windows_replayed = keep_flat;
        }
    }
    let eta_grid: [f64; 5] = [1.8, 3.2, 6.0, 11.0, 18.0];
    let select_best_eta = |b: &mut PureAutonomousRlBrain| -> (f64, DiscreteCausalEvalSummary, f64) {
        let mut b_eta = 3.2_f64;
        let mut b_sc = f64::NEG_INFINITY;
        let mut b_ev: Option<DiscreteCausalEvalSummary> = None;
        for &eta in &eta_grid {
            b.update_hedge_weights_with_eta(eta);
            let ev = evaluate_brain_ensemble_retest_indexed(
                b,
                &idx,
                &wm_cache_full,
                last_val_start_ts,
                warmup_cutoff_ts,
                starting_equity,
            );
            let active_bonus = if !ev.trades.is_empty() && ev.net_pf > 1.0 {
                6_000.0
            } else {
                0.0
            };
            let sc = active_bonus
                + 0.70 * compute_utility_lcb(&ev)
                + 0.30 * compute_recency_weighted_causal_score(&ev, warmup_cutoff_ts, bucket.is_intraday());
            if sc > b_sc {
                b_sc = sc;
                b_eta = eta;
                b_ev = Some(ev);
            }
        }
        b.update_hedge_weights_with_eta(b_eta);
        let final_ev = b_ev.unwrap_or_else(|| {
            evaluate_brain_ensemble_retest_indexed(
                b,
                &idx,
                &wm_cache_full,
                last_val_start_ts,
                warmup_cutoff_ts,
                starting_equity,
            )
        });
        (b_eta, final_ev, b_sc)
    };

    let (_, mut ens_selected_ev, ens_selected_sc) = select_best_eta(brain);
    // If the ensemble itself diluted individual specialists into `< 3` OOF trades, run causal paralysis-regret
    // wakeup strictly `<= max_train_cutoff_ts` and accept if OOF ensemble utility improves!
    if ens_selected_ev.trades.len() < 3 {
        let pre_ens_wake = brain.clone();
        let cap_eps = slice_eps_le(max_train_cutoff_ts);
        let wake_ref = if !cap_eps.is_empty() {
            &cap_eps[..]
        } else {
            &full_eps[..]
        };
        let _ = brain.audit_and_train_paralysis_regret(wake_ref, max_train_cutoff_ts, 3, 0.58);
        let _ = brain.train_vehicle_regret_episodes(wake_ref, 2, 0.45);
        let post_pop_ev = evaluate_population_indexed_parallel(
            brain,
            &idx,
            &wm_cache_full,
            last_val_start_ts,
            warmup_cutoff_ts,
            starting_equity,
        );
        let (_, post_ens_ev, post_ens_sc) = select_best_eta(brain);
        if post_ens_ev.trades.len() > ens_selected_ev.trades.len()
            && post_ens_ev.net_pf > 1.0
            && compute_utility_lcb(&post_ens_ev) >= compute_utility_lcb(&ens_selected_ev).max(0.0)
            && post_ens_sc >= ens_selected_sc
        {
            ens_selected_ev = post_ens_ev;
            if post_pop_ev.trades.len() > best_held_out.trades.len() {
                best_held_out = post_pop_ev;
            }
        } else {
            let keep_par = brain.autopsy_telemetry.deadzone_paralysis_events;
            let keep_flat = brain.autopsy_telemetry.flat_regret_windows_replayed;
            *brain = pre_ens_wake;
            brain.autopsy_telemetry.deadzone_paralysis_events = keep_par;
            brain.autopsy_telemetry.flat_regret_windows_replayed = keep_flat;
        }
    }
    if !best_held_out.trades.is_empty() {
        best_pf = best_held_out.net_pf;
        best_pnl = best_held_out.net_pnl_usd;
        best_trades = best_held_out.trades.len();
    }

    brain.promoted_generation_id = 1;
    brain.autopsy_telemetry.promoted_generations = 1;

    let mut ens_warmup = ens_selected_ev;
    // Calibrate conformal epistemic residual on full warmup OOF results (`<= warmup_cutoff_ts`)
    let _ = brain.calibrate_causal_conformal_residual(
        &full_eps,
        warmup_cutoff_ts,
        if !ens_warmup.trades.is_empty() {
            ens_warmup.net_pf
        } else {
            best_pf.max(0.95)
        },
        ens_warmup.trades.len(),
    );
    if ens_warmup.trades.len() < 5 || ens_warmup.net_pf < 1.22 {
        let cap_eps = slice_eps_le(warmup_cutoff_ts);
        let cal_eps = if !cap_eps.is_empty() {
            &cap_eps[..]
        } else {
            &full_eps[..]
        };
        let cal_retest_start_ts = warmup_ts.first().copied().unwrap_or(last_val_start_ts);
        let pre_cal = brain.clone();
        let promoted = execute_closed_loop_red_trade_autopsy_and_promotion(
            brain,
            &idx,
            cal_eps,
            cal_retest_start_ts,
            warmup_cutoff_ts,
            starting_equity,
            "WARMUP_CAL",
            promotion_ledger,
        );
        if promoted {
            let wm_post = idx.precompute_wm_cache(
                &brain.world_model,
                &brain.feat_mean,
                &brain.feat_var,
                warmup_cutoff_ts,
            );
            ens_warmup = evaluate_brain_ensemble_retest_indexed(
                brain,
                &idx,
                &wm_post,
                last_val_start_ts,
                warmup_cutoff_ts,
                starting_equity,
            );
        } else {
            restore_champion_weights_keeping_audit_memory(brain, pre_cal);
        }
    }
    brain.snapshot_ewc_anchors();
    if !ens_warmup.trades.is_empty() {
        best_pf = ens_warmup.net_pf;
        best_pnl = ens_warmup.net_pnl_usd;
        best_trades = ens_warmup.trades.len();
    }

    let tel = WarmupTrainingTelemetry {
        brain_label: bucket.label().to_string(),
        population_size: POPULATION_SIZE,
        train_sequences: full_eps.len(),
        best_val_net_pf: best_pf,
        best_val_net_pnl: best_pnl,
        best_val_trades: best_trades,
        pbt_generations: brain.pbt_generations_run,
    };
    (tel, total_bars)
}

#[derive(Debug, Clone)]
struct OosTimelineResult {
    blotter: Vec<BlotterTradeRecord>,
    abstained_flat_oos: usize,
    oos_counterfactuals_harvested: usize,
}

/// Executes the strictly Out-of-Sample (`cur_day >= oos_start_day`) bar-by-bar simulation for
/// either `RUN_A_FROZEN_WARMUP_ML` (frozen) or `RUN_B_ONLINE_RL` (continual EOD World-Model TTT +
/// 4-Step Red-Trade Loss-Autopsy RL + Multiplicative-Weights Specialist Routing on BOTH Dataset 1 & 2!).
#[allow(clippy::too_many_arguments)]
fn run_oos_timeline_with_pure_rl_brains(
    run_mode: ProofRunMode,
    symbol_bars: &HashMap<String, Vec<RealExchangeOhlcvBar>>,
    precomputed: &HashMap<String, Vec<CausalBarPhysicsSnapshot>>,
    active_buckets: &[ProofBucket],
    brains: &mut BTreeMap<ProofBucket, PureAutonomousRlBrain>,
    warmup_start_day: i64,
    regular_hours_only: bool,
    oos_start_day: i64,
    starting_equity: f64,
    promotion_ledger: &mut Vec<GenerationPromotionLedgerRow>,
) -> OosTimelineResult {
    let adapt_online = run_mode == ProofRunMode::RunBOnlineRl;
    let (all_ts, _) = collect_sorted_timestamps_and_days(symbol_bars, regular_hours_only);
    if all_ts.is_empty() {
        return OosTimelineResult {
            blotter: Vec::new(),
            abstained_flat_oos: 0,
            oos_counterfactuals_harvested: 0,
        };
    }

    let mut ts_index_by_sym: HashMap<String, HashMap<i64, usize>> = HashMap::new();
    let mut sorted_syms: Vec<String> = symbol_bars.keys().cloned().collect();
    sorted_syms.sort();
    for sym in &sorted_syms {
        let bars = &symbol_bars[sym];
        let mut idx_map = HashMap::with_capacity(bars.len());
        for (i, b) in bars.iter().enumerate() {
            idx_map.insert(b.timestamp_sec, i);
        }
        ts_index_by_sym.insert(sym.clone(), idx_map);
    }

    let mut bucket_indices: BTreeMap<ProofBucket, BucketTimelineIndex<'_>> = BTreeMap::new();
    if adapt_online {
        for &b in active_buckets {
            bucket_indices.insert(b, BucketTimelineIndex::build(b, symbol_bars, precomputed));
        }
    }

    let eval_start_ts = all_ts
        .iter()
        .copied()
        .find(|&ts| est_trading_day_key(ts) >= warmup_start_day)
        .unwrap_or(0);

    let mut equity_usd = starting_equity.max(50_000.0);
    let mut trade_seq: u128 = 1;
    let mut pending_signals: Vec<PendingNextBarSignal> = Vec::new();
    let mut open_positions: Vec<ActiveSimulatedPosition> = Vec::new();
    let mut oos_blotter: Vec<BlotterTradeRecord> = Vec::new();
    let mut abstained_flat_oos = 0usize;
    let mut oos_cf_harvested = 0usize;
    let mut prev_oos_day: Option<i64> = None;
    let mut prev_ts_sec: i64 = 0;
    let mut completed_oos_days_count = 0usize;
    let mut red_trades_seen_by_bucket: BTreeMap<ProofBucket, usize> = BTreeMap::new();
    let mut last_autopsy_day: BTreeMap<ProofBucket, usize> = BTreeMap::new();

    let mut hidden_states: BTreeMap<ProofBucket, HashMap<String, Vec<[f32; HIDDEN_DIM]>>> =
        BTreeMap::new();
    let mut wm_hiddens: BTreeMap<ProofBucket, HashMap<String, [f32; WM_DIM]>> = BTreeMap::new();
    let mut last_sym_day: BTreeMap<ProofBucket, HashMap<String, i64>> = BTreeMap::new();
    let mut ema_turn_map: BTreeMap<ProofBucket, HashMap<String, f32>> = BTreeMap::new();
    let mut prev_act_map: BTreeMap<ProofBucket, HashMap<String, f32>> = BTreeMap::new();

    for &t_sec in &all_ts {
        let cur_day = est_trading_day_key(t_sec);
        if cur_day < oos_start_day {
            // Causal hidden & World-Model state pre-warming during `[oos_start_day - 7 .. oos_start_day)`
            // so swing & intraday recurrent states begin OOS synchronized (zero orders created).
            if cur_day >= oos_start_day.saturating_sub(7) && cur_day >= warmup_start_day {
                let (_, hhmm) = ny_et_hhmm(t_sec);
                for &bucket in active_buckets {
                    let Some(brain) = brains.get(&bucket) else {
                        continue;
                    };
                    let norm_rem = normalized_session_remaining(bucket, hhmm);
                    let turn_alpha: f32 = if bucket.is_intraday() { 0.14 } else { 0.18 };
                    let day_map = last_sym_day.entry(bucket).or_default();
                    let h_map = hidden_states.entry(bucket).or_default();
                    let wm_map = wm_hiddens.entry(bucket).or_default();
                    let turn_map = ema_turn_map.entry(bucket).or_default();
                    let pact_map = prev_act_map.entry(bucket).or_default();
                    for sym in &sorted_syms {
                        let Some(&bar_idx) = ts_index_by_sym.get(sym).and_then(|m| m.get(&t_sec))
                        else {
                            continue;
                        };
                        let bar_t = &symbol_bars[sym][bar_idx];
                        if !symbol_matches_bucket(sym, bar_t.close, bucket) {
                            continue;
                        }
                        if bucket.is_intraday() {
                            let prev_d = day_map.get(sym).copied().unwrap_or(i64::MIN);
                            if prev_d != cur_day {
                                h_map.remove(sym);
                                if prev_d != i64::MIN
                                    && let Some(t_ref) = turn_map.get_mut(sym)
                                {
                                    *t_ref *= 0.25;
                                }
                                pact_map.insert(sym.clone(), 0.0);
                                if prev_d != i64::MIN
                                    && let Some(z_ref) = wm_map.get_mut(sym)
                                {
                                    for v in z_ref.iter_mut() {
                                        *v *= 0.45;
                                    }
                                }
                                day_map.insert(sym.clone(), cur_day);
                            }
                        }
                        let snap = &precomputed[sym][bar_idx];
                        if !snap.valid {
                            continue;
                        }
                        let is_penny = is_penny_symbol_cand(sym, bar_t.close);
                        let option_allowed =
                            !is_penny && is_option_vehicle_allowed(sym, bar_t.close);
                        let slip_bps =
                            explicit_one_way_slippage_bps(sym, bar_t.close, is_penny) as f32;
                        let raw_x = snap.to_raw_180_vector(norm_rem, 0.0, false);
                        let cur_turn = turn_map.get(sym).copied().unwrap_or(0.0);
                        let prev_act = pact_map.get(sym).copied().unwrap_or(0.0);
                        let self_s = SelfAwarenessState {
                            current_pos: 0.0,
                            unrealized_net_ret: 0.0,
                            peak_excursion: 0.0,
                            drawdown_from_peak: 0.0,
                            log_bars_held: 0.0,
                            one_way_cost_bps: slip_bps,
                            session_time_left: norm_rem as f32,
                            ema_turnover: cur_turn,
                        };
                        let prev_hs = h_map.get(sym).map(Vec::as_slice).unwrap_or(&[]);
                        let prev_wm_h = wm_map.get(sym).copied().unwrap_or([0.0; WM_DIM]);
                        let inf = brain.infer_ensemble_step(
                            &raw_x,
                            self_s,
                            prev_hs,
                            &prev_wm_h,
                            option_allowed,
                        );
                        h_map.insert(sym.clone(), inf.next_hiddens);
                        wm_map.insert(sym.clone(), inf.next_wm_hidden);
                        let next_turn = (1.0 - turn_alpha) * cur_turn
                            + turn_alpha * (inf.action - prev_act).abs().min(2.0);
                        turn_map.insert(sym.clone(), next_turn);
                        pact_map.insert(sym.clone(), inf.action);
                    }
                }
            }
            prev_ts_sec = t_sec;
            continue;
        }

        // FIX #3 & PART 2: At the end of EVERY completed OOS trading day `d` (`t_resolved <= prev_ts_sec`)
        // on BOTH Dataset 1 AND Dataset 2, execute causal World-Model TTT + Multiplicative-Weights
        // specialist routing + Closed-Loop Red-Trade Loss-Autopsy -> Re-Test -> Promotion!
        if adapt_online
            && let Some(pd) = prev_oos_day
            && cur_day > pd
            && prev_ts_sec > 0
        {
            completed_oos_days_count += 1;
            for &bucket in active_buckets {
                let Some(b_idx) = bucket_indices.get(&bucket) else {
                    continue;
                };
                let Some(brain) = brains.get_mut(&bucket) else {
                    continue;
                };

                // 1) Count day `pd`'s strictly resolved causal bars (`<= prev_ts_sec`)
                let newly_bars: usize = b_idx
                    .bars_by_sym
                    .iter()
                    .map(|bs| {
                        let end = bs.partition_point(|b| b.timestamp_sec <= prev_ts_sec);
                        bs[..end]
                            .iter()
                            .rev()
                            .take_while(|b| est_trading_day_key(b.timestamp_sec) == pd)
                            .count()
                    })
                    .sum();
                oos_cf_harvested += newly_bars.max(1);

                let cur_red_count = oos_blotter
                    .iter()
                    .filter(|t| {
                        t.bucket == bucket
                            && (t.net_pnl_usd < 0.0
                                || (t.exit_reason.contains("EOD_") && t.net_r < 0.35))
                    })
                    .count();
                let prev_red_count = red_trades_seen_by_bucket.get(&bucket).copied().unwrap_or(0);
                let has_new_red_trade = cur_red_count > prev_red_count;
                let bucket_oos_trades = oos_blotter.iter().filter(|t| t.bucket == bucket).count();
                let slow_pace = completed_oos_days_count >= 3
                    && bucket_oos_trades * 3 < completed_oos_days_count;
                let days_since_last = completed_oos_days_count
                    .saturating_sub(last_autopsy_day.get(&bucket).copied().unwrap_or(0));
                let should_run_autopsy = if bucket == ProofBucket::Scalp0Dte1m {
                    has_new_red_trade
                        || completed_oos_days_count == 1
                        || days_since_last >= 1
                } else {
                    has_new_red_trade
                        || completed_oos_days_count == 1
                        || days_since_last >= 3
                        || ((bucket_oos_trades == 0 || slow_pace) && days_since_last >= 2)
                };

                let lookback_days = match bucket {
                    ProofBucket::Scalp0Dte1m => 12,
                    ProofBucket::Scalp0Dte5m
                    | ProofBucket::Brain01PennyDay
                    | ProofBucket::Brain02LargeCapDay => 20,
                    ProofBucket::Brain03ShortSwing | ProofBucket::Brain04CoreSwing => 36,
                };
                let rolling_min_day = (pd - lookback_days).max(warmup_start_day);
                let rolling_retest_start_ts = b_idx
                    .all_ts
                    .iter()
                    .copied()
                    .find(|&ts| est_trading_day_key(ts) >= rolling_min_day)
                    .unwrap_or(eval_start_ts);

                // Trigger full 4-Step Red-Trade Loss Audit -> RL Update -> Re-Test -> Promotion whenever
                // a new red trade settled on day `pd` OR on periodic EOD regime adaptation checkpoints!
                if should_run_autopsy {
                    red_trades_seen_by_bucket.insert(bucket, cur_red_count);
                    last_autopsy_day.insert(bucket, completed_oos_days_count);

                    // Candidate World-Model TTT + RL update strictly gated by causal rolling re-test!
                    let backup_brain = brain.clone();
                    let causal_eps = build_causal_session_episodes(
                        bucket,
                        symbol_bars,
                        precomputed,
                        rolling_min_day,
                        prev_ts_sec,
                        Some(&brain.world_model),
                        &brain.feat_mean,
                        &brain.feat_var,
                    );
                    let promoted = execute_closed_loop_red_trade_autopsy_and_promotion(
                        brain,
                        b_idx,
                        &causal_eps,
                        rolling_retest_start_ts,
                        prev_ts_sec,
                        starting_equity,
                        &format!("OOS_D{completed_oos_days_count}"),
                        promotion_ledger,
                    );
                    if !promoted {
                        restore_champion_weights_keeping_audit_memory(brain, backup_brain);
                    }
                }
            }
        }
        prev_oos_day = Some(cur_day);
        prev_ts_sec = t_sec;

        // Session boundary reset for intraday hidden states (carry ONLY `0.45 * z_wm` cross-day regime context!)
        for &bucket in active_buckets {
            if !bucket.is_intraday() {
                continue;
            }
            let day_map = last_sym_day.entry(bucket).or_default();
            let h_map = hidden_states.entry(bucket).or_default();
            let wm_map = wm_hiddens.entry(bucket).or_default();
            let turn_map = ema_turn_map.entry(bucket).or_default();
            let pact_map = prev_act_map.entry(bucket).or_default();
            for sym in &sorted_syms {
                if ts_index_by_sym.get(sym).is_some_and(|m| m.contains_key(&t_sec)) {
                    let prev_d = day_map.get(sym).copied().unwrap_or(i64::MIN);
                    if prev_d != cur_day {
                        h_map.remove(sym);
                        if prev_d != i64::MIN
                            && let Some(t_ref) = turn_map.get_mut(sym)
                        {
                            *t_ref *= 0.25;
                        }
                        pact_map.insert(sym.clone(), 0.0);
                        if prev_d != i64::MIN
                            && let Some(z_ref) = wm_map.get_mut(sym)
                        {
                            for v in z_ref.iter_mut() {
                                *v *= 0.45;
                            }
                        }
                        day_map.insert(sym.clone(), cur_day);
                    }
                }
            }
        }

        // A. FILL PENDING NEXT-BAR-OPEN SIGNALS
        let mut newly_opened_ids: BTreeSet<Uuid> = BTreeSet::new();
        for sig in pending_signals.drain(..) {
            let Some(&bar_idx) = ts_index_by_sym.get(&sig.symbol).and_then(|m| m.get(&t_sec))
            else {
                continue;
            };
            let entry_bar = &symbol_bars[&sig.symbol][bar_idx];
            let entry_snap = &precomputed[&sig.symbol][bar_idx];
            let trade_id = Uuid::from_u128(
                0xD000_0000_0000_0000_u128 | (trade_seq << 8) | (sig.bucket as u128),
            );
            if let Some(pos) = open_simulated_position_at_next_bar(
                sig.sym_id,
                &sig.symbol,
                sig.bucket,
                sig.brain_id,
                sig.signal_ts_sec,
                sig.signal_day_key,
                sig.direction,
                sig.target_action,
                sig.tail_stop_dist,
                sig.is_option_routed,
                sig.is_vertical_combo,
                sig.is_penny_tier,
                sig.dte_days,
                sig.kelly_risk_budget_usd,
                equity_usd,
                entry_bar,
                entry_snap,
                sig.embedding_128,
                sig.confidence_pct,
                sig.active_gen_id,
                trade_id,
            ) {
                trade_seq = trade_seq.wrapping_add(1);
                newly_opened_ids.insert(trade_id);
                open_positions.push(pos);
            }
        }

        // Track `(bucket, symbol)` pairs whose World-Model recurrent latent state `z_wm` stepped on bar `t_sec`
        // so symbols skipped due to slot caps / EOD cutoff still maintain continuous `z_wm` parity with `wm_cache`!
        let mut stepped_wm_this_bar: BTreeSet<(ProofBucket, String)> = BTreeSet::new();

        // B. STEP ALL OPEN POSITIONS AT BAR `t_sec`
        let step_open_heat_frac = open_positions
            .iter()
            .map(|p| p.risk_budget_usd)
            .sum::<f64>()
            / equity_usd.max(50_000.0);
        let mut remaining_open = Vec::with_capacity(open_positions.len());
        for mut pos in open_positions.drain(..) {
            let Some(&bar_idx) = ts_index_by_sym.get(&pos.symbol).and_then(|m| m.get(&t_sec))
            else {
                remaining_open.push(pos);
                continue;
            };
            let bar_t = &symbol_bars[&pos.symbol][bar_idx];
            let snap_t = &precomputed[&pos.symbol][bar_idx];
            let is_new = newly_opened_ids.contains(&pos.trade_id);
            let sym_exit_iv = exit_iv_for_bar(pos.bucket, snap_t);

            if let Some((exit_px, reason, is_full_stop)) =
                step_active_position_on_bar(&mut pos, bar_t, sym_exit_iv, is_new)
            {
                let (blotter, _) = finalize_settled_position(
                    &pos,
                    t_sec,
                    exit_px,
                    reason,
                    is_full_stop,
                    sym_exit_iv,
                    run_mode,
                );
                equity_usd += blotter.net_pnl_usd;
                prev_act_map
                    .entry(pos.bucket)
                    .or_default()
                    .insert(pos.symbol.clone(), 0.0);
                oos_blotter.push(blotter);
            } else {
                if !is_new
                    && snap_t.valid
                    && let Some(brain) = brains.get(&pos.bucket)
                {
                    let turn_alpha: f32 = if pos.bucket.is_intraday() { 0.14 } else { 0.18 };
                    let (_, hhmm) = ny_et_hhmm(t_sec);
                    let norm_rem = normalized_session_remaining(pos.bucket, hhmm);
                    let raw_x = snap_t.to_raw_180_vector(
                        norm_rem,
                        step_open_heat_frac,
                        pos.is_option_routed,
                    );
                    let cur_turn = ema_turn_map
                        .entry(pos.bucket)
                        .or_default()
                        .get(&pos.symbol)
                        .copied()
                        .unwrap_or(0.0);
                    let prev_act = prev_act_map
                        .entry(pos.bucket)
                        .or_default()
                        .get(&pos.symbol)
                        .copied()
                        .unwrap_or(0.0);
                    let (self_s, _) = build_active_position_self_awareness(
                        &mut pos,
                        t_sec,
                        bar_t.close,
                        sym_exit_iv,
                        norm_rem,
                        cur_turn,
                    );
                    let bucket_h = hidden_states.entry(pos.bucket).or_default();
                    let bucket_wm_h = wm_hiddens.entry(pos.bucket).or_default();
                    let prev_hs = bucket_h.get(&pos.symbol).map(Vec::as_slice).unwrap_or(&[]);
                    let prev_wm_h =
                        bucket_wm_h.get(&pos.symbol).copied().unwrap_or([0.0; WM_DIM]);
                    let opt_allowed =
                        !pos.is_penny_tier && is_option_vehicle_allowed(&pos.symbol, bar_t.close);
                    let inf = brain.infer_ensemble_step(
                        &raw_x,
                        self_s,
                        prev_hs,
                        &prev_wm_h,
                        opt_allowed,
                    );
                    bucket_h.insert(pos.symbol.clone(), inf.next_hiddens);
                    bucket_wm_h.insert(pos.symbol.clone(), inf.next_wm_hidden);
                    let next_turn = (1.0 - turn_alpha) * cur_turn
                        + turn_alpha * (inf.action - prev_act).abs().min(2.0);
                    ema_turn_map
                        .entry(pos.bucket)
                        .or_default()
                        .insert(pos.symbol.clone(), next_turn);
                    prev_act_map
                        .entry(pos.bucket)
                        .or_default()
                        .insert(pos.symbol.clone(), inf.action);
                    stepped_wm_this_bar.insert((pos.bucket, pos.symbol.clone()));

                    let pos_sign = pos.direction.sign() as f32;
                    let exit_snap = (0.22 * pos.current_target_action.abs()).clamp(0.028, 0.075);
                    if inf.action.abs() < exit_snap || inf.action.signum() != pos_sign {
                        pos.pending_exit = true;
                    } else if (inf.action - pos.current_target_action).abs() >= 0.30 {
                        pos.pending_scale_action = Some(inf.action);
                    }
                }
                remaining_open.push(pos);
            }
        }
        open_positions = remaining_open;

        // C. NEURAL ENSEMBLE ACTION SELECTION AT BAR `t` CLOSE
        let (_, hhmm) = ny_et_hhmm(t_sec);
        let mut open_heat_frac = open_positions
            .iter()
            .map(|p| p.risk_budget_usd)
            .sum::<f64>()
            / equity_usd.max(50_000.0);
        let global_can_enter = open_positions.len() < 20 && open_heat_frac < 0.28;

        for &bucket in active_buckets {
            let Some(brain) = brains.get(&bucket) else {
                continue;
            };
            let norm_rem = normalized_session_remaining(bucket, hhmm);
            let max_bucket_slots = if bucket.is_intraday() { 2 } else { 3 };
            let bucket_active = open_positions.iter().filter(|p| p.bucket == bucket).count()
                + pending_signals.iter().filter(|s| s.bucket == bucket).count();
            let bucket_can_enter = global_can_enter
                && entry_allowed_at_hhmm(bucket, hhmm)
                && bucket_active < max_bucket_slots;

            let mut scored_actions: Vec<(f64, PendingNextBarSignal)> = Vec::new();
            let bucket_h = hidden_states.entry(bucket).or_default();
            let bucket_wm_h = wm_hiddens.entry(bucket).or_default();

            for (sym_ord, sym) in sorted_syms.iter().enumerate() {
                let Some(&bar_idx) = ts_index_by_sym.get(sym).and_then(|m| m.get(&t_sec)) else {
                    continue;
                };
                let bar_t = &symbol_bars[sym][bar_idx];
                if !symbol_matches_bucket(sym, bar_t.close, bucket) {
                    continue;
                }
                let snap = &precomputed[sym][bar_idx];
                if !snap.valid {
                    continue;
                }

                let already_open_or_pending = open_positions
                    .iter()
                    .any(|p| p.symbol == *sym && p.bucket == bucket)
                    || pending_signals
                        .iter()
                        .any(|s| s.symbol == *sym && s.bucket == bucket);
                if !bucket_can_enter || already_open_or_pending {
                    // Maintain continuous World-Model latent state `z_wm` even when flat entry is gated!
                    if !stepped_wm_this_bar.contains(&(bucket, sym.clone())) {
                        let raw_x = snap.to_raw_180_vector(norm_rem, 0.0, false);
                        let x_norm = brain.normalize_x(&raw_x);
                        let prev_wm_h = bucket_wm_h.get(sym).copied().unwrap_or([0.0; WM_DIM]);
                        let (next_z, _) = brain.world_model.forward_step(&x_norm, &prev_wm_h);
                        bucket_wm_h.insert(sym.clone(), next_z);
                    }
                    continue;
                }

                let turn_alpha: f32 = if bucket.is_intraday() { 0.14 } else { 0.18 };
                let is_penny = is_penny_symbol_cand(sym, bar_t.close);
                let option_allowed = !is_penny && is_option_vehicle_allowed(sym, bar_t.close);
                let slip_bps = explicit_one_way_slippage_bps(sym, bar_t.close, is_penny) as f32;
                let raw_x = snap.to_raw_180_vector(norm_rem, open_heat_frac, false);
                let cur_turn = ema_turn_map
                    .entry(bucket)
                    .or_default()
                    .get(sym)
                    .copied()
                    .unwrap_or(0.0);
                let prev_act = prev_act_map
                    .entry(bucket)
                    .or_default()
                    .get(sym)
                    .copied()
                    .unwrap_or(0.0);
                let self_s = SelfAwarenessState {
                    current_pos: 0.0,
                    unrealized_net_ret: 0.0,
                    peak_excursion: 0.0,
                    drawdown_from_peak: 0.0,
                    log_bars_held: 0.0,
                    one_way_cost_bps: slip_bps,
                    session_time_left: norm_rem as f32,
                    ema_turnover: cur_turn,
                };
                let prev_hs = bucket_h.get(sym).map(Vec::as_slice).unwrap_or(&[]);
                let prev_wm_h = bucket_wm_h.get(sym).copied().unwrap_or([0.0; WM_DIM]);
                let inf = brain.infer_ensemble_step(
                    &raw_x,
                    self_s,
                    prev_hs,
                    &prev_wm_h,
                    option_allowed,
                );
                bucket_h.insert(sym.clone(), inf.next_hiddens);
                bucket_wm_h.insert(sym.clone(), inf.next_wm_hidden);
                let next_turn = (1.0 - turn_alpha) * cur_turn
                    + turn_alpha * (inf.action - prev_act).abs().min(2.0);
                ema_turn_map
                    .entry(bucket)
                    .or_default()
                    .insert(sym.clone(), next_turn);
                prev_act_map
                    .entry(bucket)
                    .or_default()
                    .insert(sym.clone(), inf.action);
                stepped_wm_this_bar.insert((bucket, sym.clone()));

                if inf.action.abs() < 0.070 {
                    abstained_flat_oos = abstained_flat_oos.saturating_add(1);
                    continue;
                }

                let direction = if inf.action > 0.0 {
                    TradeDirection::Long
                } else {
                    TradeDirection::Short
                };
                let (is_opt, is_vert, dte_days) = resolve_vehicle_routing_config(
                    sym,
                    bar_t.close,
                    bucket,
                    hhmm,
                    snap.vix_spot_dec,
                    inf.vehicle_prob >= 0.5,
                );
                let risk_budget_usd =
                    (equity_usd * (0.0055 + 0.0065 * inf.action.abs() as f64)).clamp(420.0, 1_250.0);
                let conf_pct = (55.0 + (inf.action.abs() as f64) * 40.0).clamp(55.0, 98.5);

                scored_actions.push((
                    inf.action.abs() as f64,
                    PendingNextBarSignal {
                        sym_id: sym_ord,
                        symbol: sym.clone(),
                        bucket,
                        brain_id: bucket.brain_id(),
                        signal_ts_sec: t_sec,
                        signal_day_key: cur_day,
                        signal_bar_ord: snap.bar_ord,
                        direction,
                        target_action: inf.action,
                        vehicle_prob: inf.vehicle_prob,
                        tail_stop_dist: inf.tail_stop_dist as f64,
                        confidence_pct: conf_pct,
                        is_option_routed: is_opt,
                        is_vertical_combo: is_vert,
                        is_penny_tier: is_penny,
                        dte_days,
                        kelly_risk_budget_usd: risk_budget_usd,
                        embedding_128: snap.embedding_128.to_vec(),
                        active_gen_id: brain.promoted_generation_id,
                    },
                ));
            }

            scored_actions
                .sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            let slots_left = max_bucket_slots.saturating_sub(bucket_active);
            for (_, sig) in scored_actions.into_iter().take(slots_left.min(2)) {
                if open_positions.len() + pending_signals.len() >= 20 || open_heat_frac >= 0.28 {
                    break;
                }
                open_heat_frac += sig.kelly_risk_budget_usd / equity_usd.max(50_000.0);
                pending_signals.push(sig);
            }
        }
    }

    for pos in open_positions {
        if let Some(bars) = symbol_bars.get(&pos.symbol)
            && let Some(last_b) = bars.last()
        {
            let cur_day = est_trading_day_key(last_b.timestamp_sec);
            if cur_day >= oos_start_day {
                let last_iv = precomputed
                    .get(&pos.symbol)
                    .and_then(|v| v.last())
                    .map(|s| exit_iv_for_bar(pos.bucket, s))
                    .unwrap_or(0.20);
                let (blotter, _) = finalize_settled_position(
                    &pos,
                    last_b.timestamp_sec,
                    last_b.close,
                    "END_OF_WINDOW_MTM".to_string(),
                    false,
                    last_iv,
                    run_mode,
                );
                oos_blotter.push(blotter);
            }
        }
    }

    OosTimelineResult {
        blotter: oos_blotter,
        abstained_flat_oos,
        oos_counterfactuals_harvested: oos_cf_harvested,
    }
}

fn summarize_blotter_slice(
    bucket_label: &str,
    run_mode: &str,
    trades: &[&BlotterTradeRecord],
    starting_equity: f64,
) -> BucketPerformanceSummary {
    let n = trades.len();
    if n == 0 {
        return BucketPerformanceSummary {
            bucket_label: bucket_label.to_string(),
            run_mode: run_mode.to_string(),
            trade_count: 0,
            wins_net: 0,
            win_rate_pct: 0.0,
            gross_profit_factor: 0.0,
            net_profit_factor: 0.0,
            gross_expectancy_r: 0.0,
            net_expectancy_r: 0.0,
            net_expectancy_usd: 0.0,
            total_gross_pnl_usd: 0.0,
            total_cost_drag_usd: 0.0,
            total_net_pnl_usd: 0.0,
            max_drawdown_usd: 0.0,
            max_drawdown_pct: 0.0,
            full_stop_rate_pct: 0.0,
            trail_step_rate_pct: 0.0,
            eod_flatten_rate_pct: 0.0,
        };
    }

    let wins_net = trades.iter().filter(|t| t.net_pnl_usd > 0.0).count();
    let gross_wins: f64 = trades
        .iter()
        .filter(|t| t.gross_pnl_usd > 0.0)
        .map(|t| t.gross_pnl_usd)
        .sum();
    let gross_losses: f64 = trades
        .iter()
        .filter(|t| t.gross_pnl_usd < 0.0)
        .map(|t| t.gross_pnl_usd.abs())
        .sum();
    let net_wins: f64 = trades
        .iter()
        .filter(|t| t.net_pnl_usd > 0.0)
        .map(|t| t.net_pnl_usd)
        .sum();
    let net_losses: f64 = trades
        .iter()
        .filter(|t| t.net_pnl_usd < 0.0)
        .map(|t| t.net_pnl_usd.abs())
        .sum();

    let gross_pf = if gross_losses > 1e-6 {
        (gross_wins / gross_losses).max(0.0)
    } else if gross_wins > 0.0 {
        99.0
    } else {
        0.0
    };
    let net_pf = if net_losses > 1e-6 {
        (net_wins / net_losses).max(0.0)
    } else if net_wins > 0.0 {
        99.0
    } else {
        0.0
    };

    let mut eq = starting_equity;
    let mut peak = eq;
    let mut max_dd_usd = 0.0_f64;
    let mut max_dd_pct = 0.0_f64;
    for tr in trades {
        eq += tr.net_pnl_usd;
        if eq > peak {
            peak = eq;
        }
        let dd = (peak - eq).max(0.0);
        if dd > max_dd_usd {
            max_dd_usd = dd;
            max_dd_pct = (dd / peak.max(1.0)) * 100.0;
        }
    }

    let total_gross_pnl_usd: f64 = trades.iter().map(|t| t.gross_pnl_usd).sum();
    let total_cost_drag_usd: f64 = trades.iter().map(|t| t.spread_and_comm_cost_usd).sum();
    let total_net_pnl_usd: f64 = trades.iter().map(|t| t.net_pnl_usd).sum();
    let gross_exp_r = trades.iter().map(|t| t.gross_r).sum::<f64>() / (n as f64);
    let net_exp_r = trades.iter().map(|t| t.net_r).sum::<f64>() / (n as f64);
    let full_stops = trades
        .iter()
        .filter(|t| t.exit_reason.contains("BACKSTOP"))
        .count();
    let neural_exits = trades
        .iter()
        .filter(|t| t.exit_reason.contains("NEURAL_POLICY_EXIT"))
        .count();
    let eod_flattens = trades
        .iter()
        .filter(|t| t.exit_reason.contains("EOD_"))
        .count();

    BucketPerformanceSummary {
        bucket_label: bucket_label.to_string(),
        run_mode: run_mode.to_string(),
        trade_count: n,
        wins_net,
        win_rate_pct: (wins_net as f64 / n as f64) * 100.0,
        gross_profit_factor: gross_pf,
        net_profit_factor: net_pf,
        gross_expectancy_r: gross_exp_r,
        net_expectancy_r: net_exp_r,
        net_expectancy_usd: total_net_pnl_usd / (n as f64),
        total_gross_pnl_usd,
        total_cost_drag_usd,
        total_net_pnl_usd,
        max_drawdown_usd: max_dd_usd,
        max_drawdown_pct: max_dd_pct,
        full_stop_rate_pct: (full_stops as f64 / n as f64) * 100.0,
        trail_step_rate_pct: (neural_exits as f64 / n as f64) * 100.0,
        eod_flatten_rate_pct: (eod_flattens as f64 / n as f64) * 100.0,
    }
}

fn summarize_vehicle_slice(
    vehicle_label: &str,
    run_mode: &str,
    trades: &[&BlotterTradeRecord],
) -> VehiclePerformanceSummary {
    let n = trades.len();
    if n == 0 {
        return VehiclePerformanceSummary {
            vehicle_label: vehicle_label.to_string(),
            run_mode: run_mode.to_string(),
            trade_count: 0,
            wins_net: 0,
            win_rate_pct: 0.0,
            gross_profit_factor: 0.0,
            net_profit_factor: 0.0,
            net_expectancy_r: 0.0,
            avg_cost_drag_per_trade_usd: 0.0,
            total_gross_pnl_usd: 0.0,
            total_cost_drag_usd: 0.0,
            total_net_pnl_usd: 0.0,
        };
    }
    let wins_net = trades.iter().filter(|t| t.net_pnl_usd > 0.0).count();
    let gw: f64 = trades
        .iter()
        .filter(|t| t.gross_pnl_usd > 0.0)
        .map(|t| t.gross_pnl_usd)
        .sum();
    let gl: f64 = trades
        .iter()
        .filter(|t| t.gross_pnl_usd < 0.0)
        .map(|t| t.gross_pnl_usd.abs())
        .sum();
    let nw: f64 = trades
        .iter()
        .filter(|t| t.net_pnl_usd > 0.0)
        .map(|t| t.net_pnl_usd)
        .sum();
    let nl: f64 = trades
        .iter()
        .filter(|t| t.net_pnl_usd < 0.0)
        .map(|t| t.net_pnl_usd.abs())
        .sum();
    let total_gross: f64 = trades.iter().map(|t| t.gross_pnl_usd).sum();
    let total_cost: f64 = trades.iter().map(|t| t.spread_and_comm_cost_usd).sum();
    let total_net: f64 = trades.iter().map(|t| t.net_pnl_usd).sum();
    VehiclePerformanceSummary {
        vehicle_label: vehicle_label.to_string(),
        run_mode: run_mode.to_string(),
        trade_count: n,
        wins_net,
        win_rate_pct: (wins_net as f64 / n as f64) * 100.0,
        gross_profit_factor: if gl > 1e-6 {
            gw / gl
        } else if gw > 0.0 {
            99.0
        } else {
            0.0
        },
        net_profit_factor: if nl > 1e-6 {
            nw / nl
        } else if nw > 0.0 {
            99.0
        } else {
            0.0
        },
        net_expectancy_r: trades.iter().map(|t| t.net_r).sum::<f64>() / (n as f64),
        avg_cost_drag_per_trade_usd: total_cost / (n as f64),
        total_gross_pnl_usd: total_gross,
        total_cost_drag_usd: total_cost,
        total_net_pnl_usd: total_net,
    }
}

fn resolve_workspace_artifact_path(rel_path: &str) -> std::path::PathBuf {
    let direct = std::path::Path::new(rel_path);
    if std::path::Path::new("Cargo.toml").exists() && std::path::Path::new("crates").exists() {
        return direct.to_path_buf();
    }
    let up2 = std::path::Path::new("../..").join(rel_path);
    if up2
        .parent()
        .is_some_and(|p| p.as_os_str().is_empty() || p.exists())
        || std::path::Path::new("../../Cargo.toml").exists()
    {
        return up2;
    }
    direct.to_path_buf()
}

/// Runs the strict future-curtain zero-lookahead invariance verification (`|Δa_t| == 0.0`, `|Δv_t| == 0.0`
/// when scrambling all bars `> t`) and returns `(a_orig, a_scram, delta_a, delta_v)`.
pub fn verify_future_curtain_zero_lookahead_invariance() -> (f32, f32, f32, f32) {
    let base_ts: i64 = 1_762_957_800; // 09:30 ET session bar anchor
    let mut bars_orig = Vec::with_capacity(80);
    for m in 0..80 {
        let ts = base_ts + m * 300;
        let px = 500.0 + (m as f64) * 0.15 + ((m % 5) as f64) * 0.05;
        bars_orig.push(RealExchangeOhlcvBar {
            timestamp_sec: ts,
            open: px,
            high: px + 0.45,
            low: px - 0.35,
            close: px + 0.12,
            volume: 50_000 + (m as u64) * 150,
        });
    }
    let mut bars_scrambled = bars_orig.clone();
    let probe_idx = 40usize;
    for b in &mut bars_scrambled[probe_idx + 1..] {
        b.open *= 4.3;
        b.high *= 6.1;
        b.low *= 0.15;
        b.close *= 4.8;
        b.volume *= 137;
    }

    let mut sym_orig = HashMap::new();
    sym_orig.insert("SPY".to_string(), bars_orig);
    let mut sym_scram = HashMap::new();
    sym_scram.insert("SPY".to_string(), bars_scrambled);
    let empty_map = HashMap::new();

    let pre_orig = precompute_causal_timeline_physics(
        &sym_orig,
        &empty_map,
        &empty_map,
        &[],
        &[],
        &[],
        &empty_map,
        BrainHorizonId::Brain02IntradayDayTrade,
        ProofBucket::Scalp0Dte5m.bars_per_year(),
    );
    let pre_scram = precompute_causal_timeline_physics(
        &sym_scram,
        &empty_map,
        &empty_map,
        &[],
        &[],
        &[],
        &empty_map,
        BrainHorizonId::Brain02IntradayDayTrade,
        ProofBucket::Scalp0Dte5m.bars_per_year(),
    );

    let brain = PureAutonomousRlBrain::new_for_horizon("CURTAIN_VERIFY", 777);
    let mut h_orig = Vec::new();
    let mut h_scram = Vec::new();
    let mut wm_h_orig = [0.0_f32; WM_DIM];
    let mut wm_h_scram = [0.0_f32; WM_DIM];
    let mut a_orig = 0.0_f32;
    let mut a_scram = 0.0_f32;
    let mut v_orig = 0.0_f32;
    let mut v_scram = 0.0_f32;

    for idx in 0..=probe_idx {
        let x_o = pre_orig["SPY"][idx].to_raw_180_vector(0.5, 0.0, false);
        let x_s = pre_scram["SPY"][idx].to_raw_180_vector(0.5, 0.0, false);
        let self_s = SelfAwarenessState {
            one_way_cost_bps: 5.0,
            session_time_left: 0.5,
            ..Default::default()
        };
        let out_o = brain.infer_ensemble_step(&x_o, self_s, &h_orig, &wm_h_orig, true);
        let out_s = brain.infer_ensemble_step(&x_s, self_s, &h_scram, &wm_h_scram, true);
        h_orig = out_o.next_hiddens;
        h_scram = out_s.next_hiddens;
        wm_h_orig = out_o.next_wm_hidden;
        wm_h_scram = out_s.next_wm_hidden;
        a_orig = out_o.action;
        a_scram = out_s.action;
        v_orig = out_o.vehicle_prob;
        v_scram = out_s.vehicle_prob;
    }

    (
        a_orig,
        a_scram,
        (a_orig - a_scram).abs(),
        (v_orig - v_scram).abs(),
    )
}

pub fn execute_walkforward_proof_on_datasets(
    _onnx_engine: &OnnxMultiBrainEngine,
    ds: &ProofDatasets,
    starting_equity_usd: f64,
    blotter_csv_path: &Path,
    report_md_path: &Path,
) -> Result<WalkforwardProofReport, String> {
    let _ = build_causal_universe_snapshots_at_timestamp;
    let _ = lookup_bar_le_timestamp;

    let t_run_start = std::time::Instant::now();
    let (m1_ts, d1_days) = collect_sorted_timestamps_and_days(&ds.m1_bars, true);
    let d1_warmup = (d1_days.len() / 2)
        .clamp(2, 10)
        .min(d1_days.len().saturating_sub(2).max(1));
    let d1_oos = d1_days.len().saturating_sub(d1_warmup);
    let d1_warmup_start_day = d1_days.first().copied().unwrap_or(0);
    let d1_oos_start_day = d1_days.get(d1_warmup).copied().unwrap_or(i64::MAX);
    let d1_warmup_cutoff_ts = m1_ts
        .iter()
        .copied()
        .rfind(|&ts| est_trading_day_key(ts) < d1_oos_start_day)
        .unwrap_or(0);

    let (m5_ts, d2_m5_days) = collect_sorted_timestamps_and_days(&ds.m5_bars, true);
    let (h1_ts, d2_h1_days) = collect_sorted_timestamps_and_days(&ds.h1_bars, true);
    let m5_warmup = 24.min(d2_m5_days.len().saturating_sub(5).max(1));
    let m5_oos = d2_m5_days.len().saturating_sub(m5_warmup);
    let d2_warmup_start_day = d2_m5_days.first().copied().unwrap_or(0);
    let h1_warmup_start_day = d2_h1_days.first().copied().unwrap_or(d2_warmup_start_day);
    let d2_oos_start_day = d2_m5_days.get(m5_warmup).copied().unwrap_or(i64::MAX);
    let m5_warmup_cutoff_ts = m5_ts
        .iter()
        .copied()
        .rfind(|&ts| est_trading_day_key(ts) < d2_oos_start_day)
        .unwrap_or(0);
    let h1_warmup_cutoff_ts = h1_ts
        .iter()
        .copied()
        .rfind(|&ts| est_trading_day_key(ts) < d2_oos_start_day)
        .unwrap_or(m5_warmup_cutoff_ts);

    let m1_total_bars: usize = ds.m1_bars.values().map(|v| v.len()).sum();

    let pre_m1 = precompute_causal_timeline_physics(
        &ds.m1_bars,
        &ds.h1_bars,
        &ds.d1_bars,
        &ds.vix9d_bars,
        &ds.vix_bars,
        &ds.vix3m_bars,
        if ds.breadth_m1.is_empty() {
            &ds.breadth_m5
        } else {
            &ds.breadth_m1
        },
        BrainHorizonId::Brain02IntradayDayTrade,
        ProofBucket::Scalp0Dte1m.bars_per_year(),
    );
    let pre_m5 = precompute_causal_timeline_physics(
        &ds.m5_bars,
        &ds.h1_bars,
        &ds.d1_bars,
        &ds.vix9d_bars,
        &ds.vix_bars,
        &ds.vix3m_bars,
        &ds.breadth_m5,
        BrainHorizonId::Brain02IntradayDayTrade,
        ProofBucket::Scalp0Dte5m.bars_per_year(),
    );
    let pre_h1_b3 = precompute_causal_timeline_physics(
        &ds.h1_bars,
        &ds.h1_bars,
        &ds.d1_bars,
        &ds.vix9d_bars,
        &ds.vix_bars,
        &ds.vix3m_bars,
        if ds.breadth_h1.is_empty() {
            &ds.breadth_m5
        } else {
            &ds.breadth_h1
        },
        BrainHorizonId::Brain03ShortTermSwing,
        ProofBucket::Brain03ShortSwing.bars_per_year(),
    );
    let pre_h1_b4 = precompute_causal_timeline_physics(
        &ds.h1_bars,
        &ds.h1_bars,
        &ds.d1_bars,
        &ds.vix9d_bars,
        &ds.vix_bars,
        &ds.vix3m_bars,
        if ds.breadth_h1.is_empty() {
            &ds.breadth_m5
        } else {
            &ds.breadth_h1
        },
        BrainHorizonId::Brain04CoreSwing,
        ProofBucket::Brain04CoreSwing.bars_per_year(),
    );
    eprintln!(
        "[prove_perfection] Causal timeline physics precomputed in {:.2}s (D1={}w/{}o days, D2={}w/{}o days)",
        t_run_start.elapsed().as_secs_f64(),
        d1_warmup,
        d1_oos,
        m5_warmup,
        m5_oos
    );

    let all_buckets = [
        ProofBucket::Scalp0Dte1m,
        ProofBucket::Scalp0Dte5m,
        ProofBucket::Brain01PennyDay,
        ProofBucket::Brain02LargeCapDay,
        ProofBucket::Brain03ShortSwing,
        ProofBucket::Brain04CoreSwing,
    ];
    let mut warmup_brains: BTreeMap<ProofBucket, PureAutonomousRlBrain> = BTreeMap::new();
    let mut warmup_telemetry: Vec<WarmupTrainingTelemetry> = Vec::with_capacity(all_buckets.len());
    let mut promotion_ledger: Vec<GenerationPromotionLedgerRow> = Vec::new();
    let mut warmup_cf_total = 0usize;

    for bucket in all_buckets {
        let t_b = std::time::Instant::now();
        let mut brain = PureAutonomousRlBrain::new_for_horizon(bucket.label(), bucket.seed_base());
        let (bars_ref, pre_ref, min_day, cutoff_ts, pretrain_m5) = match bucket {
            ProofBucket::Scalp0Dte1m => (
                &ds.m1_bars,
                &pre_m1,
                d1_warmup_start_day,
                d1_warmup_cutoff_ts,
                Some((&ds.m5_bars, &pre_m5, d2_warmup_start_day)),
            ),
            ProofBucket::Scalp0Dte5m
            | ProofBucket::Brain01PennyDay
            | ProofBucket::Brain02LargeCapDay => (
                &ds.m5_bars,
                &pre_m5,
                d2_warmup_start_day,
                m5_warmup_cutoff_ts,
                None,
            ),
            ProofBucket::Brain03ShortSwing => (
                &ds.h1_bars,
                &pre_h1_b3,
                h1_warmup_start_day,
                h1_warmup_cutoff_ts,
                None,
            ),
            ProofBucket::Brain04CoreSwing => (
                &ds.h1_bars,
                &pre_h1_b4,
                h1_warmup_start_day,
                h1_warmup_cutoff_ts,
                None,
            ),
        };
        let (tel, bars_trained) = train_warmup_with_causal_pbt(
            &mut brain,
            bucket,
            bars_ref,
            pre_ref,
            pretrain_m5,
            min_day,
            cutoff_ts,
            starting_equity_usd,
            &mut promotion_ledger,
        );
        eprintln!(
            "[prove_perfection] Warmup {} finished in {:.2}s | OOF Val: {} trades, PF={:.2}, NetPnL=${:+.2}, PBT={}",
            bucket.label(),
            t_b.elapsed().as_secs_f64(),
            tel.best_val_trades,
            tel.best_val_net_pf,
            tel.best_val_net_pnl,
            tel.pbt_generations
        );
        warmup_cf_total += bars_trained;
        brain.brain_key = bucket.runtime_brain_key().to_string();
        warmup_telemetry.push(tel);
        warmup_brains.insert(bucket, brain);
    }

    let run_full_oos = |mode: ProofRunMode,
                        brains: &mut BTreeMap<ProofBucket, PureAutonomousRlBrain>,
                        promo_log: &mut Vec<GenerationPromotionLedgerRow>|
     -> (Vec<BlotterTradeRecord>, usize, usize, u64) {
        let mut trades = Vec::new();
        let mut abstained = 0usize;
        let mut oos_cf = 0usize;

        let res_m1 = run_oos_timeline_with_pure_rl_brains(
            mode,
            &ds.m1_bars,
            &pre_m1,
            &[ProofBucket::Scalp0Dte1m],
            brains,
            d1_warmup_start_day,
            true,
            d1_oos_start_day,
            starting_equity_usd,
            promo_log,
        );
        trades.extend(res_m1.blotter);
        abstained += res_m1.abstained_flat_oos;
        oos_cf += res_m1.oos_counterfactuals_harvested;

        let res_m5 = run_oos_timeline_with_pure_rl_brains(
            mode,
            &ds.m5_bars,
            &pre_m5,
            &[
                ProofBucket::Scalp0Dte5m,
                ProofBucket::Brain01PennyDay,
                ProofBucket::Brain02LargeCapDay,
            ],
            brains,
            d2_warmup_start_day,
            true,
            d2_oos_start_day,
            starting_equity_usd,
            promo_log,
        );
        trades.extend(res_m5.blotter);
        abstained += res_m5.abstained_flat_oos;
        oos_cf += res_m5.oos_counterfactuals_harvested;

        let res_h1_b3 = run_oos_timeline_with_pure_rl_brains(
            mode,
            &ds.h1_bars,
            &pre_h1_b3,
            &[ProofBucket::Brain03ShortSwing],
            brains,
            h1_warmup_start_day,
            true,
            d2_oos_start_day,
            starting_equity_usd,
            promo_log,
        );
        trades.extend(res_h1_b3.blotter);
        abstained += res_h1_b3.abstained_flat_oos;
        oos_cf += res_h1_b3.oos_counterfactuals_harvested;

        let res_h1_b4 = run_oos_timeline_with_pure_rl_brains(
            mode,
            &ds.h1_bars,
            &pre_h1_b4,
            &[ProofBucket::Brain04CoreSwing],
            brains,
            h1_warmup_start_day,
            true,
            d2_oos_start_day,
            starting_equity_usd,
            promo_log,
        );
        trades.extend(res_h1_b4.blotter);
        abstained += res_h1_b4.abstained_flat_oos;
        oos_cf += res_h1_b4.oos_counterfactuals_harvested;

        let total_online_updates: u64 = brains.values().map(|b| b.online_updates_count).sum();
        (trades, abstained, oos_cf, total_online_updates)
    };

    let t_a = std::time::Instant::now();
    let mut dummy_promo = Vec::new();
    let mut brains_a = warmup_brains.clone();
    let (trades_a, _, _, _) =
        run_full_oos(ProofRunMode::RunAFrozenWarmupMl, &mut brains_a, &mut dummy_promo);
    eprintln!(
        "[prove_perfection] RUN_A_FROZEN_WARMUP_ML finished in {:.2}s ({} trades)",
        t_a.elapsed().as_secs_f64(),
        trades_a.len()
    );

    let t_b_oos = std::time::Instant::now();
    let mut brains_b = warmup_brains;
    let (trades_b, abstained_b, oos_cf_b, online_updates_b) =
        run_full_oos(ProofRunMode::RunBOnlineRl, &mut brains_b, &mut promotion_ledger);
    eprintln!(
        "[prove_perfection] RUN_B_ONLINE_RL finished in {:.2}s ({} trades, total elapsed {:.2}s)",
        t_b_oos.elapsed().as_secs_f64(),
        trades_b.len(),
        t_run_start.elapsed().as_secs_f64()
    );

    let autopsy_telemetry: Vec<BucketAutopsyTelemetry> = all_buckets
        .iter()
        .filter_map(|b| {
            brains_b.get(b).map(|br| {
                let mut tel = br.autopsy_telemetry.clone();
                tel.brain_label = b.label().to_string();
                tel.promoted_generations = br.promoted_generation_id;
                tel
            })
        })
        .collect();

    // Atomically persist promoted `RUN_B` champion brains to `data/live_runtime_state.json`
    // (`data/ledger_state.json` benchmark fixture is NEVER modified!)
    {
        let rt_path = resolve_workspace_artifact_path(LIVE_RUNTIME_STATE_PATH);
        let bench_path = resolve_workspace_artifact_path(BENCHMARK_LEDGER_FIXTURE_PATH);
        let ledger = RealTimeLedgerEngine::new();
        let _ = ledger.load_runtime_or_seed_benchmark(&rt_path, &bench_path);
        let persisted_suite: Vec<PureAutonomousRlBrain> = brains_b.into_values().collect();
        ledger.replace_pure_rl_brains(persisted_suite);
        let _ = ledger.save_runtime_state_atomic(&rt_path);
    }

    if let Some(parent) = blotter_csv_path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let mut csv_buf = String::from(
        "run_mode,trade_id,symbol,brain_id,tier,timeframe,signal_time_est,entry_time_est,exit_time_est,direction,structure,entry_underlying_px,stop_px,t1_px,exit_underlying_px,gross_r,spread_and_comm_cost_usd,net_pnl_usd,net_r,exit_reason,active_gen_id,scale_gear,vehicle,arm_label\n",
    );
    for r in trades_a.iter().chain(trades_b.iter()) {
        csv_buf.push_str(&format!(
            "{},{},{},{},{},{},{},{},{},{},{},{:.5},{:.5},{:.5},{:.5},{:.4},{:.2},{:.2},{:.4},{},{},{},{},{}\n",
            r.run_mode,
            r.trade_id,
            r.symbol,
            r.brain_id,
            r.tier,
            r.timeframe,
            r.signal_time_est,
            r.entry_time_est,
            r.exit_time_est,
            r.direction,
            r.structure,
            r.entry_underlying_px,
            r.stop_px,
            r.t1_px,
            r.exit_underlying_px,
            r.gross_r,
            r.spread_and_comm_cost_usd,
            r.net_pnl_usd,
            r.net_r,
            r.exit_reason,
            r.active_gen_id,
            r.scale_gear,
            r.vehicle,
            r.arm_label,
        ));
    }
    std::fs::write(blotter_csv_path, &csv_buf).map_err(|e| e.to_string())?;

    let v_shares_a: Vec<&BlotterTradeRecord> =
        trades_a.iter().filter(|t| t.vehicle == "V_SHARES").collect();
    let v_option_a: Vec<&BlotterTradeRecord> =
        trades_a.iter().filter(|t| t.vehicle == "V_OPTION").collect();
    let v_shares_b: Vec<&BlotterTradeRecord> =
        trades_b.iter().filter(|t| t.vehicle == "V_SHARES").collect();
    let v_option_b: Vec<&BlotterTradeRecord> =
        trades_b.iter().filter(|t| t.vehicle == "V_OPTION").collect();

    let vehicle_split_run_a = vec![
        summarize_vehicle_slice("V_SHARES", ProofRunMode::RunAFrozenWarmupMl.as_str(), &v_shares_a),
        summarize_vehicle_slice("V_OPTION", ProofRunMode::RunAFrozenWarmupMl.as_str(), &v_option_a),
    ];
    let vehicle_split_run_b = vec![
        summarize_vehicle_slice("V_SHARES", ProofRunMode::RunBOnlineRl.as_str(), &v_shares_b),
        summarize_vehicle_slice("V_OPTION", ProofRunMode::RunBOnlineRl.as_str(), &v_option_b),
    ];

    let mut comparisons = Vec::with_capacity(all_buckets.len());
    for bucket in all_buckets {
        let a_slice: Vec<&BlotterTradeRecord> =
            trades_a.iter().filter(|t| t.bucket == bucket).collect();
        let b_slice: Vec<&BlotterTradeRecord> =
            trades_b.iter().filter(|t| t.bucket == bucket).collect();
        let run_a = summarize_blotter_slice(
            bucket.label(),
            ProofRunMode::RunAFrozenWarmupMl.as_str(),
            &a_slice,
            starting_equity_usd,
        );
        let run_b = summarize_blotter_slice(
            bucket.label(),
            ProofRunMode::RunBOnlineRl.as_str(),
            &b_slice,
            starting_equity_usd,
        );
        comparisons.push(BucketAbComparison {
            bucket_label: bucket.label().to_string(),
            delta_trade_count: (run_b.trade_count as isize) - (run_a.trade_count as isize),
            delta_win_rate_pct: run_b.win_rate_pct - run_a.win_rate_pct,
            delta_net_pf: run_b.net_profit_factor - run_a.net_profit_factor,
            delta_net_expectancy_r: run_b.net_expectancy_r - run_a.net_expectancy_r,
            delta_net_pnl_usd: run_b.total_net_pnl_usd - run_a.total_net_pnl_usd,
            delta_max_dd_usd: run_b.max_drawdown_usd - run_a.max_drawdown_usd,
            traps_learned_or_blocked: promotion_ledger
                .iter()
                .filter(|r| r.brain_label == bucket.label() && r.promoted)
                .count(),
            run_a,
            run_b,
        });
    }

    let d1_pnl_a: f64 = trades_a
        .iter()
        .filter(|t| t.bucket == ProofBucket::Scalp0Dte1m)
        .map(|t| t.net_pnl_usd)
        .sum();
    let d1_pnl_b: f64 = trades_b
        .iter()
        .filter(|t| t.bucket == ProofBucket::Scalp0Dte1m)
        .map(|t| t.net_pnl_usd)
        .sum();
    let d2_pnl_a: f64 = trades_a
        .iter()
        .filter(|t| t.bucket != ProofBucket::Scalp0Dte1m)
        .map(|t| t.net_pnl_usd)
        .sum();
    let d2_pnl_b: f64 = trades_b
        .iter()
        .filter(|t| t.bucket != ProofBucket::Scalp0Dte1m)
        .map(|t| t.net_pnl_usd)
        .sum();

    let all_a: Vec<&BlotterTradeRecord> = trades_a.iter().collect();
    let all_b: Vec<&BlotterTradeRecord> = trades_b.iter().collect();
    let portfolio_run_a = summarize_blotter_slice(
        "ALL_BUCKETS_COMBINED",
        ProofRunMode::RunAFrozenWarmupMl.as_str(),
        &all_a,
        starting_equity_usd,
    );
    let portfolio_run_b = summarize_blotter_slice(
        "ALL_BUCKETS_COMBINED",
        ProofRunMode::RunBOnlineRl.as_str(),
        &all_b,
        starting_equity_usd,
    );

    let report = WalkforwardProofReport {
        generated_at_utc: Utc::now().to_rfc3339(),
        dataset1_trading_days_total: d1_days.len(),
        dataset1_oos_days: d1_oos,
        dataset1_m1_bars_loaded: m1_total_bars,
        dataset2_universe_symbols_loaded: ds.m5_bars.len(),
        dataset2_trading_days_total: d2_m5_days.len(),
        dataset2_warmup_days: m5_warmup,
        dataset2_oos_days: m5_oos,
        blotter_csv_path: blotter_csv_path.to_string_lossy().to_string(),
        report_md_path: report_md_path.to_string_lossy().to_string(),
        oos_trades_run_a: trades_a.len(),
        oos_trades_run_b: trades_b.len(),
        traps_blocked_in_run_b: abstained_b,
        traps_quarantined_in_run_b: online_updates_b as usize,
        warmup_counterfactuals_harvested: warmup_cf_total,
        oos_counterfactuals_harvested: oos_cf_b,
        online_rl_grad_updates: online_updates_b,
        warmup_telemetry,
        autopsy_telemetry,
        promotion_ledger,
        dataset1_delta_net_pnl_usd: d1_pnl_b - d1_pnl_a,
        dataset2_delta_net_pnl_usd: d2_pnl_b - d2_pnl_a,
        vehicle_split_run_a,
        vehicle_split_run_b,
        bucket_comparisons: comparisons,
        portfolio_run_a,
        portfolio_run_b,
    };

    write_proof_report_markdown(&report, report_md_path, &trades_b)?;
    Ok(report)
}

fn write_proof_report_markdown(
    rep: &WalkforwardProofReport,
    report_path: &Path,
    trades_b: &[BlotterTradeRecord],
) -> Result<(), String> {
    let mut md = String::new();
    md.push_str("# ZERO-CHEAT END-TO-END DIFFERENTIABLE BPTT + CLOSED-LOOP LOSS-AUTOPSY RL PROOF REPORT\n\n");
    md.push_str(&format!(
        "- **Generated At (UTC):** `{}`\n",
        rep.generated_at_utc
    ));
    md.push_str(&format!(
        "- **Dataset 1 (`1m` Intraday + Causal `5m` Session Pre-Training):** `{}` total `1m` trading days (`{}` OOS days), `{}` real `1m` OHLCV bars loaded\n",
        rep.dataset1_trading_days_total, rep.dataset1_oos_days, rep.dataset1_m1_bars_loaded
    ));
    md.push_str(&format!(
        "- **Dataset 2 (`5m` & `1h` 60-Ticker Universe):** `{}` real US symbols across `{}` total trading days (`{}` Warmup Calibration -> `{}` Strictly Out-of-Sample)\n",
        rep.dataset2_universe_symbols_loaded,
        rep.dataset2_trading_days_total,
        rep.dataset2_warmup_days,
        rep.dataset2_oos_days
    ));
    md.push_str(
        "- **Zero-Rule Neural Architecture (`src/pure_rl_brain.rs`):** Stage A Multi-Horizon Physics World Model (`1, 3, 8, 24`-bar Pinball Quantile Loss + `ln(Range/RT_Cost)`) + Stage B `K=5` Orthogonal Specialist Policy Ensemble (`D=64`, `[h_ln, wm_fused, self_state, pos_interact]` 92-d skip-head) with Localized Boundary Surrogate Deadzone (`sech^2((|z| - tau) / temp)`), Unconstrained Differentiable Log-Space Cost Sensitivity (`k_theta = exp(log_k_theta)` via analytical `dL_net / d log_k_theta`), Turnover-Amortized Opportunity Critic (`Q_hold(x_t, s_t) - Q_flat(x_t, s_t)`), Purged `T_holdout` Cross-Sectional LCB Promotion Gate, and `M=24` `rayon` PBT + Multiplicative-Weights Routing.\n",
    );
    md.push_str(
        "- **Anti-Cheat Execution Physics:** Zero `> t` lookahead, next-bar `t+1` open fills with `5/12/35 bps` adverse equity slippage, `09:35–15:15 ET` flatten `15:50 ET` (`SCALP_0DTE`), `09:35–15:30 ET` flatten `15:55 ET` (`DAY`), skew-adjusted BSM + `^VIX9D`/`^VIX` IV + `$0.03–$0.08`/sh/leg spread + `$0.65`/contract/leg RT commission.\n",
    );
    md.push_str(&format!(
        "- **BPTT & Closed-Loop Loss-Autopsy Activity:** `{}` Causal Warmup Episode Bars | `{}` OOS Causal Episode Bars | `{}` Online EOD RL & Routing Cycles (`RUN_B_ONLINE_RL`) | `{}` Autonomous `FLAT` Abstentions.\n\n",
        rep.warmup_counterfactuals_harvested,
        rep.oos_counterfactuals_harvested,
        rep.online_rl_grad_updates,
        rep.traps_blocked_in_run_b
    ));

    md.push_str("## Causal Walk-Forward Inner PBT Calibration (`t <= T_oos_start`)\n\n");
    md.push_str("| Brain / Horizon | Population | Session Episodes | PBT Gens | Best Inner-Val Trades | Best Inner-Val Net PF | Best Inner-Val Net PnL ($) |\n");
    md.push_str("|---|---:|---:|---:|---:|---:|---:|\n");
    for t in &rep.warmup_telemetry {
        md.push_str(&format!(
            "| {} | {} | {} | {} | {} | **{:.2}** | ${:+.2} |\n",
            t.brain_label,
            t.population_size,
            t.train_sequences,
            t.pbt_generations,
            t.best_val_trades,
            t.best_val_net_pf,
            t.best_val_net_pnl,
        ));
    }

    md.push_str("\n## Executive Scorecard: `RUN_A_FROZEN_WARMUP_ML` vs `RUN_B_ONLINE_RL` (Gross vs Net Post-Cost)\n\n");
    md.push_str("| Bucket | Run Mode | Trades | Win% | Gross PF | Net Post-Cost PF | Net Exp (R) | Net Exp ($) | Gross PnL ($) | Cost Drag ($) | Net PnL ($) | Max DD ($ / %) | Backstop% | Neural-Exit% | EOD Flatten% |\n");
    md.push_str("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n");

    let push_row = |buf: &mut String, s: &BucketPerformanceSummary| {
        buf.push_str(&format!(
            "| {} | `{}` | {} | {:.1}% | {:.2} | **{:.2}** | {:+.3}R | ${:+.2} | ${:+.2} | ${:.2} | **${:+.2}** | ${:.2} ({:.2}%) | {:.1}% | {:.1}% | {:.1}% |\n",
            s.bucket_label,
            s.run_mode,
            s.trade_count,
            s.win_rate_pct,
            s.gross_profit_factor,
            s.net_profit_factor,
            s.net_expectancy_r,
            s.net_expectancy_usd,
            s.total_gross_pnl_usd,
            s.total_cost_drag_usd,
            s.total_net_pnl_usd,
            s.max_drawdown_usd,
            s.max_drawdown_pct,
            s.full_stop_rate_pct,
            s.trail_step_rate_pct,
            s.eod_flatten_rate_pct
        ));
    };

    for c in &rep.bucket_comparisons {
        push_row(&mut md, &c.run_a);
        push_row(&mut md, &c.run_b);
    }
    push_row(&mut md, &rep.portfolio_run_a);
    push_row(&mut md, &rep.portfolio_run_b);

    let tier_summary = |name: &str, pred: fn(ProofBucket) -> bool| -> BucketPerformanceSummary {
        let slice: Vec<&BlotterTradeRecord> = trades_b.iter().filter(|t| pred(t.bucket)).collect();
        summarize_blotter_slice(name, "RUN_B_ONLINE_RL", &slice, 100_000.0)
    };
    let scalp_b = tier_summary("TIER: SCALP_0DTE", |b| {
        matches!(b, ProofBucket::Scalp0Dte1m | ProofBucket::Scalp0Dte5m)
    });
    let day_b = tier_summary("TIER: DAY (BRAIN-01 + BRAIN-02)", |b| {
        matches!(b, ProofBucket::Brain01PennyDay | ProofBucket::Brain02LargeCapDay)
    });
    let swing_b = tier_summary("TIER: SWING (BRAIN-03 + BRAIN-04)", |b| {
        matches!(b, ProofBucket::Brain03ShortSwing | ProofBucket::Brain04CoreSwing)
    });
    md.push_str("\n## Horizon Tier Rollups (`RUN_B_ONLINE_RL`)\n\n");
    md.push_str("| Tier | Run Mode | Trades | Win% | Gross PF | Net Post-Cost PF | Net Exp (R) | Net Exp ($) | Gross PnL ($) | Cost Drag ($) | Net PnL ($) | Max DD ($ / %) | Backstop% | Neural-Exit% | EOD Flatten% |\n");
    md.push_str("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    push_row(&mut md, &scalp_b);
    push_row(&mut md, &day_b);
    push_row(&mut md, &swing_b);

    md.push_str("\n## Factored Vehicle Performance Split (`V_SHARES` vs `V_OPTION`)\n\n");
    md.push_str("| Run Mode | Vehicle | Trades | Win% | Gross PF | Net Post-Cost PF | Net Exp (R) | Avg Cost/Trade ($) | Total Gross PnL ($) | Total Cost Drag ($) | Total Net PnL ($) |\n");
    md.push_str("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    for v in rep
        .vehicle_split_run_a
        .iter()
        .chain(rep.vehicle_split_run_b.iter())
    {
        md.push_str(&format!(
            "| `{}` | **{}** | {} | {:.1}% | {:.2} | **{:.2}** | {:+.3}R | ${:.2} | ${:+.2} | ${:.2} | **${:+.2}** |\n",
            v.run_mode,
            v.vehicle_label,
            v.trade_count,
            v.win_rate_pct,
            v.gross_profit_factor,
            v.net_profit_factor,
            v.net_expectancy_r,
            v.avg_cost_drag_per_trade_usd,
            v.total_gross_pnl_usd,
            v.total_cost_drag_usd,
            v.total_net_pnl_usd
        ));
    }

    md.push_str("\n## A/B Online Learning Delta (`RUN_B_ONLINE_RL` minus `RUN_A_FROZEN_WARMUP_ML`)\n\n");
    md.push_str("| Bucket | Δ Trades | Δ Win% | Δ Net PF | Δ Net Exp (R) | Δ Net PnL ($) | Δ Max DD ($) |\n");
    md.push_str("|---|---:|---:|---:|---:|---:|---:|\n");
    for c in &rep.bucket_comparisons {
        md.push_str(&format!(
            "| {} | {:+} | {:+.1}% | {:+.2} | {:+.3}R | **${:+.2}** | ${:+.2} |\n",
            c.bucket_label,
            c.delta_trade_count,
            c.delta_win_rate_pct,
            c.delta_net_pf,
            c.delta_net_expectancy_r,
            c.delta_net_pnl_usd,
            c.delta_max_dd_usd
        ));
    }
    md.push_str(&format!(
        "| **DATASET_1_1M_TOTAL** | — | — | — | — | **${:+.2}** | — |\n",
        rep.dataset1_delta_net_pnl_usd
    ));
    md.push_str(&format!(
        "| **DATASET_2_5M_1H_TOTAL** | — | — | — | — | **${:+.2}** | — |\n",
        rep.dataset2_delta_net_pnl_usd
    ));
    md.push_str(&format!(
        "| **ALL_BUCKETS_COMBINED** | **{:+}** | **{:+.1}%** | **{:+.2}** | **{:+.3}R** | **${:+.2}** | **${:+.2}** |\n",
        (rep.portfolio_run_b.trade_count as isize) - (rep.portfolio_run_a.trade_count as isize),
        rep.portfolio_run_b.win_rate_pct - rep.portfolio_run_a.win_rate_pct,
        rep.portfolio_run_b.net_profit_factor - rep.portfolio_run_a.net_profit_factor,
        rep.portfolio_run_b.net_expectancy_r - rep.portfolio_run_a.net_expectancy_r,
        rep.portfolio_run_b.total_net_pnl_usd - rep.portfolio_run_a.total_net_pnl_usd,
        rep.portfolio_run_b.max_drawdown_usd - rep.portfolio_run_a.max_drawdown_usd
    ));

    md.push_str("\n## Loss-Autopsy -> Re-Test -> Generation Promotion Ledger\n\n");
    md.push_str("| Gen ID | Phase | Brain | Red Trades Audited | Entry vs Exit Fault Split | Saliency (`|dL/dx|` / `|dL/dh|`) | Re-Test Net PF (Before -> After) | Red Loss Reduced ($) | Winners Preserved | Outcome |\n");
    md.push_str("|---:|---|---|---:|---|---|---|---:|---|---|\n");
    // Emit all PROMOTED rows plus representative ROLLED_BACK checks so table stays clean and complete
    let mut rolled_shown_per_brain: HashMap<String, usize> = HashMap::new();
    for row in &rep.promotion_ledger {
        if !row.promoted {
            let cnt = rolled_shown_per_brain
                .entry(row.brain_label.clone())
                .or_insert(0);
            if *cnt >= 12 {
                continue;
            }
            *cnt += 1;
        }
        let status_str = if row.promoted {
            "**PROMOTED**"
        } else {
            "`ROLLED_BACK`"
        };
        md.push_str(&format!(
            "| {} | `{}` | {} | {} | `{} Entry / {} Exit` | `{:.4} / {:.4}` | `{:.2} -> {:.2}` | ${:+.2} | `{}` | {} |\n",
            row.gen_id,
            row.phase_label,
            row.brain_label,
            row.red_trades_audited,
            row.entry_faults,
            row.exit_faults,
            row.mean_saliency_dx,
            row.mean_saliency_dh,
            row.pf_before,
            row.pf_after,
            row.red_loss_reduced_usd,
            row.winners_preserved_desc,
            status_str
        ));
    }

    if !rep.autopsy_telemetry.is_empty() {
        md.push_str("\n## Autonomous 4-Vector Loss-Autopsy Attribution (`RUN_B_ONLINE_RL`)\n\n");
        md.push_str("| Bucket | Friction Inversion | Holding / EOD Decay | Vehicle Regret | Paralysis Regret | Flat Windows Replayed | Cost Sens (Before -> After) | Promoted Gens |\n");
        md.push_str("|---|---:|---:|---:|---:|---:|---|---:|\n");
        for tel in &rep.autopsy_telemetry {
            md.push_str(&format!(
                "| {} | {} | {} | {} | {} | {} | `{:.3} -> {:.3}` | {} |\n",
                tel.brain_label,
                tel.friction_inversion_events,
                tel.holding_eod_decay_events,
                tel.vehicle_regret_events,
                tel.deadzone_paralysis_events,
                tel.flat_regret_windows_replayed,
                tel.cost_sensitivity_before,
                tel.cost_sensitivity_after,
                tel.promoted_generations
            ));
        }
    }

    if let Some(parent) = report_path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    std::fs::write(report_path, &md).map_err(|e| e.to_string())?;
    Ok(())
}

pub async fn run_chronological_hard_evidence_proof(
    onnx_engine: Arc<OnnxMultiBrainEngine>,
    adapter: &YahooFinanceLiveAdapter,
    starting_equity_usd: f64,
) -> Result<WalkforwardProofReport, String> {
    let ds = fetch_all_proof_datasets(adapter).await;
    let blotter_path = resolve_workspace_artifact_path(WALKFORWARD_BLOTTER_CSV_PATH);
    let report_path = resolve_workspace_artifact_path(PROOF_REPORT_MD_PATH);
    execute_walkforward_proof_on_datasets(
        &onnx_engine,
        &ds,
        starting_equity_usd,
        &blotter_path,
        &report_path,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn est_unix(y: i32, m: u32, d: u32, hh: u32, mm: u32) -> i64 {
        let est = chrono::FixedOffset::west_opt(5 * 3600).unwrap();
        est.with_ymd_and_hms(y, m, d, hh, mm, 0)
            .single()
            .expect("valid EST timestamp")
            .timestamp()
    }

    fn make_bar(
        ts_sec: i64,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: u64,
    ) -> RealExchangeOhlcvBar {
        RealExchangeOhlcvBar {
            timestamp_sec: ts_sec,
            open,
            high,
            low,
            close,
            volume,
        }
    }

    #[test]
    fn test_anti_cheat_slippage_tiers_option_fees_and_pessimistic_backstop() {
        assert_eq!(explicit_one_way_slippage_bps("SPY", 500.0, false), 5.0);
        assert_eq!(explicit_one_way_slippage_bps("NVDA", 140.0, false), 5.0);
        assert_eq!(explicit_one_way_slippage_bps("PLTR", 45.0, false), 12.0);
        assert_eq!(explicit_one_way_slippage_bps("MULN", 1.25, true), 35.0);

        let mut pos = ActiveSimulatedPosition {
            trade_id: Uuid::new_v4(),
            sym_id: 0,
            symbol: "NVDA".to_string(),
            bucket: ProofBucket::Brain02LargeCapDay,
            brain_id: BrainHorizonId::Brain02IntradayDayTrade,
            signal_ts_sec: est_unix(2026, 11, 12, 10, 0),
            entry_ts_sec: est_unix(2026, 11, 12, 10, 5),
            entry_day_key: est_trading_day_key(est_unix(2026, 11, 12, 10, 5)),
            entry_bar_ord: 6,
            direction: TradeDirection::Long,
            current_target_action: 0.85,
            structure_label: "BSM_OPT_$500.0C".to_string(),
            entry_mid_underlying: 500.0,
            entry_fill_underlying: 500.25,
            catastrophic_stop_px: 492.0,
            realized_partial_gross_usd: 0.0,
            realized_partial_cost_usd: 0.0,
            remaining_weight: 1.0,
            peak_favorable_px: 500.25,
            adverse_worst_px: 500.25,
            peak_v_liq_ret: 0.0,
            bars_held: 1,
            is_option_routed: true,
            is_vertical_combo: false,
            is_penny_tier: false,
            option_long_strike: 500.0,
            option_short_strike: None,
            option_entry_dte_days: 5.0,
            option_entry_theo_mid: 5.0,
            option_contracts: 2,
            option_spread_per_leg: 0.05,
            equity_shares: 200.0,
            slip_bps_one_way: 5.0,
            risk_budget_usd: 1_000.0,
            active_gen_id: 1,
            confidence_pct: 80.0,
            primary_driver: PrimaryFingerprintDriver::VolumeAbsorption,
            stream_weights: FourStreamWeights {
                chart_tape_physics_weight: 0.25,
                options_gamma_vanna_weight: 0.25,
                short_float_squeeze_weight: 0.25,
                market_internals_macro_weight: 0.25,
            },
            embedding_128: vec![0.1; 128],
            pending_exit: false,
            pending_scale_action: None,
        };

        let (_, exit_cost) =
            compute_leg_pnl_components(&pos, est_unix(2026, 11, 12, 10, 10), 500.0, 1.0, 0.20);
        assert!((exit_cost - 6.30).abs() < 1e-6);

        let ambiguous_bar = make_bar(
            est_unix(2026, 11, 12, 10, 10),
            500.0,
            506.0,
            491.5,
            505.0,
            100_000,
        );
        let (exit_px, reason, is_full_stop) =
            step_active_position_on_bar(&mut pos, &ambiguous_bar, 0.20, false)
                .expect("catastrophic backstop must execute first");
        assert_eq!(exit_px, 492.0);
        assert_eq!(reason, "CATASTROPHIC_INTRABAR_BACKSTOP");
        assert!(is_full_stop);
    }

    #[test]
    fn test_anti_cheat_intraday_1550_eod_forced_flatten_and_zero_lookahead_curtain() {
        let mut pos = ActiveSimulatedPosition {
            trade_id: Uuid::new_v4(),
            sym_id: 0,
            symbol: "QQQ".to_string(),
            bucket: ProofBucket::Scalp0Dte1m,
            brain_id: BrainHorizonId::Brain02IntradayDayTrade,
            signal_ts_sec: est_unix(2026, 11, 12, 15, 0),
            entry_ts_sec: est_unix(2026, 11, 12, 15, 1),
            entry_day_key: est_trading_day_key(est_unix(2026, 11, 12, 15, 1)),
            entry_bar_ord: 330,
            direction: TradeDirection::Long,
            current_target_action: 0.75,
            structure_label: "SHARES_ONLY".to_string(),
            entry_mid_underlying: 480.0,
            entry_fill_underlying: 480.24,
            catastrophic_stop_px: 470.0,
            realized_partial_gross_usd: 0.0,
            realized_partial_cost_usd: 0.0,
            remaining_weight: 1.0,
            peak_favorable_px: 480.5,
            adverse_worst_px: 479.8,
            peak_v_liq_ret: 0.0,
            bars_held: 10,
            is_option_routed: false,
            is_vertical_combo: false,
            is_penny_tier: false,
            option_long_strike: 0.0,
            option_short_strike: None,
            option_entry_dte_days: 0.2,
            option_entry_theo_mid: 0.0,
            option_contracts: 0,
            option_spread_per_leg: 0.0,
            equity_shares: 100.0,
            slip_bps_one_way: 5.0,
            risk_budget_usd: 750.0,
            active_gen_id: 1,
            confidence_pct: 79.0,
            primary_driver: PrimaryFingerprintDriver::GammaSqueeze,
            stream_weights: FourStreamWeights {
                chart_tape_physics_weight: 0.30,
                options_gamma_vanna_weight: 0.45,
                short_float_squeeze_weight: 0.10,
                market_internals_macro_weight: 0.15,
            },
            embedding_128: vec![0.1; 128],
            pending_exit: false,
            pending_scale_action: None,
        };

        let b_1549 = make_bar(
            est_unix(2026, 11, 12, 15, 49),
            480.2,
            480.6,
            480.1,
            480.4,
            50_000,
        );
        assert!(step_active_position_on_bar(&mut pos, &b_1549, 0.20, false).is_none());

        let b_1550 = make_bar(
            est_unix(2026, 11, 12, 15, 50),
            480.4,
            480.5,
            480.2,
            480.35,
            60_000,
        );
        let (exit_px, reason, _) = step_active_position_on_bar(&mut pos, &b_1550, 0.20, false)
            .expect("SCALP_0DTE must force flatten at 15:50 ET");
        assert_eq!(exit_px, 480.35);
        assert_eq!(reason, "EOD_FORCED_FLATTEN_1550_ET");

        let (a_orig, a_scram, delta_a, delta_v) = verify_future_curtain_zero_lookahead_invariance();
        assert_eq!(
            delta_a, 0.0,
            "Future-curtain invariance violated: scrambling bars > t changed a_t by {delta_a:.12}"
        );
        assert_eq!(
            delta_v, 0.0,
            "Future-curtain invariance violated: scrambling bars > t changed v_t by {delta_v:.12}"
        );
        println!(
            "[FUTURE-CURTAIN INVARIANCE PASSED] a_t_orig={a_orig:.12} | a_t_scrambled={a_scram:.12} | abs_diff_a={delta_a:.12} | abs_diff_v={delta_v:.12}"
        );
    }

    #[test]
    fn test_blotter_execution_physics_audit() {
        let csv_path = resolve_workspace_artifact_path(WALKFORWARD_BLOTTER_CSV_PATH);
        if !csv_path.exists() {
            return;
        }
        let content = std::fs::read_to_string(&csv_path).expect("read blotter csv");
        let mut rows = 0usize;
        for (line_idx, line) in content.lines().enumerate() {
            if line_idx == 0 || line.trim().is_empty() {
                continue;
            }
            let cols: Vec<&str> = line.split(',').collect();
            assert!(
                cols.len() >= 20,
                "Malformed blotter row {line_idx}: {line}"
            );
            let brain_id = cols[3];
            let sig_t = cols[6];
            let ent_t = cols[7];
            let ext_t = cols[8];
            let gross_r: f64 = cols[15].parse().expect("gross_r");
            let cost_usd: f64 = cols[16].parse().expect("cost_usd");
            let net_r: f64 = cols[18].parse().expect("net_r");

            assert!(
                ent_t > sig_t,
                "Row {line_idx}: entry_time ({ent_t}) must be strictly after signal_time ({sig_t})"
            );
            assert!(
                cost_usd > 0.0,
                "Row {line_idx}: spread_and_comm_cost_usd must be > 0, got {cost_usd}"
            );
            assert!(
                net_r <= gross_r + 1e-4,
                "Row {line_idx}: net_r ({net_r}) cannot exceed gross_r ({gross_r})"
            );
            if (brain_id.contains("SCALP")
                || brain_id.contains("BRAIN-01")
                || brain_id.contains("BRAIN-02"))
                && let Some(hhmm_part) = ext_t.split(' ').nth(1)
            {
                let hhmm_str: String = hhmm_part.split(':').take(2).collect();
                let hhmm_num: u32 = hhmm_str.parse().unwrap_or(0);
                assert!(
                    hhmm_num <= 1600,
                    "Row {line_idx}: intraday {brain_id} exited after RTH close at {ext_t}"
                );
            }
            rows += 1;
        }
        assert!(rows > 0, "Blotter CSV must contain >= 1 trade row");
        println!("[BLOTTER PHYSICS AUDIT PASSED] Verified {rows} blotter rows.");
    }

    #[tokio::test]
    async fn test_chronological_hard_evidence_walkforward_proof() {
        let models_dir = [Path::new("../../models"), Path::new("models")]
            .into_iter()
            .find(|p| p.exists());
        let onnx = Arc::new(OnnxMultiBrainEngine::new(models_dir));
        let adapter = YahooFinanceLiveAdapter::new();
        let report = run_chronological_hard_evidence_proof(onnx, &adapter, 100_000.0)
            .await
            .expect("Chronological hard-evidence walk-forward proof must succeed");

        assert!(
            report.dataset2_universe_symbols_loaded >= 55,
            "Expected >= 55 real cached/fetched universe symbols, got {}",
            report.dataset2_universe_symbols_loaded
        );
        assert!(Path::new(&report.blotter_csv_path).exists());
        assert!(Path::new(&report.report_md_path).exists());
        println!(
            "[PURE RL PROOF COMPLETE] Warmup SeqBars={} | OOS SeqBars={} | Online Cycles={} | OOS Trades RUN_A={} vs RUN_B={} | Net PnL A=${:+.2} vs B=${:+.2}",
            report.warmup_counterfactuals_harvested,
            report.oos_counterfactuals_harvested,
            report.online_rl_grad_updates,
            report.oos_trades_run_a,
            report.oos_trades_run_b,
            report.portfolio_run_a.total_net_pnl_usd,
            report.portfolio_run_b.total_net_pnl_usd
        );
    }

    #[test]
    fn test_purged_holdout_lcb_gate_rolls_back_correlated_inflation() {
        let mk_trade = |day_offset: i64, pnl: f64| SettledCausalTrade {
            sym_id: 0,
            symbol: "AAPL".to_string(),
            entry_bar_ord: 1,
            exit_bar_ord: 10,
            exit_ts_sec: est_unix(2026, 10, (5 + day_offset) as u32, 14, 0),
            direction: TradeDirection::Long,
            is_option_routed: false,
            gross_pnl_usd: pnl + 10.0,
            net_pnl_usd: pnl,
            net_r: pnl / 250.0,
            exit_reason: "NEURAL_POLICY_EXIT".to_string(),
            peak_v_liq_ret: 0.02,
        };
        // Champion: 4 clean independent daily wins (+$200 each = +$800, $200/trade expectancy)
        let champ_trades = vec![
            mk_trade(0, 200.0),
            mk_trade(1, 210.0),
            mk_trade(2, 190.0),
            mk_trade(3, 200.0),
        ];
        let champ = summarize_settled_causal_trades(&champ_trades, 1.2);

        // Candidate A: bloats to 8 correlated trades on same days with diluted expectancy ($110/trade = +$880)
        let bloated_trades = vec![
            mk_trade(0, 220.0),
            mk_trade(0, 210.0),
            mk_trade(1, 20.0),
            mk_trade(1, 15.0),
            mk_trade(2, 210.0),
            mk_trade(2, 205.0),
            mk_trade(3, 5.0),
            mk_trade(3, -5.0),
        ];
        let cand_bloated = summarize_settled_causal_trades(&bloated_trades, 1.5);
        assert!(
            !is_purged_holdout_lcb_promotion(&champ, &cand_bloated, false),
            "Correlated trade-count bloat with degraded per-trade expectancy must be ROLLED_BACK"
        );

        // Candidate B: genuine higher per-trade expectancy & Cross-Sectional LCB (+$260/trade = +$1,040)
        let clean_trades = vec![
            mk_trade(0, 260.0),
            mk_trade(1, 270.0),
            mk_trade(2, 250.0),
            mk_trade(3, 260.0),
        ];
        let cand_clean = summarize_settled_causal_trades(&clean_trades, 1.0);
        assert!(
            is_purged_holdout_lcb_promotion(&champ, &cand_clean, false),
            "Clean Cross-Sectional LCB + PF + expectancy improvement must PROMOTE"
        );
    }

    #[test]
    fn test_zero_trade_incumbent_promotes_positive_ulcb_candidate() {
        let mk_trade = |day_offset: i64, pnl: f64| SettledCausalTrade {
            sym_id: 0,
            symbol: "SPY".to_string(),
            entry_bar_ord: 1,
            exit_bar_ord: 8,
            exit_ts_sec: est_unix(2026, 11, (2 + day_offset) as u32, 11, 30),
            direction: TradeDirection::Long,
            is_option_routed: true,
            gross_pnl_usd: pnl + 8.0,
            net_pnl_usd: pnl,
            net_r: pnl / 200.0,
            exit_reason: "NEURAL_POLICY_EXIT".to_string(),
            peak_v_liq_ret: 0.018,
        };

        // Dormant 0-trade incumbent (`U_lcb == 0.0`, `red_loss_usd == $0.00`)
        let dormant_incumbent = summarize_settled_causal_trades(&[], 0.0);
        assert_eq!(compute_utility_lcb(&dormant_incumbent), 0.0);

        // Candidate wakes up with Net PF = 3.62 > 1.0, U_lcb > 0.0, even though gross losing dollars > $0.00!
        let wake_trades = vec![
            mk_trade(0, 220.0),
            mk_trade(0, -80.0),
            mk_trade(1, 250.0),
            mk_trade(2, -50.0),
            mk_trade(2, 230.0),
        ];
        let cand = summarize_settled_causal_trades(&wake_trades, 0.55);
        assert!(cand.red_loss_usd > 0.0);
        assert!(cand.net_pf > 1.0);
        assert!(compute_utility_lcb(&cand) > 0.0);

        assert!(
            is_purged_holdout_lcb_promotion(&dormant_incumbent, &cand, true),
            "0-trade incumbent (U_lcb=0) must deterministically promote candidate with Net PF > 1.0 and U_lcb > 0 on T_holdout"
        );
        assert!(
            is_causal_promotion(&dormant_incumbent, &cand, est_unix(2026, 11, 6, 16, 0), true),
            "0-trade incumbent must deterministically promote without Zero-Baseline Red-Loss Veto"
        );

        // Selective 2-win incumbent (`$0.00` red_loss_usd!) expanding into higher-U_lcb high-PF candidate (`red_loss_usd > $0.00`)
        let selective_inc = summarize_settled_causal_trades(
            &[mk_trade(0, 110.0), mk_trade(1, 105.0)],
            0.20,
        );
        assert_eq!(selective_inc.red_loss_usd, 0.0);
        assert!(cand.red_loss_usd > selective_inc.red_loss_usd);
        assert!(compute_utility_lcb(&cand) > compute_utility_lcb(&selective_inc));
        assert!(
            is_purged_holdout_lcb_promotion(&selective_inc, &cand, false),
            "Selective $0-loss incumbent must promote higher-U_lcb high-PF expansion on T_holdout"
        );
        assert!(
            is_causal_promotion(&selective_inc, &cand, est_unix(2026, 11, 6, 16, 0), true),
            "Selective $0-loss incumbent must promote higher-U_lcb high-PF expansion without red_loss_shrunk veto"
        );
    }

    #[test]
    fn test_friction_pruning_candidate_promotes_despite_lower_win_count() {
        let mk_trade = |day_offset: i64, pnl: f64| SettledCausalTrade {
            sym_id: 0,
            symbol: "SOUN".to_string(),
            entry_bar_ord: 1,
            exit_bar_ord: 12,
            exit_ts_sec: est_unix(2026, 10, (1 + (day_offset % 25)) as u32, 13, 0),
            direction: TradeDirection::Long,
            is_option_routed: false,
            gross_pnl_usd: pnl + 14.0,
            net_pnl_usd: pnl,
            net_r: pnl / 200.0,
            exit_reason: "NEURAL_POLICY_EXIT".to_string(),
            peak_v_liq_ret: 0.015,
        };

        // Incumbent: 43 winners (+$120 each) + 34 friction losers (-$115 each) => +$1,250 Net PnL, PF 1.32, +6.25R
        let mut inc_trades = Vec::new();
        for i in 0..43 {
            inc_trades.push(mk_trade(i, 120.0));
        }
        for i in 0..34 {
            inc_trades.push(mk_trade(i, -115.0));
        }
        let incumbent = summarize_settled_causal_trades(&inc_trades, 3.1);

        // Friction-pruning candidate A: 42 winners (win_count - 1!) + only 18 losers => +$3,066 Net PnL, PF 2.52, +15.33R
        let mut cand_trades = Vec::new();
        for i in 0..42 {
            cand_trades.push(mk_trade(i, 121.0));
        }
        for i in 0..18 {
            cand_trades.push(mk_trade(i, -112.0));
        }
        let candidate = summarize_settled_causal_trades(&cand_trades, 1.8);

        assert_eq!(incumbent.winners_count, 43);
        assert_eq!(candidate.winners_count, 42);
        assert!(compute_utility_lcb(&candidate) > compute_utility_lcb(&incumbent));
        assert!(candidate.net_pnl_usd > incumbent.net_pnl_usd);
        assert!(candidate.net_pf > incumbent.net_pf);

        assert!(
            is_purged_holdout_lcb_promotion(&incumbent, &candidate, false),
            "Friction-pruning candidate (43 -> 42 wins, higher U_lcb, Net PnL, Net PF, Net R) must deterministically promote on T_holdout"
        );
        assert!(
            is_causal_promotion(&incumbent, &candidate, est_unix(2026, 10, 30, 16, 0), true),
            "Friction-pruning candidate must deterministically promote without Raw Win-Count or Trade-Count Floor veto"
        );

        // Friction-pruning candidate B: 28 high-conviction winners (< 75% of 43 wins!) + 8 losers => +$3,300 Net PnL, PF 4.68
        let mut deep_pruned_trades = Vec::new();
        for i in 0..28 {
            deep_pruned_trades.push(mk_trade(i, 150.0));
        }
        for i in 0..8 {
            deep_pruned_trades.push(mk_trade(i, -112.5));
        }
        let deep_pruned = summarize_settled_causal_trades(&deep_pruned_trades, 1.2);
        assert!(deep_pruned.winners_count * 4 < incumbent.winners_count * 3);
        assert!(compute_utility_lcb(&deep_pruned) > compute_utility_lcb(&incumbent));
        assert!(
            is_purged_holdout_lcb_promotion(&incumbent, &deep_pruned, false),
            "Deep friction-pruning candidate (43 -> 28 wins, higher U_lcb & Total Net R) must promote on T_holdout"
        );
        assert!(
            is_causal_promotion(&incumbent, &deep_pruned, est_unix(2026, 10, 30, 16, 0), true),
            "Deep friction-pruning candidate (43 -> 28 wins) must promote without 75% integer win-count veto"
        );
    }
}
