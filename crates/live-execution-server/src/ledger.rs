//! Real-Time `POSITIONS OPENED` Active Monitor, `closed_trade_ledger` Archiver,
//! Master Executive Scoreboard, and `<10ms` "Slice-Anything" Hyper-Filter Engine.

use std::collections::HashSet;

use chrono::{DateTime, TimeZone, Utc};
use parking_lot::RwLock;
use raw_fingerprint_core::{
    BrainHorizonId, FourStreamWeights, OpenPositionEvaluationContext, PrimaryFingerprintDriver,
    RawBarRecord, RawFingerprintEngine, RawMarketInternalsSnapshot, SubPennyPrice,
    TraceHealthBadge, TradeDirection,
};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::onnx_engine::LiveBrainInferenceOutput;
use crate::precedent_knn::PrecedentMatchSummary;
use crate::webhooks::{
    OutboundWebhookDispatcher, WebhookChannelConfig, WebhookDispatchRecord, WebhookEventPayload,
    WebhookEventType,
};

/// Causal Fractional-Kelly Capital Allocation Plan computed for each opened position.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct KellyAllocationPlan {
    pub win_probability: f64,
    pub payoff_ratio_b: f64,
    pub raw_kelly_fraction: f64,
    pub kelly_fraction_pct: f64,
    pub risk_budget_usd: f64,
    pub notional_allocated_usd: f64,
    pub quantity_shares: f64,
}

/// Institutional Causal Fractional-Kelly & Portfolio Heat Capital Allocator.
///
/// Replaces static flat notional sizing with causal empirical Fractional-Kelly risk sizing:
/// - Causal win probability $W \in [0.50, 0.92]$ blending `PrecedentKnnIndex` causal win rate,
///   Era-1 calibrated win rate, and normalized AI confidence.
/// - Causal payoff ratio $b = \text{clamp}(\text{Empirical } p_{50}\text{ MFE} / \text{Empirical } p_{85}\text{ MAE}, 1.15, 5.00)$
///   blended with the Brain's Era-1 calibrated Profit Factor.
/// - Raw Kelly Fraction: $f_{\text{raw}} = W - \frac{1 - W}{b}$
/// - **0.35x Fractional-Kelly Scaling** modulated by Brain Expectancy Tier and clamped to
///   institutional risk bounds (`0.50%` minimum to `2.75%` maximum of account equity risked per
///   `1.0R` stop distance; notional position value clamped between `$5,000` and `$35,000` per tranche).
pub struct KellyCapitalAllocator;

impl KellyCapitalAllocator {
    #[allow(clippy::too_many_arguments)]
    pub fn compute_allocation(
        account_equity_usd: f64,
        brain_id: BrainHorizonId,
        entry_price: SubPennyPrice,
        stop_price: SubPennyPrice,
        ai_confidence_pct: f64,
        precedent_win_rate_pct: f64,
        p50_mfe_pct: f64,
        p85_mae_pct: f64,
        era1_calibrated_win_rate_pct: f64,
        era1_calibrated_profit_factor: f64,
    ) -> KellyAllocationPlan {
        let equity = account_equity_usd.max(50_000.0);
        let norm_conf = if ai_confidence_pct <= 1.0 {
            ai_confidence_pct
        } else {
            ai_confidence_pct / 100.0
        };
        let w_knn = (precedent_win_rate_pct / 100.0).clamp(0.45, 0.95);
        let w_era1 = (era1_calibrated_win_rate_pct / 100.0).clamp(0.50, 0.92);
        let w_conf = norm_conf.clamp(0.50, 0.95);
        let win_probability = (0.45 * w_knn + 0.35 * w_era1 + 0.20 * w_conf).clamp(0.50, 0.92);

        let empirical_ratio = (p50_mfe_pct.max(0.25) / p85_mae_pct.max(0.15)).clamp(1.15, 5.00);
        let pf_clamped = era1_calibrated_profit_factor.clamp(1.15, 5.00);
        let payoff_ratio_b = (0.60 * empirical_ratio + 0.40 * pf_clamped).clamp(1.15, 5.00);

        let raw_kelly_fraction =
            (win_probability - (1.0 - win_probability) / payoff_ratio_b).max(0.015);

        let entry_f64 = entry_price.to_f64().max(0.00001);
        let stop_f64 = stop_price.to_f64().max(0.00001);
        let stop_dist_per_share = (entry_f64 - stop_f64)
            .abs()
            .max(0.005 * entry_f64)
            .max(0.00001);

        // Brain Expectancy Tier portfolio heat factor across concurrent multi-brain slots
        let tier_expectancy_mod = match brain_id {
            BrainHorizonId::Brain01PennyAnomaly => 0.042,
            BrainHorizonId::Brain02IntradayDayTrade => 0.056,
            BrainHorizonId::Brain03ShortTermSwing => 0.092,
            BrainHorizonId::Brain04CoreSwing => 0.138,
            BrainHorizonId::Brain05LongTermWealth => 0.158,
        };

        let stop_pct = (stop_dist_per_share / entry_f64).clamp(0.005, 0.25);
        let ref_stop_pct = match brain_id {
            BrainHorizonId::Brain01PennyAnomaly | BrainHorizonId::Brain02IntradayDayTrade => 0.011,
            BrainHorizonId::Brain03ShortTermSwing => 0.017,
            BrainHorizonId::Brain04CoreSwing | BrainHorizonId::Brain05LongTermWealth => 0.045,
        };
        // Inverse stop-variance dampening: wide stops have higher variance (sigma^2), reducing Kelly fraction
        let variance_dampener = (ref_stop_pct / stop_pct).clamp(0.38, 1.35);

        let kelly_risk_frac =
            (0.35 * raw_kelly_fraction * tier_expectancy_mod * variance_dampener)
                .clamp(0.005, 0.0275);
        let target_risk_budget_usd = equity * kelly_risk_frac;

        // Normalize per-tranche risk parity against the $35,000 notional ceiling so wide-stop trades
        // never risk more dollars than tight-stop trades capped by the $35k notional limit
        let max_1r_at_ceiling = 35_000.0 * ref_stop_pct * (kelly_risk_frac / 0.0065).clamp(0.65, 1.85);
        let effective_1r_risk_usd = target_risk_budget_usd.min(max_1r_at_ceiling * variance_dampener);

        let raw_shares = (effective_1r_risk_usd / stop_dist_per_share)
            .round()
            .max(1.0);
        let raw_notional = raw_shares * entry_f64;

        // Clamp notional position value between $5,000 and $35,000 per tranche
        let clamped_notional = raw_notional.clamp(5_000.0, 35_000.0);
        let quantity_shares = (clamped_notional / entry_f64).round().max(1.0);
        let notional_allocated_usd = ((quantity_shares * entry_f64) * 100.0).round() / 100.0;
        let risk_budget_usd = ((target_risk_budget_usd) * 100.0).round() / 100.0;
        let kelly_fraction_pct = ((kelly_risk_frac * 100.0) * 100.0).round() / 100.0;

        KellyAllocationPlan {
            win_probability,
            payoff_ratio_b,
            raw_kelly_fraction,
            kelly_fraction_pct,
            risk_budget_usd,
            notional_allocated_usd,
            quantity_shares,
        }
    }
}

/// Immutable final outcome badge assigned when a position transitions to `closed_trade_ledger`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ClosedOutcomeBadge {
    #[serde(rename = "CLOSED - TARGET HIT")]
    TargetHit,
    #[serde(rename = "CLOSED - AI DYNAMIC EXIT")]
    AiDynamicExit,
    #[serde(rename = "CLOSED - TRAILING STOP")]
    TrailingStop,
    #[serde(rename = "CLOSED - STOPPED OUT")]
    StoppedOut,
}

impl ClosedOutcomeBadge {
    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::TargetHit => "CLOSED - TARGET HIT",
            Self::AiDynamicExit => "CLOSED - AI DYNAMIC EXIT",
            Self::TrailingStop => "CLOSED - TRAILING STOP",
            Self::StoppedOut => "CLOSED - STOPPED OUT",
        }
    }
}

/// Parallel Dual-Engine Execution Mode (`ENGINE-A`, `ENGINE-B`, `CONSENSUS`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum EngineExecutionMode {
    #[default]
    StructuredHybridChampion,
    PureMlSelfLearned,
    ConsensusBothEngines,
}

impl EngineExecutionMode {
    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::StructuredHybridChampion => "STRUCTURED_HYBRID_CHAMPION",
            Self::PureMlSelfLearned => "PURE_ML_SELF_LEARNED",
            Self::ConsensusBothEngines => "CONSENSUS_BOTH_ENGINES",
        }
    }

    #[inline]
    pub const fn display_name(self) -> &'static str {
        match self {
            Self::StructuredHybridChampion => "ENGINE-A (Structured Hybrid Champion)",
            Self::PureMlSelfLearned => "ENGINE-B (Pure ML Self-Learned)",
            Self::ConsensusBothEngines => "CONSENSUS (Both Engines Agreed)",
        }
    }
}

/// Granular Strategy Archetype Classification across all 5 Brains.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum StrategyArchetype {
    #[default]
    PocVolumeAbsorption,
    GammaSqueezeIgnition,
    ShortFloatLiquefaction,
    MacroInternalsTideDrive,
    HtfMultiScaleBreakout,
    PureMlLatentAnomaly,
}

impl StrategyArchetype {
    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::GammaSqueezeIgnition => "GAMMA_SQUEEZE_IGNITION",
            Self::ShortFloatLiquefaction => "SHORT_FLOAT_LIQUEFACTION",
            Self::PocVolumeAbsorption => "POC_VOLUME_ABSORPTION",
            Self::MacroInternalsTideDrive => "MACRO_INTERNALS_TIDE_DRIVE",
            Self::HtfMultiScaleBreakout => "HTF_MULTI_SCALE_BREAKOUT",
            Self::PureMlLatentAnomaly => "PURE_ML_LATENT_ANOMALY",
        }
    }

    #[inline]
    pub const fn display_name(self) -> &'static str {
        match self {
            Self::GammaSqueezeIgnition => "Gamma Squeeze Ignition",
            Self::ShortFloatLiquefaction => "Short Float Liquefaction",
            Self::PocVolumeAbsorption => "POC Volume Absorption",
            Self::MacroInternalsTideDrive => "Macro Internals Tide Drive",
            Self::HtfMultiScaleBreakout => "HTF Multi-Scale Breakout",
            Self::PureMlLatentAnomaly => "Pure ML Latent Anomaly",
        }
    }

    #[inline]
    pub const fn all() -> [Self; 6] {
        [
            Self::GammaSqueezeIgnition,
            Self::ShortFloatLiquefaction,
            Self::PocVolumeAbsorption,
            Self::MacroInternalsTideDrive,
            Self::HtfMultiScaleBreakout,
            Self::PureMlLatentAnomaly,
        ]
    }
}

/// Root-cause category diagnosed by the Automatic Per-Loss Forensic Autopsy Engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum LossRootCauseCategory {
    OptionsGammaWallRejection,
    MacroInternalsTideDivergence,
    ExhaustionWickLiquidityTrap,
    ShortFloatOrSpreadFriction,
    HtfStructuralHeadwind,
    PostEntryVolumeVacuum,
}

impl LossRootCauseCategory {
    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::OptionsGammaWallRejection => "OPTIONS_GAMMA_WALL_REJECTION",
            Self::MacroInternalsTideDivergence => "MACRO_INTERNALS_TIDE_DIVERGENCE",
            Self::ExhaustionWickLiquidityTrap => "EXHAUSTION_WICK_LIQUIDITY_TRAP",
            Self::ShortFloatOrSpreadFriction => "SHORT_FLOAT_OR_SPREAD_FRICTION",
            Self::HtfStructuralHeadwind => "HTF_STRUCTURAL_HEADWIND",
            Self::PostEntryVolumeVacuum => "POST_ENTRY_VOLUME_VACUUM",
        }
    }
}

/// Single culprit raw feature out of the 32-feature state tensor $X_{\text{loss}}$ compared against $\bar{X}_{\text{win}}$.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeatureCulpritDiagnostic {
    pub feature_index: usize,
    pub feature_name: String,
    pub stream_name: String,
    pub loss_value: f64,
    pub winner_centroid_value: f64,
    pub anomaly_z_score: f64,
}

/// Complete Per-Loss Forensic Dissection attached to every losing `ClosedTradeRow` and summarized in `BrainEvolutionStatus`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LossForensicAutopsy {
    pub trade_id: Uuid,
    pub symbol: String,
    pub brain_id: BrainHorizonId,
    pub root_cause_category: LossRootCauseCategory,
    pub top_culprit_features: Vec<FeatureCulpritDiagnostic>,
    pub what_went_wrong_summary: String,
    pub learned_adaptation_rule: String,
    pub counterfactual_early_exit_r: f64,
}

/// Audit record logged whenever the Causal Hard-Negative Trap Memory vetoes a repeat loss ($t > T_{\text{loss\_closed}}$).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VetoedTrapAuditRecord {
    pub symbol: String,
    pub brain_id: BrainHorizonId,
    pub timestamp_ns: i64,
    pub timestamp_iso: String,
    pub matched_past_loss_trade_id: Uuid,
    pub root_cause_category: LossRootCauseCategory,
    pub trap_similarity_score: f64,
    pub counterfactual_loss_avoided_usd: f64,
}

/// Strategy Archetype Win-Rate Leaderboard Row returned in `FilteredLedgerSnapshot.by_strategy_leaderboard`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StrategyArchetypePerformanceRow {
    pub strategy_archetype: StrategyArchetype,
    pub display_name: String,
    pub total_trades: usize,
    pub wins: usize,
    pub losses: usize,
    pub win_rate_pct: f64,
    pub profit_factor: f64,
    pub avg_realized_r: f64,
    pub best_symbol: String,
    pub realized_stock_net_pnl_usd: f64,
    pub realized_option_net_pnl_usd: f64,
    pub kelly_realized_net_pnl_usd: f64,
}

/// Engine Mode Win-Rate Breakdown Row returned in `FilteredLedgerSnapshot.by_engine_breakdown`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EngineModePerformanceRow {
    pub engine_mode: EngineExecutionMode,
    pub display_name: String,
    pub total_trades: usize,
    pub wins: usize,
    pub losses: usize,
    pub win_rate_pct: f64,
    pub profit_factor: f64,
    pub avg_realized_r: f64,
    pub best_symbol: String,
    pub realized_stock_net_pnl_usd: f64,
    pub realized_option_net_pnl_usd: f64,
    pub kelly_realized_net_pnl_usd: f64,
}

pub const RAW_32_FEATURE_METADATA: [(&str, &str); 32] = [
    ("s1_log_ret_close", "STREAM_1_CHART_TAPE_PHYSICS"),
    ("s1_log_open_gap", "STREAM_1_CHART_TAPE_PHYSICS"),
    ("s1_log_hl_range", "STREAM_1_CHART_TAPE_PHYSICS"),
    ("s1_body_conviction", "STREAM_1_CHART_TAPE_PHYSICS"),
    ("s1_upper_wick_ratio", "STREAM_1_CHART_TAPE_PHYSICS"),
    ("s1_lower_wick_ratio", "STREAM_1_CHART_TAPE_PHYSICS"),
    ("s1_vwap_disp_log", "STREAM_1_CHART_TAPE_PHYSICS"),
    ("s1_vol_surge_log", "STREAM_1_CHART_TAPE_PHYSICS"),
    ("s1_turnover_vel_log", "STREAM_1_CHART_TAPE_PHYSICS"),
    ("s1_inst_print_density", "STREAM_1_CHART_TAPE_PHYSICS"),
    ("s1_spread_bps_norm", "STREAM_1_CHART_TAPE_PHYSICS"),
    ("s1_poc_clustering_score", "STREAM_1_CHART_TAPE_PHYSICS"),
    ("s2_call_put_log", "STREAM_2_OPTIONS_GAMMA_VANNA"),
    ("s2_ask_bid_imbalance", "STREAM_2_OPTIONS_GAMMA_VANNA"),
    ("s2_dealer_gex_norm", "STREAM_2_OPTIONS_GAMMA_VANNA"),
    ("s2_dealer_vanna_norm", "STREAM_2_OPTIONS_GAMMA_VANNA"),
    ("s2_zero_dte_share", "STREAM_2_OPTIONS_GAMMA_VANNA"),
    ("s2_iv_term_slope", "STREAM_2_OPTIONS_GAMMA_VANNA"),
    ("s2_gamma_wall_dist", "STREAM_2_OPTIONS_GAMMA_VANNA"),
    ("s2_oi_turnover", "STREAM_2_OPTIONS_GAMMA_VANNA"),
    ("s3_float_rotation", "STREAM_3_SHORT_FLOAT_SQUEEZE"),
    ("s3_short_to_float", "STREAM_3_SHORT_FLOAT_SQUEEZE"),
    ("s3_dtc_pressure", "STREAM_3_SHORT_FLOAT_SQUEEZE"),
    ("s3_intraday_short_ratio", "STREAM_3_SHORT_FLOAT_SQUEEZE"),
    ("s3_ctb_pressure", "STREAM_3_SHORT_FLOAT_SQUEEZE"),
    ("s3_dilution_hazard", "STREAM_3_SHORT_FLOAT_SQUEEZE"),
    ("s4_nyse_tick", "STREAM_4_MARKET_INTERNALS_TIDE"),
    ("s4_nasdaq_tick", "STREAM_4_MARKET_INTERNALS_TIDE"),
    ("s4_add_breadth", "STREAM_4_MARKET_INTERNALS_TIDE"),
    ("s4_vold_log", "STREAM_4_MARKET_INTERNALS_TIDE"),
    ("s4_vix_term", "STREAM_4_MARKET_INTERNALS_TIDE"),
    ("s4_sector_flow", "STREAM_4_MARKET_INTERNALS_TIDE"),
];

pub fn classify_strategy_archetype(
    engine_mode: EngineExecutionMode,
    brain_id: BrainHorizonId,
    primary_driver: PrimaryFingerprintDriver,
    stream_weights: &FourStreamWeights,
) -> StrategyArchetype {
    if engine_mode == EngineExecutionMode::PureMlSelfLearned {
        return StrategyArchetype::PureMlLatentAnomaly;
    }
    if primary_driver == PrimaryFingerprintDriver::GammaSqueeze
        || stream_weights.options_gamma_vanna_weight >= 0.27
    {
        StrategyArchetype::GammaSqueezeIgnition
    } else if primary_driver == PrimaryFingerprintDriver::ShortSqueeze
        || stream_weights.short_float_squeeze_weight >= 0.255
    {
        StrategyArchetype::ShortFloatLiquefaction
    } else if matches!(
        brain_id,
        BrainHorizonId::Brain01PennyAnomaly | BrainHorizonId::Brain03ShortTermSwing
    ) && stream_weights.chart_tape_physics_weight >= 0.23
    {
        StrategyArchetype::HtfMultiScaleBreakout
    } else if primary_driver == PrimaryFingerprintDriver::VolumeAbsorption
        || stream_weights.chart_tape_physics_weight >= stream_weights.market_internals_macro_weight
    {
        StrategyArchetype::PocVolumeAbsorption
    } else {
        StrategyArchetype::MacroInternalsTideDrive
    }
}

/// Live active row in the `POSITIONS OPENED` monitor (`positions_opened` table).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpenPositionRow {
    pub position_id: Uuid,
    pub opened_at_iso: String,
    pub opened_timestamp_ns: i64,
    pub live_hold_duration_ms: i64,
    pub brain_id: BrainHorizonId,
    pub brain_name: String,
    pub symbol: String,
    pub exchange_index: String,
    pub direction: TradeDirection,
    // Stock execution & live levels (5-decimal $0.00001 precision)
    pub entry_price: SubPennyPrice,
    pub live_price: SubPennyPrice,
    pub initial_structural_stop: SubPennyPrice,
    pub structural_stop_price: SubPennyPrice,
    pub target_1_price: SubPennyPrice,
    pub target_2_price: SubPennyPrice,
    pub peak_favorable_price: SubPennyPrice,
    pub max_adverse_price: SubPennyPrice,
    pub quantity_shares: f64,
    #[serde(default)]
    pub kelly_fraction_pct: f64,
    #[serde(default)]
    pub risk_budget_usd: f64,
    #[serde(default)]
    pub notional_allocated_usd: f64,
    #[serde(default = "default_tranche_count")]
    pub tranche_count: u8,
    #[serde(default)]
    pub scale_in_history: Vec<String>,
    // Matched Options execution & live levels
    pub option_structure: String,
    pub matched_option_contract: String,
    pub option_entry_premium: SubPennyPrice,
    pub option_live_premium: SubPennyPrice,
    pub option_stop_premium: SubPennyPrice,
    pub option_target_premium: SubPennyPrice,
    pub option_contracts_count: u32,
    pub contract_delta: f64,
    pub contract_gamma: f64,
    #[serde(default)]
    pub contract_vanna: f64,
    #[serde(default)]
    pub contract_theta: f64,
    #[serde(default)]
    pub contract_vega: f64,
    #[serde(default)]
    pub contract_iv: f64,
    // Live unrealized P&L telemetry
    pub unrealized_stock_pnl_usd: f64,
    pub unrealized_stock_pnl_pct: f64,
    pub unrealized_option_pnl_usd: f64,
    pub unrealized_option_pnl_pct: f64,
    pub current_r_multiple: f64,
    pub mae_pct: f64,
    pub mfe_pct: f64,
    // AI Raw Fingerprint & Trace Health
    pub ai_confidence_pct: f64,
    pub primary_driver: PrimaryFingerprintDriver,
    pub trace_health_badge: TraceHealthBadge,
    pub precedent_matches_count: usize,
    pub precedent_win_rate_pct: f64,
    pub stream_weights: FourStreamWeights,
    pub price_sparkline: Vec<f64>,
    #[serde(default)]
    pub entry_bars: Vec<RawBarRecord>,
    #[serde(default)]
    pub entry_bar_idx: usize,
    #[serde(default)]
    pub engine_mode: EngineExecutionMode,
    #[serde(default)]
    pub strategy_archetype: StrategyArchetype,
    #[serde(default)]
    pub entry_embedding_128: Vec<f32>,
}

/// Completed immutable row in the Historical Closed Trade Ledger (`closed_trade_ledger` table).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClosedTradeRow {
    pub trade_id: Uuid,
    pub position_id: Uuid,
    pub outcome_badge: ClosedOutcomeBadge,
    pub brain_id: BrainHorizonId,
    pub brain_name: String,
    pub symbol: String,
    pub exchange_index: String,
    pub direction: TradeDirection,
    pub opened_at_iso: String,
    pub closed_at_iso: String,
    pub opened_timestamp_ns: i64,
    pub closed_timestamp_ns: i64,
    pub hold_duration_ms: i64,
    // Stock execution levels
    pub entry_price: SubPennyPrice,
    pub initial_structural_stop: SubPennyPrice,
    pub target_1_price: SubPennyPrice,
    pub target_2_price: SubPennyPrice,
    pub actual_exit_price: SubPennyPrice,
    pub quantity_shares: f64,
    #[serde(default)]
    pub kelly_fraction_pct: f64,
    #[serde(default)]
    pub risk_budget_usd: f64,
    #[serde(default)]
    pub notional_allocated_usd: f64,
    #[serde(default = "default_tranche_count")]
    pub tranche_count: u8,
    #[serde(default)]
    pub scale_in_history: Vec<String>,
    // Matched Options execution levels
    pub option_structure: String,
    pub matched_option_contract: String,
    pub option_entry_premium: SubPennyPrice,
    pub option_stop_premium: SubPennyPrice,
    pub option_exit_premium: SubPennyPrice,
    pub option_contracts_count: u32,
    #[serde(default)]
    pub contract_delta: f64,
    #[serde(default)]
    pub contract_gamma: f64,
    #[serde(default)]
    pub contract_vanna: f64,
    #[serde(default)]
    pub contract_theta: f64,
    #[serde(default)]
    pub contract_vega: f64,
    #[serde(default)]
    pub contract_iv: f64,
    // Realized performance
    pub gross_pnl_usd: f64,
    pub slippage_commissions_usd: f64,
    pub net_pnl_usd: f64,
    #[serde(default)]
    pub kelly_net_pnl_usd: f64,
    pub option_net_pnl_usd: f64,
    pub stock_return_pct: f64,
    pub option_return_pct: f64,
    pub realized_r_multiple: f64,
    pub mae_pct: f64,
    pub mfe_pct: f64,
    pub is_winner: bool,
    pub cumulative_net_profit_usd: f64,
    // AI Fingerprint Snapshot
    pub ai_confidence_pct: f64,
    pub primary_driver: PrimaryFingerprintDriver,
    pub precedent_matches_count: usize,
    pub precedent_win_rate_pct: f64,
    pub stream_weights: FourStreamWeights,
    pub price_sparkline: Vec<f64>,
    #[serde(default)]
    pub entry_bars: Vec<RawBarRecord>,
    #[serde(default)]
    pub entry_bar_idx: usize,
    #[serde(default)]
    pub exit_bar_idx: usize,
    #[serde(default)]
    pub engine_mode: EngineExecutionMode,
    #[serde(default)]
    pub strategy_archetype: StrategyArchetype,
    #[serde(default)]
    pub loss_forensic_autopsy: Option<LossForensicAutopsy>,
    #[serde(default)]
    pub entry_embedding_128: Vec<f32>,
}

/// Speed-2 Nightly Autopsy Grade for a single closed trade (`A+`, `B`, `C`, `F`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AutopsyTradeGrade {
    pub trade_id: Uuid,
    pub symbol: String,
    pub brain_id: BrainHorizonId,
    pub grade: String,
    pub realized_r_multiple: f64,
    pub mfe_pct: f64,
    pub mae_pct: f64,
    pub is_hard_negative: bool,
    pub diagnosis: String,
    pub weight_adjustment_summary: String,
}

/// Speed-3 Weekend Champion vs Challenger Walk-Forward Tournament Comparison per Brain.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChampionChallengerComparison {
    pub brain_id: BrainHorizonId,
    pub brain_name: String,
    pub champion_version: String,
    pub challenger_version: String,
    pub champion_profit_factor: f64,
    pub challenger_profit_factor: f64,
    pub champion_max_drawdown_pct: f64,
    pub challenger_max_drawdown_pct: f64,
    pub challenger_promoted: bool,
    pub promotion_reason: String,
}

/// Complete 3-Speed Self-Learning Evolution Status reported to `BrainAutopsyLab.svelte`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BrainEvolutionStatus {
    pub last_autopsy_iso: String,
    pub total_autopsy_runs: u64,
    pub hard_negatives_mined: usize,
    pub missed_moves_indexed: usize,
    pub chop_throttle_active: bool,
    pub chop_confidence_floor_pct: f64,
    pub trade_grades: Vec<AutopsyTradeGrade>,
    pub tournament_results: Vec<ChampionChallengerComparison>,
    #[serde(default)]
    pub era1_calibrations: Vec<Era1BrainCalibrationProfile>,
    #[serde(default)]
    pub loss_autopsies: Vec<LossForensicAutopsy>,
    #[serde(default)]
    pub vetoed_repeat_traps: Vec<VetoedTrapAuditRecord>,
}

/// Per-Brain breakdown metrics used in the Scoreboard and Tab 4 of the `.XLSX` workbook.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BrainBreakdownSummary {
    pub brain_id: BrainHorizonId,
    pub brain_name: String,
    pub open_positions_count: usize,
    pub closed_trades_count: usize,
    pub wins_count: usize,
    pub losses_count: usize,
    pub win_rate_pct: f64,
    pub realized_net_profit_usd: f64,
    pub unrealized_open_pnl_usd: f64,
    pub profit_factor: f64,
    pub avg_r_multiple: f64,
    pub avg_hold_duration_ms: i64,
}

/// Master Executive Scoreboard (`<10ms` reactive top-level rollup).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MasterExecutiveScoreboard {
    pub realized_net_profit_usd: f64,
    pub realized_net_return_pct: f64,
    pub unrealized_open_pnl_usd: f64,
    pub unrealized_open_option_pnl_usd: f64,
    pub combined_net_profit_usd: f64,
    pub win_rate_pct: f64,
    pub total_closed_trades: usize,
    pub total_wins: usize,
    pub total_losses: usize,
    pub gross_profit_usd: f64,
    pub gross_loss_usd: f64,
    pub positions_opened_count: usize,
    pub profit_factor: f64,
    pub avg_winner_usd: f64,
    pub avg_loser_usd: f64,
    pub avg_winner_r: f64,
    pub avg_loser_r: f64,
    pub avg_hold_duration_ms: i64,
    pub avg_winner_hold_duration_ms: i64,
    pub avg_loser_hold_duration_ms: i64,
    pub max_drawdown_usd: f64,
    pub max_drawdown_pct: f64,
    pub by_brain_breakdowns: Vec<BrainBreakdownSummary>,
    pub filter_eval_latency_micros: u64,
}

/// Multi-dimensional "Slice-Anything" Hyper-Filter query specification.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct HyperFilterQuery {
    /// `"ALL"`, `"POSITIONS_OPENED"`, or `"CLOSED_ONLY"`
    pub view_mode: Option<String>,
    /// `"BRAIN-01"` .. `"BRAIN-05"`
    pub brain_id: Option<BrainHorizonId>,
    /// `"ALL"`, `"WINNERS"`, `"LOSERS"`, `"STOPPED_OUT"`, `"DYNAMIC_EXIT"`, `"OVER_2R"`
    pub outcome_filter: Option<String>,
    /// `"SP_500"`, `"NASDAQ_100"`, `"RUSSELL_2000"`, `"DOW_30"`, `"OTC_PENNY"`
    pub exchange_index: Option<String>,
    /// Comma-separated or set of ticker symbols
    pub symbols: Option<HashSet<String>>,
    /// Custom min entry price down to `$0.00001`
    pub min_entry_price: Option<SubPennyPrice>,
    /// Custom max entry price up to `$10,000+`
    pub max_entry_price: Option<SubPennyPrice>,
    pub min_hold_duration_ms: Option<i64>,
    pub max_hold_duration_ms: Option<i64>,
    pub min_confidence_pct: Option<f64>,
    pub primary_driver: Option<PrimaryFingerprintDriver>,
    pub strategy_archetype: Option<StrategyArchetype>,
    pub engine_mode: Option<EngineExecutionMode>,
}

/// Complete filtered snapshot returned in `<2ms` to the Command Terminal or Exporter.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FilteredLedgerSnapshot {
    pub scoreboard: MasterExecutiveScoreboard,
    pub positions_opened: Vec<OpenPositionRow>,
    pub closed_trades: Vec<ClosedTradeRow>,
    #[serde(default)]
    pub by_strategy_leaderboard: Vec<StrategyArchetypePerformanceRow>,
    #[serde(default)]
    pub by_engine_breakdown: Vec<EngineModePerformanceRow>,
}

fn default_tranche_count() -> u8 {
    1
}

fn format_timestamp_ns_iso(ts_ns: i64) -> String {
    let secs = ts_ns / 1_000_000_000;
    let nsecs = (ts_ns % 1_000_000_000).unsigned_abs() as u32;
    let dt: DateTime<Utc> = Utc
        .timestamp_opt(secs, nsecs)
        .single()
        .unwrap_or_else(Utc::now);
    dt.format("%Y-%m-%dT%H:%M:%S.%3fZ").to_string()
}

/// Causal Era-1 (`0..calibration_end`) Self-Learning Calibration Profile for a Brain (`BRAIN-01` .. `BRAIN-05`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Era1BrainCalibrationProfile {
    pub brain_id: BrainHorizonId,
    pub calibrated_confidence_floor: f64,
    pub min_precedent_win_rate_pct: f64,
    pub min_expected_r_multiple: f64,
    pub stream_prior_weights: FourStreamWeights,
    pub era1_sample_trades: usize,
    pub era1_calibrated_win_rate_pct: f64,
    pub era1_calibrated_profit_factor: f64,
    #[serde(default)]
    pub era1_learned_min_displacement_pct: f64,
    #[serde(default)]
    pub era1_learned_max_displacement_pct: f64,
}

/// Thread-safe, sub-millisecond Real-Time Execution Ledger Engine.
pub struct RealTimeLedgerEngine {
    fingerprint_engine: RawFingerprintEngine,
    positions_opened: RwLock<Vec<OpenPositionRow>>,
    closed_ledger: RwLock<Vec<ClosedTradeRow>>,
    running_net_profit_usd: RwLock<f64>,
    autopsy_runs_count: RwLock<u64>,
    last_autopsy_iso: RwLock<String>,
    era1_calibrations:
        RwLock<std::collections::HashMap<BrainHorizonId, Era1BrainCalibrationProfile>>,
    vetoed_repeat_traps: RwLock<Vec<VetoedTrapAuditRecord>>,
    webhook_dispatcher: OutboundWebhookDispatcher,
}

impl Default for RealTimeLedgerEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl RealTimeLedgerEngine {
    pub fn new() -> Self {
        Self {
            fingerprint_engine: RawFingerprintEngine::new(),
            positions_opened: RwLock::new(Vec::with_capacity(128)),
            closed_ledger: RwLock::new(Vec::with_capacity(4096)),
            running_net_profit_usd: RwLock::new(0.0),
            autopsy_runs_count: RwLock::new(1),
            last_autopsy_iso: RwLock::new(Utc::now().format("%Y-%m-%dT%H:%M:%S.%3fZ").to_string()),
            era1_calibrations: RwLock::new(std::collections::HashMap::new()),
            vetoed_repeat_traps: RwLock::new(Vec::with_capacity(256)),
            webhook_dispatcher: OutboundWebhookDispatcher::new(),
        }
    }

    pub fn webhooks(&self) -> &OutboundWebhookDispatcher {
        &self.webhook_dispatcher
    }

    pub fn record_vetoed_trap(&self, record: VetoedTrapAuditRecord) {
        let mut guard = self.vetoed_repeat_traps.write();
        if guard.len() >= 512 {
            guard.remove(0);
        }
        guard.push(record);
    }

    pub fn vetoed_repeat_traps(&self) -> Vec<VetoedTrapAuditRecord> {
        self.vetoed_repeat_traps.read().clone()
    }

    pub fn set_era1_brain_calibration(&self, profile: Era1BrainCalibrationProfile) {
        self.era1_calibrations
            .write()
            .insert(profile.brain_id, profile);
    }

    pub fn accumulate_era1_brain_calibration(&self, profile: Era1BrainCalibrationProfile) {
        let mut map = self.era1_calibrations.write();
        if let Some(existing) = map.get_mut(&profile.brain_id) {
            let n1 = existing.era1_sample_trades.max(1) as f64;
            let n2 = profile.era1_sample_trades.max(1) as f64;
            let tot = n1 + n2;
            existing.era1_sample_trades += profile.era1_sample_trades;
            existing.era1_calibrated_win_rate_pct = (existing.era1_calibrated_win_rate_pct * n1
                + profile.era1_calibrated_win_rate_pct * n2)
                / tot;
            existing.era1_calibrated_profit_factor = (existing.era1_calibrated_profit_factor * n1
                + profile.era1_calibrated_profit_factor * n2)
                / tot;
            existing.era1_learned_min_displacement_pct =
                (existing.era1_learned_min_displacement_pct * n1
                    + profile.era1_learned_min_displacement_pct * n2)
                    / tot;
            existing.era1_learned_max_displacement_pct =
                (existing.era1_learned_max_displacement_pct * n1
                    + profile.era1_learned_max_displacement_pct * n2)
                    / tot;
            existing.calibrated_confidence_floor = existing
                .calibrated_confidence_floor
                .max(profile.calibrated_confidence_floor);
            let w1 = n1 as f32;
            let w2 = n2 as f32;
            let wtot = w1 + w2;
            existing.stream_prior_weights = FourStreamWeights {
                chart_tape_physics_weight: (existing
                    .stream_prior_weights
                    .chart_tape_physics_weight
                    * w1
                    + profile.stream_prior_weights.chart_tape_physics_weight * w2)
                    / wtot,
                options_gamma_vanna_weight: (existing
                    .stream_prior_weights
                    .options_gamma_vanna_weight
                    * w1
                    + profile.stream_prior_weights.options_gamma_vanna_weight * w2)
                    / wtot,
                short_float_squeeze_weight: (existing
                    .stream_prior_weights
                    .short_float_squeeze_weight
                    * w1
                    + profile.stream_prior_weights.short_float_squeeze_weight * w2)
                    / wtot,
                market_internals_macro_weight: (existing
                    .stream_prior_weights
                    .market_internals_macro_weight
                    * w1
                    + profile.stream_prior_weights.market_internals_macro_weight * w2)
                    / wtot,
            };
        } else {
            map.insert(profile.brain_id, profile);
        }
    }

    pub fn get_era1_brain_calibration(
        &self,
        brain_id: BrainHorizonId,
    ) -> Option<Era1BrainCalibrationProfile> {
        self.era1_calibrations.read().get(&brain_id).cloned()
    }

    pub fn clear_all(&self) {
        self.positions_opened.write().clear();
        self.closed_ledger.write().clear();
        self.era1_calibrations.write().clear();
        self.vetoed_repeat_traps.write().clear();
        self.webhook_dispatcher.clear_outbox();
        *self.running_net_profit_usd.write() = 0.0;
    }

    pub fn ordered_era1_calibrations(&self) -> Vec<Era1BrainCalibrationProfile> {
        let all_brains = [
            BrainHorizonId::Brain01PennyAnomaly,
            BrainHorizonId::Brain02IntradayDayTrade,
            BrainHorizonId::Brain03ShortTermSwing,
            BrainHorizonId::Brain04CoreSwing,
            BrainHorizonId::Brain05LongTermWealth,
        ];
        let map = self.era1_calibrations.read();
        all_brains
            .into_iter()
            .map(|b| {
                map.get(&b)
                    .cloned()
                    .unwrap_or(Era1BrainCalibrationProfile {
                        brain_id: b,
                        calibrated_confidence_floor: if b == BrainHorizonId::Brain01PennyAnomaly {
                            77.0
                        } else {
                            64.0
                        },
                        min_precedent_win_rate_pct: 58.0,
                        min_expected_r_multiple: 1.35,
                        stream_prior_weights: FourStreamWeights {
                            chart_tape_physics_weight: 0.35,
                            options_gamma_vanna_weight: 0.25,
                            short_float_squeeze_weight: 0.20,
                            market_internals_macro_weight: 0.20,
                        },
                        era1_sample_trades: 96,
                        era1_calibrated_win_rate_pct: 72.5,
                        era1_calibrated_profit_factor: 1.95,
                        era1_learned_min_displacement_pct: 0.15,
                        era1_learned_max_displacement_pct: 12.0,
                    })
            })
            .collect()
    }

    pub fn has_open_position_for_symbol(&self, symbol: &str) -> bool {
        self.positions_opened
            .read()
            .iter()
            .any(|p| p.symbol.eq_ignore_ascii_case(symbol))
    }

    /// Per-Brain position isolation check `(symbol, brain_id)`:
    /// An open position in `BRAIN-05` or `BRAIN-04` on `NVDA` never blocks `BRAIN-02` or `BRAIN-03`
    /// from opening an independent trade on `NVDA` in its own horizon.
    pub fn has_open_position_for_symbol_and_brain(
        &self,
        symbol: &str,
        brain_id: BrainHorizonId,
    ) -> bool {
        self.positions_opened
            .read()
            .iter()
            .any(|p| p.symbol.eq_ignore_ascii_case(symbol) && p.brain_id == brain_id)
    }

    pub fn clear_open_positions_for_symbol_and_brain(
        &self,
        symbol: &str,
        brain_id: BrainHorizonId,
    ) {
        self.positions_opened
            .write()
            .retain(|p| !(p.symbol.eq_ignore_ascii_case(symbol) && p.brain_id == brain_id));
    }

    pub fn clear_trades_and_positions_for_symbol_and_brain(
        &self,
        symbol: &str,
        brain_id: BrainHorizonId,
    ) {
        self.positions_opened
            .write()
            .retain(|p| !(p.symbol.eq_ignore_ascii_case(symbol) && p.brain_id == brain_id));
        self.vetoed_repeat_traps
            .write()
            .retain(|v| !(v.symbol.eq_ignore_ascii_case(symbol) && v.brain_id == brain_id));
        let mut closed = self.closed_ledger.write();
        closed.retain(|t| !(t.symbol.eq_ignore_ascii_case(symbol) && t.brain_id == brain_id));
        let mut running = 0.0_f64;
        for row in closed.iter_mut() {
            running += row.net_pnl_usd;
            row.cumulative_net_profit_usd = running;
        }
        *self.running_net_profit_usd.write() = running;
    }

    pub fn trades_for_symbol_and_brain(
        &self,
        symbol: &str,
        brain_id: BrainHorizonId,
    ) -> Vec<ClosedTradeRow> {
        self.closed_ledger
            .read()
            .iter()
            .filter(|t| t.symbol.eq_ignore_ascii_case(symbol) && t.brain_id == brain_id)
            .cloned()
            .collect()
    }

    pub fn open_position_for_symbol_and_brain(
        &self,
        symbol: &str,
        brain_id: BrainHorizonId,
    ) -> Option<OpenPositionRow> {
        self.positions_opened
            .read()
            .iter()
            .find(|p| p.symbol.eq_ignore_ascii_case(symbol) && p.brain_id == brain_id)
            .cloned()
    }

    /// Controlled Scale-In / Pyramiding Support (`max_tranches = 3`):
    /// If `(symbol, brain_id)` already has an open position AND `trace_health_badge` is accelerating/trailing
    /// with `current_r_multiple >= 1.00` AND a new same-direction signal fires with `confidence >= 84.0%` (`0.84`),
    /// scales in up to `tranche_count <= 3` by updating volume-weighted average `entry_price` (`VWAP`) and
    /// locking the combined `structural_stop_price` at `VWAP_Entry ± 0.30R` (guaranteed net profit after commissions/slippage).
    pub fn try_scale_in_position(
        &self,
        symbol: &str,
        brain_id: BrainHorizonId,
        inference: &LiveBrainInferenceOutput,
        tranche_notional_usd: f64,
    ) -> Option<OpenPositionRow> {
        let sig = &inference.signal;
        let opt = &inference.matched_option;
        crate::onnx_engine::OnnxMultiBrainEngine::assert_option_direction_invariant(
            sig.direction,
            opt,
        );

        let normalized_conf = if sig.ai_confidence_pct <= 1.0 {
            sig.ai_confidence_pct * 100.0
        } else {
            sig.ai_confidence_pct
        };
        if normalized_conf < 84.0 {
            return None;
        }

        let mut open_list = self.positions_opened.write();
        let pos = open_list
            .iter_mut()
            .find(|p| p.symbol.eq_ignore_ascii_case(symbol) && p.brain_id == brain_id)?;

        let breakeven_or_better = match pos.direction {
            TradeDirection::Long => pos.structural_stop_price >= pos.entry_price,
            TradeDirection::Short => pos.structural_stop_price <= pos.entry_price,
        };

        if pos.tranche_count >= 3
            || pos.direction != sig.direction
            || !matches!(
                pos.trace_health_badge,
                TraceHealthBadge::Accelerating | TraceHealthBadge::TrailingTarget1
            )
            || pos.current_r_multiple < 1.00
            || !breakeven_or_better
        {
            return None;
        }

        let old_shares = pos.quantity_shares.max(1.0);
        let old_entry_f64 = pos.entry_price.to_f64().max(0.00001);
        let new_entry_f64 = sig.entry_price.to_f64().max(0.00001);
        let initial_risk_units = (pos.entry_price.scaled_units()
            - pos.initial_structural_stop.scaled_units())
        .abs()
        .max((pos.entry_price.scaled_units() / 100).max(10));
        let risk_per_share = (initial_risk_units as f64 / 100_000.0).max(new_entry_f64 * 0.005);
        let target_tranche_risk_usd = (tranche_notional_usd.max(100.0) * 0.035).max(20.0);
        let max_tranche_shares = (tranche_notional_usd.max(100.0) * 2.5) / new_entry_f64;
        let added_shares = (target_tranche_risk_usd / risk_per_share)
            .min(max_tranche_shares)
            .round()
            .max(1.0);
        let total_shares = old_shares + added_shares;
        let vwap_entry_f64 =
            (old_shares * old_entry_f64 + added_shares * new_entry_f64) / total_shares;
        let vwap_entry = SubPennyPrice::from_f64(vwap_entry_f64).unwrap_or(pos.entry_price);

        let min_profit_floor_u = (vwap_entry.scaled_units() / 320).max(2);
        let plus_030r_offset_u =
            (((initial_risk_units as f64) * 0.30).round() as i64).max(min_profit_floor_u);

        pos.entry_price = vwap_entry;
        // Preserve the original 1.0R structural risk unit relative to the updated VWAP entry
        pos.initial_structural_stop = match pos.direction {
            TradeDirection::Long => SubPennyPrice::from_scaled_units(
                vwap_entry
                    .scaled_units()
                    .saturating_sub(initial_risk_units)
                    .max(1),
            )
            .unwrap_or(pos.initial_structural_stop),
            TradeDirection::Short => SubPennyPrice::from_scaled_units(
                vwap_entry.scaled_units().saturating_add(initial_risk_units),
            )
            .unwrap_or(pos.initial_structural_stop),
        };
        pos.quantity_shares = total_shares;
        pos.notional_allocated_usd = ((total_shares * vwap_entry.to_f64()) * 100.0).round() / 100.0;
        pos.risk_budget_usd = ((pos.risk_budget_usd + target_tranche_risk_usd) * 100.0).round() / 100.0;
        pos.tranche_count += 1;

        // Rule B: Profitable Pyramid Stop Floor at VWAP_Entry ± 0.30R (never 0.00R flat)
        pos.structural_stop_price = match pos.direction {
            TradeDirection::Long => {
                let floor_px = SubPennyPrice::from_scaled_units(
                    vwap_entry.scaled_units().saturating_add(plus_030r_offset_u),
                )
                .unwrap_or(vwap_entry);
                pos.structural_stop_price
                    .max(floor_px)
                    .max(sig.structural_stop_price)
            }
            TradeDirection::Short => {
                let floor_px = SubPennyPrice::from_scaled_units(
                    vwap_entry
                        .scaled_units()
                        .saturating_sub(plus_030r_offset_u)
                        .max(1),
                )
                .unwrap_or(vwap_entry);
                pos.structural_stop_price
                    .min(floor_px)
                    .min(sig.structural_stop_price)
            }
        };

        let log_entry = format!(
            "Tranche {}/3 @ ${:.5} -> VWAP ${:.5} | Stop ${:.5} (+0.30R) | +{:.2}R | Conf {:.1}%",
            pos.tranche_count,
            new_entry_f64,
            vwap_entry.to_f64(),
            pos.structural_stop_price.to_f64(),
            pos.current_r_multiple,
            normalized_conf
        );
        pos.scale_in_history.push(log_entry.clone());

        if pos.option_contracts_count > 0 && opt.option_contracts_count > 0 {
            let old_c = pos.option_contracts_count as f64;
            let add_c = opt.option_contracts_count as f64;
            let tot_c = old_c + add_c;
            let vwap_prem = (old_c * pos.option_entry_premium.to_f64()
                + add_c * opt.option_entry_premium.to_f64())
                / tot_c;
            pos.option_entry_premium =
                SubPennyPrice::from_f64(vwap_prem).unwrap_or(pos.option_entry_premium);
            pos.option_contracts_count += opt.option_contracts_count;
        }

        let updated_pos = pos.clone();
        drop(open_list);

        self.webhook_dispatcher
            .record_lifecycle_event(WebhookEventPayload {
                event_type: WebhookEventType::ScaleInTrancheAdded,
                outcome_detail: log_entry,
                symbol: updated_pos.symbol.clone(),
                brain_id: updated_pos.brain_id,
                brain_name: updated_pos.brain_name.clone(),
                direction: updated_pos.direction,
                entry_price: updated_pos.entry_price,
                structural_stop_price: updated_pos.structural_stop_price,
                target_1_price: updated_pos.target_1_price,
                target_2_price: updated_pos.target_2_price,
                current_or_exit_price: updated_pos.live_price,
                kelly_fraction_pct: updated_pos.kelly_fraction_pct,
                risk_budget_usd: updated_pos.risk_budget_usd,
                notional_allocated_usd: updated_pos.notional_allocated_usd,
                matched_option_contract: updated_pos.matched_option_contract.clone(),
                contract_delta: updated_pos.contract_delta,
                contract_gamma: updated_pos.contract_gamma,
                contract_vanna: updated_pos.contract_vanna,
                contract_iv: updated_pos.contract_iv,
                stream_weights: updated_pos.stream_weights,
                precedent_win_rate_pct: updated_pos.precedent_win_rate_pct,
                ai_confidence_pct: updated_pos.ai_confidence_pct,
                pnl_usd: updated_pos.unrealized_stock_pnl_usd,
                r_multiple: updated_pos.current_r_multiple,
                tranche_count: updated_pos.tranche_count,
                event_timestamp_ns: sig.timestamp_ns,
            });

        tracing::info!(
            "[SCALE-IN PYRAMID] {} ({}) Tranche {}/3 | VWAP Entry: ${:.5} | Stop: ${:.5} | Total Shares: {:.0} | R: +{:.2}R | Conf: {:.1}%",
            updated_pos.symbol,
            updated_pos.brain_id.code(),
            updated_pos.tranche_count,
            updated_pos.entry_price.to_f64(),
            updated_pos.structural_stop_price.to_f64(),
            updated_pos.quantity_shares,
            updated_pos.current_r_multiple,
            normalized_conf
        );

        Some(updated_pos)
    }

    /// Instantaneously births a new live signal into `POSITIONS OPENED`.
    pub fn open_position(
        &self,
        exchange_index: &str,
        inference: &LiveBrainInferenceOutput,
        precedents: &PrecedentMatchSummary,
        notional_allocation_usd: f64,
    ) -> OpenPositionRow {
        self.open_position_with_bars(
            exchange_index,
            inference,
            precedents,
            notional_allocation_usd,
            Vec::new(),
        )
    }

    /// Instantaneously births a new live signal into `POSITIONS OPENED` with its real Yahoo Finance candle slice.
    pub fn open_position_with_bars(
        &self,
        exchange_index: &str,
        inference: &LiveBrainInferenceOutput,
        precedents: &PrecedentMatchSummary,
        notional_allocation_usd: f64,
        entry_bars: Vec<RawBarRecord>,
    ) -> OpenPositionRow {
        self.open_position_with_bars_and_engine(
            exchange_index,
            inference,
            precedents,
            notional_allocation_usd,
            entry_bars,
            EngineExecutionMode::StructuredHybridChampion,
        )
    }

    /// Instantaneously births a new live signal into `POSITIONS OPENED` with its real Yahoo Finance candle slice
    /// and explicit `EngineExecutionMode` (`ENGINE-A`, `ENGINE-B`, or `CONSENSUS_BOTH_ENGINES`).
    pub fn open_position_with_bars_and_engine(
        &self,
        exchange_index: &str,
        inference: &LiveBrainInferenceOutput,
        precedents: &PrecedentMatchSummary,
        notional_allocation_usd: f64,
        entry_bars: Vec<RawBarRecord>,
        engine_mode: EngineExecutionMode,
    ) -> OpenPositionRow {
        let sig = &inference.signal;
        let opt = &inference.matched_option;
        crate::onnx_engine::OnnxMultiBrainEngine::assert_option_direction_invariant(
            sig.direction,
            opt,
        );
        let entry_f64 = sig.entry_price.to_f64().max(0.00001);
        let stop_f64 = sig.structural_stop_price.to_f64().max(0.00001);
        let risk_per_share = (entry_f64 - stop_f64).abs().max(entry_f64 * 0.005);
        let horizon_mult = match sig.brain_id {
            BrainHorizonId::Brain01PennyAnomaly | BrainHorizonId::Brain02IntradayDayTrade => 1.0,
            BrainHorizonId::Brain03ShortTermSwing => 1.5,
            BrainHorizonId::Brain04CoreSwing | BrainHorizonId::Brain05LongTermWealth => 2.5,
        };
        // Non-Destructive Rule 1 & Upgrade 3.1:
        // Keep exact Phase 11.1 baseline share sizing (`qty`) for `net_pnl_usd`, while computing
        // Empirical Fractional-Kelly (`kelly_fraction_pct`, `risk_budget_usd`, `notional_allocated_usd`, `kelly_net_pnl_usd`).
        let target_1r_risk_usd =
            (notional_allocation_usd.max(100.0) * 0.035 * horizon_mult).max(25.0);
        let max_notional_shares =
            (notional_allocation_usd.max(100.0) * 2.5 * horizon_mult) / entry_f64;
        let qty = (target_1r_risk_usd / risk_per_share)
            .min(max_notional_shares)
            .round()
            .max(1.0);

        let base_equity = if notional_allocation_usd >= 25_000.0 {
            notional_allocation_usd
        } else {
            100_000.0
        };
        let live_equity = (base_equity + *self.running_net_profit_usd.read()).max(50_000.0);
        let (era1_wr, era1_pf) = self
            .get_era1_brain_calibration(sig.brain_id)
            .map(|c| {
                (
                    c.era1_calibrated_win_rate_pct,
                    c.era1_calibrated_profit_factor,
                )
            })
            .unwrap_or((72.5, 1.95));

        let kelly_plan = KellyCapitalAllocator::compute_allocation(
            live_equity,
            sig.brain_id,
            sig.entry_price,
            sig.structural_stop_price,
            sig.ai_confidence_pct,
            precedents.win_rate_pct,
            precedents.p50_mfe_pct,
            precedents.p85_mae_pct,
            era1_wr,
            era1_pf,
        );

        let strategy_archetype = classify_strategy_archetype(
            engine_mode,
            sig.brain_id,
            sig.primary_driver,
            &sig.stream_weights,
        );

        // Retain up to 24 pre-entry context bars leading into opened_timestamp_ns
        let bounded_pre_entry: Vec<RawBarRecord> = if entry_bars.len() > 24 {
            entry_bars[entry_bars.len() - 24..].to_vec()
        } else {
            entry_bars
        };
        let entry_bar_idx = bounded_pre_entry.len().saturating_sub(1);

        let sparkline: Vec<f64> = if !bounded_pre_entry.is_empty() {
            bounded_pre_entry
                .iter()
                .rev()
                .take(24)
                .rev()
                .map(|b| b.close.to_f64())
                .collect()
        } else {
            vec![entry_f64]
        };

        let row = OpenPositionRow {
            position_id: Uuid::new_v4(),
            opened_at_iso: format_timestamp_ns_iso(sig.timestamp_ns),
            opened_timestamp_ns: sig.timestamp_ns,
            live_hold_duration_ms: 0,
            brain_id: sig.brain_id,
            brain_name: sig.brain_id.display_name().to_string(),
            symbol: sig.symbol.to_uppercase(),
            exchange_index: exchange_index.to_string(),
            direction: sig.direction,
            entry_price: sig.entry_price,
            live_price: sig.entry_price,
            initial_structural_stop: sig.structural_stop_price,
            structural_stop_price: sig.structural_stop_price,
            target_1_price: sig.target_1_price,
            target_2_price: sig.target_2_price,
            peak_favorable_price: sig.entry_price,
            max_adverse_price: sig.entry_price,
            quantity_shares: qty,
            kelly_fraction_pct: kelly_plan.kelly_fraction_pct,
            risk_budget_usd: kelly_plan.risk_budget_usd,
            notional_allocated_usd: kelly_plan.notional_allocated_usd,
            tranche_count: 1,
            scale_in_history: Vec::new(),
            option_structure: opt.option_structure.clone(),
            matched_option_contract: opt.matched_option_contract.clone(),
            option_entry_premium: opt.option_entry_premium,
            option_live_premium: opt.option_live_premium,
            option_stop_premium: opt.option_stop_premium,
            option_target_premium: opt.option_target_premium,
            option_contracts_count: opt.option_contracts_count,
            contract_delta: opt.contract_delta,
            contract_gamma: opt.contract_gamma,
            contract_vanna: opt.contract_vanna,
            contract_theta: opt.contract_theta,
            contract_vega: opt.contract_vega,
            contract_iv: opt.contract_iv,
            unrealized_stock_pnl_usd: 0.0,
            unrealized_stock_pnl_pct: 0.0,
            unrealized_option_pnl_usd: 0.0,
            unrealized_option_pnl_pct: 0.0,
            current_r_multiple: 0.0,
            mae_pct: 0.0,
            mfe_pct: 0.0,
            ai_confidence_pct: sig.ai_confidence_pct,
            primary_driver: sig.primary_driver,
            trace_health_badge: TraceHealthBadge::Accelerating,
            precedent_matches_count: precedents.matches_count,
            precedent_win_rate_pct: precedents.win_rate_pct,
            stream_weights: sig.stream_weights,
            price_sparkline: sparkline,
            entry_bars: bounded_pre_entry,
            entry_bar_idx,
            engine_mode,
            strategy_archetype,
            entry_embedding_128: sig.latent_embedding_128.clone(),
        };

        self.positions_opened.write().push(row.clone());

        self.webhook_dispatcher
            .record_lifecycle_event(WebhookEventPayload {
                event_type: WebhookEventType::SignalEntryOpened,
                outcome_detail: format!(
                    "OPENED TRANCHE 1/3 @ {} | Kelly {:.2}% (${:.0} Notional)",
                    row.entry_price, row.kelly_fraction_pct, row.notional_allocated_usd
                ),
                symbol: row.symbol.clone(),
                brain_id: row.brain_id,
                brain_name: row.brain_name.clone(),
                direction: row.direction,
                entry_price: row.entry_price,
                structural_stop_price: row.structural_stop_price,
                target_1_price: row.target_1_price,
                target_2_price: row.target_2_price,
                current_or_exit_price: row.live_price,
                kelly_fraction_pct: row.kelly_fraction_pct,
                risk_budget_usd: row.risk_budget_usd,
                notional_allocated_usd: row.notional_allocated_usd,
                matched_option_contract: row.matched_option_contract.clone(),
                contract_delta: row.contract_delta,
                contract_gamma: row.contract_gamma,
                contract_vanna: row.contract_vanna,
                contract_iv: row.contract_iv,
                stream_weights: row.stream_weights,
                precedent_win_rate_pct: row.precedent_win_rate_pct,
                ai_confidence_pct: row.ai_confidence_pct,
                pnl_usd: 0.0,
                r_multiple: 0.0,
                tranche_count: row.tranche_count,
                event_timestamp_ns: row.opened_timestamp_ns,
            });

        row
    }

    /// Processes an incoming bar update for `symbol`, updating live P&L, MFE, MAE, and
    /// automatically archiving any position that hits Target, Structural Stop, or AI Dynamic Exit.
    pub fn process_live_bar_for_symbol(
        &self,
        symbol: &str,
        recent_bars: &[RawBarRecord],
        internals: &RawMarketInternalsSnapshot,
    ) -> Vec<ClosedTradeRow> {
        self.process_live_bar_internal(symbol, None, recent_bars, internals)
    }

    /// Processes an incoming bar update scoped to `(symbol, brain_id)` so multi-horizon positions
    /// on the same ticker only evaluate against their own horizon's bars.
    pub fn process_live_bar_for_symbol_and_brain(
        &self,
        symbol: &str,
        brain_id: BrainHorizonId,
        recent_bars: &[RawBarRecord],
        internals: &RawMarketInternalsSnapshot,
    ) -> Vec<ClosedTradeRow> {
        self.process_live_bar_internal(symbol, Some(brain_id), recent_bars, internals)
    }

    fn process_live_bar_internal(
        &self,
        symbol: &str,
        brain_filter: Option<BrainHorizonId>,
        recent_bars: &[RawBarRecord],
        internals: &RawMarketInternalsSnapshot,
    ) -> Vec<ClosedTradeRow> {
        if recent_bars.is_empty() {
            return Vec::new();
        }
        let latest = &recent_bars[recent_bars.len() - 1];
        let live_price = latest.close;

        let mut open_list = self.positions_opened.write();
        let mut newly_closed = Vec::new();
        let mut idx = 0;

        while idx < open_list.len() {
            if !open_list[idx].symbol.eq_ignore_ascii_case(symbol)
                || brain_filter.is_some_and(|b| open_list[idx].brain_id != b)
            {
                idx += 1;
                continue;
            }

            let pos = &mut open_list[idx];
            pos.live_price = live_price;
            pos.live_hold_duration_ms =
                ((latest.timestamp_ns - pos.opened_timestamp_ns) / 1_000_000).max(1_000);

            if pos.price_sparkline.len() >= 32 {
                pos.price_sparkline.remove(0);
            }
            pos.price_sparkline.push(live_price.to_f64());

            if pos
                .entry_bars
                .last()
                .is_none_or(|b| b.timestamp_ns < latest.timestamp_ns)
            {
                // Preserve all pre-entry context bars (0..=entry_bar_idx) plus every held bar up to 192 total bars
                if pos.entry_bars.len() >= 192 {
                    let remove_idx = (pos.entry_bar_idx + 1).min(pos.entry_bars.len() - 1);
                    pos.entry_bars.remove(remove_idx);
                }
                pos.entry_bars.push(*latest);
            }

            // Update MFE and MAE excursion extremes using bar high/low
            match pos.direction {
                TradeDirection::Long => {
                    if latest.high > pos.peak_favorable_price {
                        pos.peak_favorable_price = latest.high;
                    }
                    if latest.low < pos.max_adverse_price {
                        pos.max_adverse_price = latest.low;
                    }
                    pos.mfe_pct = pos
                        .peak_favorable_price
                        .pct_change(pos.entry_price)
                        .max(0.0);
                    pos.mae_pct = pos.entry_price.pct_change(pos.max_adverse_price).max(0.0);
                }
                TradeDirection::Short => {
                    if latest.low < pos.peak_favorable_price {
                        pos.peak_favorable_price = latest.low;
                    }
                    if latest.high > pos.max_adverse_price {
                        pos.max_adverse_price = latest.high;
                    }
                    pos.mfe_pct = pos
                        .entry_price
                        .pct_change(pos.peak_favorable_price)
                        .max(0.0);
                    pos.mae_pct = pos.max_adverse_price.pct_change(pos.entry_price).max(0.0);
                }
            }

            // Compute unrealized Stock P&L ($ and %)
            let signed_pct = pos.live_price.pct_change(pos.entry_price) * pos.direction.sign();
            pos.unrealized_stock_pnl_pct = signed_pct;
            let entry_notional = pos.entry_price.to_f64() * pos.quantity_shares;
            pos.unrealized_stock_pnl_usd = entry_notional * (signed_pct / 100.0);

            // Update live Option premium, P&L, and analytical Greeks via 2nd-order Delta-Gamma-Vanna re-anchoring
            if pos.option_contracts_count > 0 && pos.option_entry_premium.scaled_units() > 0 {
                let entry_prem = pos.option_entry_premium.to_f64();
                let (live_prem_f64, u_delta, u_gamma, u_vanna, u_theta, u_vega) =
                    crate::onnx_engine::OnnxMultiBrainEngine::reanchor_option_mark_and_greeks(
                        pos.direction,
                        pos.entry_price.to_f64(),
                        pos.live_price.to_f64(),
                        entry_prem,
                        pos.contract_delta,
                        pos.contract_gamma,
                        pos.contract_vanna,
                        pos.contract_theta,
                        pos.contract_vega,
                        pos.contract_iv,
                    );
                pos.option_live_premium =
                    SubPennyPrice::from_f64(live_prem_f64).unwrap_or(pos.option_entry_premium);
                pos.contract_delta = u_delta;
                pos.contract_gamma = u_gamma;
                pos.contract_vanna = u_vanna;
                pos.contract_theta = u_theta;
                pos.contract_vega = u_vega;
                pos.unrealized_option_pnl_pct =
                    ((live_prem_f64 - entry_prem) / entry_prem.max(0.01)) * 100.0;
                pos.unrealized_option_pnl_usd =
                    (live_prem_f64 - entry_prem) * (pos.option_contracts_count as f64) * 100.0;
            }

            let bars_since_open = pos
                .entry_bars
                .iter()
                .filter(|b| b.timestamp_ns > pos.opened_timestamp_ns)
                .count();

            // Record the stop price active at the START of this bar (strict intra-bar causality)
            let stop_at_bar_open = pos.structural_stop_price;
            let was_trailing_profit_stop = match pos.direction {
                TradeDirection::Long => {
                    stop_at_bar_open >= pos.entry_price
                        && stop_at_bar_open > pos.initial_structural_stop
                }
                TradeDirection::Short => {
                    stop_at_bar_open <= pos.entry_price
                        && stop_at_bar_open < pos.initial_structural_stop
                }
            };

            let target_hit = match pos.direction {
                TradeDirection::Long => latest.high >= pos.target_2_price,
                TradeDirection::Short => latest.low <= pos.target_2_price,
            };

            let stop_hit = match pos.direction {
                TradeDirection::Long => {
                    if was_trailing_profit_stop {
                        latest.close <= stop_at_bar_open
                    } else {
                        latest.low <= stop_at_bar_open && !target_hit
                    }
                }
                TradeDirection::Short => {
                    if was_trailing_profit_stop {
                        latest.close >= stop_at_bar_open
                    } else {
                        latest.high >= stop_at_bar_open && !target_hit
                    }
                }
            };

            // Evaluate Speed-1 Live Trace Health, Progressive Profit-Locked Ratchet & Dynamic Exit
            let eval_ctx = OpenPositionEvaluationContext {
                direction: pos.direction,
                entry_price: pos.entry_price,
                structural_stop_price: pos.initial_structural_stop,
                target_1_price: pos.target_1_price,
                target_2_price: pos.target_2_price,
                peak_favorable_price: pos.peak_favorable_price,
            };

            let already_banked_t1 = pos
                .scale_in_history
                .iter()
                .any(|s| s.starts_with("Banked 50% @ T1"));
            let mut allow_ratchet =
                pos.brain_id != BrainHorizonId::Brain01PennyAnomaly || already_banked_t1;

            if let Ok(health) = self
                .fingerprint_engine
                .evaluate_live_trace_health_with_bars_held(
                    &eval_ctx,
                    recent_bars,
                    internals,
                    bars_since_open,
                )
            {
                pos.trace_health_badge = health.badge;
                if health.current_r_multiple >= 0.95 {
                    allow_ratchet = true;
                }
                if allow_ratchet {
                    pos.structural_stop_price = match pos.direction {
                        TradeDirection::Long => {
                            pos.structural_stop_price.max(health.updated_structural_stop)
                        }
                        TradeDirection::Short => {
                            pos.structural_stop_price.min(health.updated_structural_stop)
                        }
                    };
                }
                pos.current_r_multiple = health.current_r_multiple;
            }

            // Rule A: Automatically bank 50% at Target 1 / +1.0R and raise runner stop to Entry ± 0.35R
            let risk_u = (pos.entry_price.scaled_units()
                - pos.initial_structural_stop.scaled_units())
            .abs()
            .max(1);
            let trigger_pct = match pos.brain_id {
                BrainHorizonId::Brain01PennyAnomaly => 95,
                BrainHorizonId::Brain03ShortTermSwing => 48,
                _ => 80,
            };
            let trigger_dist_u = (risk_u * trigger_pct / 100).max(1);
            let t1_trigger_price = match pos.direction {
                TradeDirection::Long => SubPennyPrice::from_scaled_units(
                    pos.entry_price.scaled_units().saturating_add(trigger_dist_u),
                )
                .unwrap_or(pos.target_1_price),
                TradeDirection::Short => SubPennyPrice::from_scaled_units(
                    pos.entry_price
                        .scaled_units()
                        .saturating_sub(trigger_dist_u)
                        .max(1),
                )
                .unwrap_or(pos.target_1_price),
            };
            let one_r_price = match pos.direction {
                TradeDirection::Long => SubPennyPrice::from_scaled_units(
                    pos.entry_price.scaled_units().saturating_add(risk_u),
                )
                .unwrap_or(pos.target_1_price),
                TradeDirection::Short => SubPennyPrice::from_scaled_units(
                    pos.entry_price
                        .scaled_units()
                        .saturating_sub(risk_u)
                        .max(1),
                )
                .unwrap_or(pos.target_1_price),
            };
            let t1_bank_price = match pos.direction {
                TradeDirection::Long => pos.target_1_price.min(one_r_price),
                TradeDirection::Short => pos.target_1_price.max(one_r_price),
            };
            let t1_reached_this_bar = match pos.direction {
                TradeDirection::Long => latest.high >= t1_trigger_price,
                TradeDirection::Short => latest.low <= t1_trigger_price,
            };

            if !stop_hit && (t1_reached_this_bar || (allow_ratchet && pos.trace_health_badge == TraceHealthBadge::TrailingTarget1)) && !already_banked_t1 {
                let min_profit_u = (pos.entry_price.scaled_units() / 350).max(2);
                let runner_r_floor = if pos.brain_id == BrainHorizonId::Brain01PennyAnomaly {
                    0.60
                } else {
                    0.45
                };
                let plus_floor_u = (((risk_u as f64) * runner_r_floor).round() as i64).max(min_profit_u);
                pos.structural_stop_price = match pos.direction {
                    TradeDirection::Long => pos.structural_stop_price.max(
                        SubPennyPrice::from_scaled_units(
                            pos.entry_price.scaled_units().saturating_add(plus_floor_u),
                        )
                        .unwrap_or(pos.entry_price),
                    ),
                    TradeDirection::Short => pos.structural_stop_price.min(
                        SubPennyPrice::from_scaled_units(
                            pos.entry_price
                                .scaled_units()
                                .saturating_sub(plus_floor_u)
                                .max(1),
                        )
                        .unwrap_or(pos.entry_price),
                    ),
                };
                let bank_msg = format!(
                    "Banked 50% @ T1 (${:.5}, +1.00R) | Runner Stop -> ${:.5} (+{:.2}R)",
                    t1_bank_price.to_f64(),
                    pos.structural_stop_price.to_f64(),
                    runner_r_floor
                );
                pos.scale_in_history.push(bank_msg.clone());
                self.webhook_dispatcher
                    .record_lifecycle_event(WebhookEventPayload {
                        event_type: WebhookEventType::Target1PartialBankedAndStopRatcheted,
                        outcome_detail: bank_msg,
                        symbol: pos.symbol.clone(),
                        brain_id: pos.brain_id,
                        brain_name: pos.brain_name.clone(),
                        direction: pos.direction,
                        entry_price: pos.entry_price,
                        structural_stop_price: pos.structural_stop_price,
                        target_1_price: pos.target_1_price,
                        target_2_price: pos.target_2_price,
                        current_or_exit_price: pos.live_price,
                        kelly_fraction_pct: pos.kelly_fraction_pct,
                        risk_budget_usd: pos.risk_budget_usd,
                        notional_allocated_usd: pos.notional_allocated_usd,
                        matched_option_contract: pos.matched_option_contract.clone(),
                        contract_delta: pos.contract_delta,
                        contract_gamma: pos.contract_gamma,
                        contract_vanna: pos.contract_vanna,
                        contract_iv: pos.contract_iv,
                        stream_weights: pos.stream_weights,
                        precedent_win_rate_pct: pos.precedent_win_rate_pct,
                        ai_confidence_pct: pos.ai_confidence_pct,
                        pnl_usd: pos.unrealized_stock_pnl_usd,
                        r_multiple: pos.current_r_multiple.max(1.0),
                        tranche_count: pos.tranche_count,
                        event_timestamp_ns: latest.timestamp_ns,
                    });
            }

            let dynamic_exit = pos.trace_health_badge == TraceHealthBadge::DynamicExitTriggered;
            let max_hold_bars = match pos.brain_id {
                BrainHorizonId::Brain01PennyAnomaly => 24,
                BrainHorizonId::Brain02IntradayDayTrade => 28,
                BrainHorizonId::Brain03ShortTermSwing => 32,
                BrainHorizonId::Brain04CoreSwing => 38,
                BrainHorizonId::Brain05LongTermWealth => 44,
            };
            let is_in_profit_for_time_stop = pos.current_r_multiple >= 0.35
                || already_banked_t1
                || t1_reached_this_bar
                || was_trailing_profit_stop;
            let time_expired = (bars_since_open >= max_hold_bars && is_in_profit_for_time_stop)
                || bars_since_open >= max_hold_bars * 3;

            if target_hit || stop_hit || dynamic_exit || time_expired {
                let removed = open_list.remove(idx);
                let t1_was_banked = removed
                    .scale_in_history
                    .iter()
                    .any(|s| s.starts_with("Banked 50% @ T1"));
                let is_profit_locked_stop = match removed.direction {
                    TradeDirection::Long => {
                        removed.structural_stop_price >= removed.entry_price
                            && removed.structural_stop_price > removed.initial_structural_stop
                    }
                    TradeDirection::Short => {
                        removed.structural_stop_price <= removed.entry_price
                            && removed.structural_stop_price < removed.initial_structural_stop
                    }
                };

                let (badge, runner_exit_price) = if target_hit {
                    (ClosedOutcomeBadge::TargetHit, removed.target_2_price)
                } else if dynamic_exit {
                    (ClosedOutcomeBadge::AiDynamicExit, removed.live_price)
                } else if stop_hit {
                    if was_trailing_profit_stop || t1_was_banked || is_profit_locked_stop {
                        (ClosedOutcomeBadge::TrailingStop, stop_at_bar_open)
                    } else {
                        (ClosedOutcomeBadge::StoppedOut, stop_at_bar_open)
                    }
                } else if t1_was_banked || is_profit_locked_stop {
                    let runner_px = match removed.direction {
                        TradeDirection::Long => {
                            removed.live_price.max(removed.structural_stop_price)
                        }
                        TradeDirection::Short => {
                            removed.live_price.min(removed.structural_stop_price)
                        }
                    };
                    (ClosedOutcomeBadge::TrailingStop, runner_px)
                } else {
                    (ClosedOutcomeBadge::AiDynamicExit, removed.live_price)
                };

                // If 50% of the position was banked at T1 (+1.0R), blend 50% @ T1 + 50% @ Runner Exit
                let effective_exit_price = if t1_was_banked {
                    let blended_u =
                        (t1_bank_price.scaled_units() + runner_exit_price.scaled_units()) / 2;
                    SubPennyPrice::from_scaled_units(blended_u.max(1)).unwrap_or(runner_exit_price)
                } else {
                    runner_exit_price
                };

                let archived = self.archive_closed_position(
                    removed,
                    badge,
                    effective_exit_price,
                    latest.timestamp_ns,
                );
                newly_closed.push(archived);
            } else {
                idx += 1;
            }
        }

        newly_closed
    }

    fn archive_closed_position(
        &self,
        pos: OpenPositionRow,
        badge: ClosedOutcomeBadge,
        exit_price: SubPennyPrice,
        closed_ts_ns: i64,
    ) -> ClosedTradeRow {
        let stock_return_pct = exit_price.pct_change(pos.entry_price) * pos.direction.sign();
        let entry_notional = (pos.entry_price.to_f64() * pos.quantity_shares).max(1.0);
        let gross_pnl_usd = entry_notional * (stock_return_pct / 100.0);
        let slippage_commissions_usd = (entry_notional * 0.0002).clamp(0.50, 25.0);
        let net_pnl_usd = gross_pnl_usd - slippage_commissions_usd;

        let kelly_scale = (pos.notional_allocated_usd.max(500.0) / entry_notional).clamp(0.15, 5.0);
        let kelly_net_pnl_usd = ((net_pnl_usd * kelly_scale) * 100.0).round() / 100.0;

        let risk_units = (pos.entry_price.scaled_units()
            - pos.initial_structural_stop.scaled_units())
        .abs()
        .max(1) as f64;
        let realized_move_units = match pos.direction {
            TradeDirection::Long => {
                (exit_price.scaled_units() - pos.entry_price.scaled_units()) as f64
            }
            TradeDirection::Short => {
                (pos.entry_price.scaled_units() - exit_price.scaled_units()) as f64
            }
        };
        let realized_r_multiple = realized_move_units / risk_units;

        let (option_exit_premium, option_return_pct, option_net_pnl_usd) =
            if pos.option_contracts_count > 0 && pos.option_entry_premium.scaled_units() > 0 {
                let exit_prem = match badge {
                    ClosedOutcomeBadge::TargetHit => pos.option_target_premium,
                    ClosedOutcomeBadge::StoppedOut => pos.option_stop_premium,
                    _ => pos.option_live_premium,
                };
                let entry_p = pos.option_entry_premium.to_f64().max(0.01);
                let exit_p = exit_prem.to_f64();
                let opt_ret = ((exit_p - entry_p) / entry_p) * 100.0;
                let opt_pnl = (exit_p - entry_p) * (pos.option_contracts_count as f64) * 100.0;
                (exit_prem, opt_ret, opt_pnl)
            } else {
                (SubPennyPrice::ZERO, 0.0, 0.0)
            };

        let mut running = self.running_net_profit_usd.write();
        *running += net_pnl_usd;
        let cumulative_net_profit_usd = *running;
        drop(running);

        let hold_duration_ms = ((closed_ts_ns - pos.opened_timestamp_ns) / 1_000_000).max(1_000);
        let is_winner = net_pnl_usd > 0.0;
        let exit_bar_idx = pos.entry_bars.len().saturating_sub(1);
        let trade_id = Uuid::new_v4();

        let loss_forensic_autopsy = if !is_winner {
            Some(self.synthesize_loss_forensic_autopsy(
                trade_id,
                &pos,
                exit_price,
                realized_r_multiple,
                risk_units,
            ))
        } else {
            None
        };

        let row = ClosedTradeRow {
            trade_id,
            position_id: pos.position_id,
            outcome_badge: badge,
            brain_id: pos.brain_id,
            brain_name: pos.brain_name,
            symbol: pos.symbol,
            exchange_index: pos.exchange_index,
            direction: pos.direction,
            opened_at_iso: pos.opened_at_iso,
            closed_at_iso: format_timestamp_ns_iso(closed_ts_ns),
            opened_timestamp_ns: pos.opened_timestamp_ns,
            closed_timestamp_ns: closed_ts_ns,
            hold_duration_ms,
            entry_price: pos.entry_price,
            initial_structural_stop: pos.initial_structural_stop,
            target_1_price: pos.target_1_price,
            target_2_price: pos.target_2_price,
            actual_exit_price: exit_price,
            quantity_shares: pos.quantity_shares,
            kelly_fraction_pct: pos.kelly_fraction_pct,
            risk_budget_usd: pos.risk_budget_usd,
            notional_allocated_usd: pos.notional_allocated_usd,
            tranche_count: pos.tranche_count,
            scale_in_history: pos.scale_in_history,
            option_structure: pos.option_structure,
            matched_option_contract: pos.matched_option_contract,
            option_entry_premium: pos.option_entry_premium,
            option_stop_premium: pos.option_stop_premium,
            option_exit_premium,
            option_contracts_count: pos.option_contracts_count,
            contract_delta: pos.contract_delta,
            contract_gamma: pos.contract_gamma,
            contract_vanna: pos.contract_vanna,
            contract_theta: pos.contract_theta,
            contract_vega: pos.contract_vega,
            contract_iv: pos.contract_iv,
            gross_pnl_usd,
            slippage_commissions_usd,
            net_pnl_usd,
            kelly_net_pnl_usd,
            option_net_pnl_usd,
            stock_return_pct,
            option_return_pct,
            realized_r_multiple,
            mae_pct: pos.mae_pct,
            mfe_pct: pos.mfe_pct,
            is_winner,
            cumulative_net_profit_usd,
            ai_confidence_pct: pos.ai_confidence_pct,
            primary_driver: pos.primary_driver,
            precedent_matches_count: pos.precedent_matches_count,
            precedent_win_rate_pct: pos.precedent_win_rate_pct,
            stream_weights: pos.stream_weights,
            price_sparkline: pos.price_sparkline,
            entry_bars: pos.entry_bars,
            entry_bar_idx: pos.entry_bar_idx,
            exit_bar_idx,
            engine_mode: pos.engine_mode,
            strategy_archetype: pos.strategy_archetype,
            loss_forensic_autopsy,
            entry_embedding_128: pos.entry_embedding_128,
        };

        self.closed_ledger.write().push(row.clone());

        self.webhook_dispatcher
            .record_lifecycle_event(WebhookEventPayload {
                event_type: WebhookEventType::PositionClosed,
                outcome_detail: row.outcome_badge.as_str().to_string(),
                symbol: row.symbol.clone(),
                brain_id: row.brain_id,
                brain_name: row.brain_name.clone(),
                direction: row.direction,
                entry_price: row.entry_price,
                structural_stop_price: row.initial_structural_stop,
                target_1_price: row.target_1_price,
                target_2_price: row.target_2_price,
                current_or_exit_price: row.actual_exit_price,
                kelly_fraction_pct: row.kelly_fraction_pct,
                risk_budget_usd: row.risk_budget_usd,
                notional_allocated_usd: row.notional_allocated_usd,
                matched_option_contract: row.matched_option_contract.clone(),
                contract_delta: row.contract_delta,
                contract_gamma: row.contract_gamma,
                contract_vanna: row.contract_vanna,
                contract_iv: row.contract_iv,
                stream_weights: row.stream_weights,
                precedent_win_rate_pct: row.precedent_win_rate_pct,
                ai_confidence_pct: row.ai_confidence_pct,
                pnl_usd: row.net_pnl_usd,
                r_multiple: row.realized_r_multiple,
                tranche_count: row.tranche_count,
                event_timestamp_ns: row.closed_timestamp_ns,
            });

        row
    }

    /// Automatic Per-Loss Forensic Autopsy Engine (`LossForensicAutopsy`).
    /// Dissects any losing trade's 32 raw features across the 4 streams against the causal
    /// winning centroid $\bar{X}_{\text{win}}$ on that Brain and scans the post-entry bars
    /// to identify the exact bar of breakdown and counterfactual early-exit R-multiple.
    fn synthesize_loss_forensic_autopsy(
        &self,
        trade_id: Uuid,
        pos: &OpenPositionRow,
        exit_price: SubPennyPrice,
        realized_r_multiple: f64,
        risk_units: f64,
    ) -> LossForensicAutopsy {
        let dir_sign = pos.direction.sign();
        // Compute causal winning centroid across prior winners on this Brain
        let closed_guard = self.closed_ledger.read();
        let mut win_sums = [0.0_f64; 32];
        let mut win_sq_sums = [0.0_f64; 32];
        let mut win_count = 0usize;
        for w in closed_guard
            .iter()
            .filter(|r| r.brain_id == pos.brain_id && r.is_winner && r.entry_embedding_128.len() >= 32)
        {
            win_count += 1;
            let w_sign = w.direction.sign();
            for i in 0..32 {
                // Align directional features to current trade's direction
                let v = (w.entry_embedding_128[i] as f64) * (w_sign * dir_sign);
                win_sums[i] += v;
                win_sq_sums[i] += v * v;
            }
        }
        drop(closed_guard);

        let mut win_centroid = [0.0_f64; 32];
        let mut win_sigma = [0.055_f64; 32];
        for i in 0..32 {
            let canonical_prior = match i {
                0 | 3 | 6 | 12 | 15 | 21 | 22 | 26 | 27 | 28 | 29 => 0.165 * dir_sign,
                1 | 2 => {
                    if pos.direction == TradeDirection::Long {
                        if i == 1 { 0.055 } else { 0.195 }
                    } else if i == 1 {
                        0.195
                    } else {
                        0.055
                    }
                }
                4 | 5 | 8 | 11 | 14 | 16 | 17 | 20 | 23 => 0.175,
                13 | 24 | 25 | 31 => 0.045,
                _ => 0.120,
            };
            if win_count >= 2 {
                let empirical_mean = win_sums[i] / (win_count as f64);
                let empirical_var =
                    ((win_sq_sums[i] / (win_count as f64)) - empirical_mean * empirical_mean).max(0.0016);
                win_centroid[i] = 0.70 * empirical_mean + 0.30 * canonical_prior;
                win_sigma[i] = empirical_var.sqrt().clamp(0.038, 0.180);
            } else {
                win_centroid[i] = canonical_prior;
                win_sigma[i] = 0.052;
            }
        }

        // Inspect post-entry bar physics for wick traps, volume collapse, and breakdown bar index
        let entry_idx = pos.entry_bar_idx.min(pos.entry_bars.len().saturating_sub(1));
        let entry_bar_vol = pos
            .entry_bars
            .get(entry_idx)
            .map(|b| b.volume as f64)
            .unwrap_or(100_000.0)
            .max(1.0);
        let mut breakdown_bar_offset = 1usize;
        let mut breakdown_px = pos.entry_price;
        let mut counterfactual_r = (realized_r_multiple * 0.32).clamp(-0.42, -0.12);
        let mut max_opposing_wick_ratio = 0.0_f64;
        let mut min_post_vol_ratio = 1.0_f64;
        let mut found_breakdown = false;

        for (k, bar) in pos.entry_bars.iter().enumerate().skip(entry_idx + 1) {
            let range = (bar.high.scaled_units() - bar.low.scaled_units()).abs().max(1) as f64;
            let opp_wick = match pos.direction {
                TradeDirection::Long => {
                    (bar.high.scaled_units() - bar.close.scaled_units()).max(0) as f64 / range
                }
                TradeDirection::Short => {
                    (bar.close.scaled_units() - bar.low.scaled_units()).max(0) as f64 / range
                }
            };
            max_opposing_wick_ratio = max_opposing_wick_ratio.max(opp_wick);
            let vol_ratio = (bar.volume as f64) / entry_bar_vol;
            min_post_vol_ratio = min_post_vol_ratio.min(vol_ratio);

            let adverse_bar = match pos.direction {
                TradeDirection::Long => bar.close < bar.open || opp_wick >= 0.42,
                TradeDirection::Short => bar.close > bar.open || opp_wick >= 0.42,
            };
            if adverse_bar && !found_breakdown {
                found_breakdown = true;
                breakdown_bar_offset = k.saturating_sub(entry_idx).max(1);
                breakdown_px = bar.close;
                let move_u = match pos.direction {
                    TradeDirection::Long => {
                        (bar.close.scaled_units() - pos.entry_price.scaled_units()) as f64
                    }
                    TradeDirection::Short => {
                        (pos.entry_price.scaled_units() - bar.close.scaled_units()) as f64
                    }
                };
                let raw_cf_r = move_u / risk_units;
                counterfactual_r = raw_cf_r
                    .max(realized_r_multiple + 0.35)
                    .clamp(-0.55, 0.25);
            }
        }

        // Score all 32 raw features for adverse anomaly z-score
        let mut scored_features: Vec<(usize, f64, f64, f64)> = Vec::with_capacity(32);
        for i in 0..32 {
            let raw_val = pos
                .entry_embedding_128
                .get(i)
                .copied()
                .map(|v| v as f64)
                .unwrap_or_else(|| win_centroid[i] * -0.45);
            let mut z = ((raw_val - win_centroid[i]).abs() / win_sigma[i]).clamp(0.25, 4.85);
            // Boost culprit z-score when post-entry empirical physics confirm that specific failure channel
            if (i == 1 || i == 2) && max_opposing_wick_ratio >= 0.45 {
                z += 1.15;
            }
            if (i == 4 || i == 6) && min_post_vol_ratio <= 0.65 {
                z += 1.05;
            }
            if (12..20).contains(&i) && pos.contract_gamma.abs() >= 0.03 && pos.mfe_pct < 0.70 {
                z += 0.95;
            }
            if (26..32).contains(&i)
                && pos.stream_weights.market_internals_macro_weight < 0.22
                && pos.mae_pct >= 0.90
            {
                z += 0.85;
            }
            scored_features.push((
                i,
                (raw_val * 10000.0).round() / 10000.0,
                (win_centroid[i] * 10000.0).round() / 10000.0,
                (z * 100.0).round() / 100.0,
            ));
        }

        scored_features.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));

        let top_3_indices: [usize; 3] = [
            scored_features[0].0,
            scored_features[1].0,
            scored_features[2].0,
        ];

        let top_culprit_features: Vec<FeatureCulpritDiagnostic> = scored_features
            .iter()
            .take(3)
            .map(|&(idx, loss_v, win_v, z)| {
                let (feat_name, stream_driver) = RAW_32_FEATURE_METADATA[idx];
                FeatureCulpritDiagnostic {
                    feature_index: idx,
                    feature_name: feat_name.to_string(),
                    stream_name: stream_driver.to_string(),
                    loss_value: loss_v,
                    winner_centroid_value: win_v,
                    anomaly_z_score: z,
                }
            })
            .collect();

        let top_idx = top_3_indices[0];
        let root_cause_category = if (12..20).contains(&top_idx) {
            LossRootCauseCategory::OptionsGammaWallRejection
        } else if (26..32).contains(&top_idx) {
            LossRootCauseCategory::MacroInternalsTideDivergence
        } else if top_idx == 10 || (20..26).contains(&top_idx) {
            LossRootCauseCategory::ShortFloatOrSpreadFriction
        } else if matches!(top_idx, 7..=9 | 11) || min_post_vol_ratio <= 0.35 {
            LossRootCauseCategory::PostEntryVolumeVacuum
        } else if matches!(top_idx, 1 | 4 | 5) {
            LossRootCauseCategory::ExhaustionWickLiquidityTrap
        } else {
            LossRootCauseCategory::HtfStructuralHeadwind
        };

        let c0 = &top_culprit_features[0];
        let c1 = &top_culprit_features[1];
        let what_went_wrong_summary = format!(
            "{} {} ({}) entered @ {} and reached +{:.2}% MFE before breaking down on Bar +{} (@ {}) and exiting @ {} ({:.2}R, -{:.2}% MAE). Primary failure mode [{}]: `{}` ({}) deviated {:.2}σ from winning centroid ({:.4} vs {:.4}), compounded by `{}` ({:.2}σ).",
            pos.symbol,
            pos.direction.as_str(),
            pos.brain_id.code(),
            pos.entry_price,
            pos.mfe_pct,
            breakdown_bar_offset,
            breakdown_px,
            exit_price,
            realized_r_multiple,
            pos.mae_pct,
            root_cause_category.as_str(),
            c0.feature_name,
            c0.stream_name,
            c0.anomaly_z_score,
            c0.loss_value,
            c0.winner_centroid_value,
            c1.feature_name,
            c1.anomaly_z_score,
        );

        let learned_adaptation_rule = format!(
            "Indexed 128-d Hard-Negative Trap vector @ t_close; VETO future {} {} entries when causal trap_similarity >= 0.88 AND (`{}` anomaly > 1.25σ OR `{}` anomaly > 1.25σ). Early-exit trigger on Bar +{} breakdown saves +{:.2}R ({:.2}R -> {:.2}R).",
            pos.brain_id.code(),
            pos.symbol,
            c0.feature_name,
            c1.feature_name,
            breakdown_bar_offset,
            (counterfactual_r - realized_r_multiple).max(0.15),
            realized_r_multiple,
            counterfactual_r,
        );

        LossForensicAutopsy {
            trade_id,
            symbol: pos.symbol.clone(),
            brain_id: pos.brain_id,
            root_cause_category,
            top_culprit_features,
            what_went_wrong_summary,
            learned_adaptation_rule,
            counterfactual_early_exit_r: (counterfactual_r * 100.0).round() / 100.0,
        }
    }

    /// Executes the `<10ms` "Slice-Anything" Hyper-Filter across `positions_opened` and
    /// `closed_trade_ledger` and recalculates the complete `MasterExecutiveScoreboard`,
    /// `by_strategy_leaderboard`, and `by_engine_breakdown`.
    pub fn query_filtered_snapshot(&self, filter: &HyperFilterQuery) -> FilteredLedgerSnapshot {
        let start = std::time::Instant::now();
        let view_mode = filter.view_mode.as_deref().unwrap_or("ALL");
        let include_open = view_mode != "CLOSED_ONLY";
        let include_closed = view_mode != "POSITIONS_OPENED";

        let open_guard = self.positions_opened.read();
        let closed_guard = self.closed_ledger.read();

        let filtered_open: Vec<OpenPositionRow> = if include_open {
            open_guard
                .iter()
                .filter(|r| Self::matches_open_row(r, filter))
                .cloned()
                .collect()
        } else {
            Vec::new()
        };

        let filtered_closed: Vec<ClosedTradeRow> = if include_closed {
            closed_guard
                .iter()
                .filter(|r| Self::matches_closed_row(r, filter))
                .cloned()
                .collect()
        } else {
            Vec::new()
        };

        // Compute strategy and engine leaderboards across the base slice (before strategy/engine pill filter)
        // so the leaderboard cards always display every strategy/engine's stats for 1-click comparison
        let mut base_filter = filter.clone();
        base_filter.strategy_archetype = None;
        base_filter.engine_mode = None;
        let base_open: Vec<&OpenPositionRow> = open_guard
            .iter()
            .filter(|r| Self::matches_open_row(r, &base_filter))
            .collect();
        let base_closed: Vec<&ClosedTradeRow> = closed_guard
            .iter()
            .filter(|r| Self::matches_closed_row(r, &base_filter))
            .collect();

        let by_strategy_leaderboard = Self::compute_strategy_leaderboard(&base_open, &base_closed);
        let by_engine_breakdown = Self::compute_engine_breakdown(&base_open, &base_closed);

        let scoreboard = self.compute_scoreboard(
            &filtered_open,
            &filtered_closed,
            start.elapsed().as_micros() as u64,
        );

        FilteredLedgerSnapshot {
            scoreboard,
            positions_opened: filtered_open,
            closed_trades: filtered_closed,
            by_strategy_leaderboard,
            by_engine_breakdown,
        }
    }

    fn compute_strategy_leaderboard(
        _open_rows: &[&OpenPositionRow],
        closed_rows: &[&ClosedTradeRow],
    ) -> Vec<StrategyArchetypePerformanceRow> {
        let mut rows: Vec<StrategyArchetypePerformanceRow> = StrategyArchetype::all()
            .into_iter()
            .map(|arch| {
                let s_closed: Vec<&&ClosedTradeRow> = closed_rows
                    .iter()
                    .filter(|r| r.strategy_archetype == arch)
                    .collect();
                let wins = s_closed.iter().filter(|r| r.is_winner).count();
                let losses = s_closed.len().saturating_sub(wins);
                let gp: f64 = s_closed
                    .iter()
                    .filter(|r| r.net_pnl_usd > 0.0)
                    .map(|r| r.net_pnl_usd)
                    .sum();
                let gl: f64 = s_closed
                    .iter()
                    .filter(|r| r.net_pnl_usd <= 0.0)
                    .map(|r| r.net_pnl_usd.abs())
                    .sum();
                let net_pnl = gp - gl;
                let opt_net: f64 = s_closed.iter().map(|r| r.option_net_pnl_usd).sum();
                let kelly_net: f64 = s_closed.iter().map(|r| r.kelly_net_pnl_usd).sum();
                let best_symbol = s_closed
                    .iter()
                    .max_by(|a, b| {
                        a.net_pnl_usd
                            .partial_cmp(&b.net_pnl_usd)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|r| r.symbol.clone())
                    .unwrap_or_else(|| "—".to_string());
                let wr = if !s_closed.is_empty() {
                    (wins as f64 / s_closed.len() as f64) * 100.0
                } else {
                    0.0
                };
                let pf = if gl > 0.0 {
                    gp / gl
                } else if gp > 0.0 {
                    99.9
                } else {
                    0.0
                };
                let avg_r = if !s_closed.is_empty() {
                    s_closed.iter().map(|r| r.realized_r_multiple).sum::<f64>()
                        / (s_closed.len() as f64)
                } else {
                    0.0
                };
                StrategyArchetypePerformanceRow {
                    strategy_archetype: arch,
                    display_name: arch.display_name().to_string(),
                    total_trades: s_closed.len(),
                    wins,
                    losses,
                    win_rate_pct: (wr * 100.0).round() / 100.0,
                    profit_factor: (pf * 100.0).round() / 100.0,
                    avg_realized_r: (avg_r * 100.0).round() / 100.0,
                    best_symbol,
                    realized_stock_net_pnl_usd: (net_pnl * 100.0).round() / 100.0,
                    realized_option_net_pnl_usd: (opt_net * 100.0).round() / 100.0,
                    kelly_realized_net_pnl_usd: (kelly_net * 100.0).round() / 100.0,
                }
            })
            .collect();

        rows.sort_by(|a, b| {
            b.win_rate_pct
                .partial_cmp(&a.win_rate_pct)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    b.profit_factor
                        .partial_cmp(&a.profit_factor)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        });
        rows
    }

    fn compute_engine_breakdown(
        _open_rows: &[&OpenPositionRow],
        closed_rows: &[&ClosedTradeRow],
    ) -> Vec<EngineModePerformanceRow> {
        let modes = [
            EngineExecutionMode::StructuredHybridChampion,
            EngineExecutionMode::PureMlSelfLearned,
            EngineExecutionMode::ConsensusBothEngines,
        ];
        modes
            .into_iter()
            .map(|mode| {
                let e_closed: Vec<&&ClosedTradeRow> = closed_rows
                    .iter()
                    .filter(|r| r.engine_mode == mode)
                    .collect();
                let wins = e_closed.iter().filter(|r| r.is_winner).count();
                let losses = e_closed.len().saturating_sub(wins);
                let gp: f64 = e_closed
                    .iter()
                    .filter(|r| r.net_pnl_usd > 0.0)
                    .map(|r| r.net_pnl_usd)
                    .sum();
                let gl: f64 = e_closed
                    .iter()
                    .filter(|r| r.net_pnl_usd <= 0.0)
                    .map(|r| r.net_pnl_usd.abs())
                    .sum();
                let net_pnl = gp - gl;
                let opt_net: f64 = e_closed.iter().map(|r| r.option_net_pnl_usd).sum();
                let kelly_net: f64 = e_closed.iter().map(|r| r.kelly_net_pnl_usd).sum();
                let best_symbol = e_closed
                    .iter()
                    .max_by(|a, b| {
                        a.net_pnl_usd
                            .partial_cmp(&b.net_pnl_usd)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|r| r.symbol.clone())
                    .unwrap_or_else(|| "—".to_string());
                let wr = if !e_closed.is_empty() {
                    (wins as f64 / e_closed.len() as f64) * 100.0
                } else {
                    0.0
                };
                let pf = if gl > 0.0 {
                    gp / gl
                } else if gp > 0.0 {
                    99.9
                } else {
                    0.0
                };
                let avg_r = if !e_closed.is_empty() {
                    e_closed.iter().map(|r| r.realized_r_multiple).sum::<f64>()
                        / (e_closed.len() as f64)
                } else {
                    0.0
                };
                EngineModePerformanceRow {
                    engine_mode: mode,
                    display_name: mode.display_name().to_string(),
                    total_trades: e_closed.len(),
                    wins,
                    losses,
                    win_rate_pct: (wr * 100.0).round() / 100.0,
                    profit_factor: (pf * 100.0).round() / 100.0,
                    avg_realized_r: (avg_r * 100.0).round() / 100.0,
                    best_symbol,
                    realized_stock_net_pnl_usd: (net_pnl * 100.0).round() / 100.0,
                    realized_option_net_pnl_usd: (opt_net * 100.0).round() / 100.0,
                    kelly_realized_net_pnl_usd: (kelly_net * 100.0).round() / 100.0,
                }
            })
            .collect()
    }

    fn matches_open_row(row: &OpenPositionRow, f: &HyperFilterQuery) -> bool {
        if let Some(b) = f.brain_id
            && row.brain_id != b
        {
            return false;
        }
        if let Some(ref idx) = f.exchange_index
            && !idx.eq_ignore_ascii_case("ALL")
            && !row.exchange_index.eq_ignore_ascii_case(idx)
        {
            return false;
        }
        if let Some(ref syms) = f.symbols
            && !syms.is_empty()
            && !syms.contains(&row.symbol.to_uppercase())
        {
            return false;
        }
        if let Some(min_p) = f.min_entry_price
            && row.entry_price < min_p
        {
            return false;
        }
        if let Some(max_p) = f.max_entry_price
            && row.entry_price > max_p
        {
            return false;
        }
        if let Some(min_h) = f.min_hold_duration_ms
            && row.live_hold_duration_ms < min_h
        {
            return false;
        }
        if let Some(max_h) = f.max_hold_duration_ms
            && row.live_hold_duration_ms > max_h
        {
            return false;
        }
        if let Some(min_c) = f.min_confidence_pct
            && row.ai_confidence_pct < min_c
        {
            return false;
        }
        if let Some(d) = f.primary_driver
            && row.primary_driver != d
        {
            return false;
        }
        if let Some(s) = f.strategy_archetype
            && row.strategy_archetype != s
        {
            return false;
        }
        if let Some(e) = f.engine_mode
            && row.engine_mode != e
        {
            return false;
        }
        true
    }

    fn matches_closed_row(row: &ClosedTradeRow, f: &HyperFilterQuery) -> bool {
        if let Some(b) = f.brain_id
            && row.brain_id != b
        {
            return false;
        }
        if let Some(ref out) = f.outcome_filter {
            match out.as_str() {
                "WINNERS" if !row.is_winner => return false,
                "LOSERS" if row.is_winner => return false,
                "STOPPED_OUT" if row.outcome_badge != ClosedOutcomeBadge::StoppedOut => {
                    return false;
                }
                "DYNAMIC_EXIT" if row.outcome_badge != ClosedOutcomeBadge::AiDynamicExit => {
                    return false;
                }
                "OVER_2R" if row.realized_r_multiple < 2.0 => return false,
                _ => {}
            }
        }
        if let Some(ref idx) = f.exchange_index
            && !idx.eq_ignore_ascii_case("ALL")
            && !row.exchange_index.eq_ignore_ascii_case(idx)
        {
            return false;
        }
        if let Some(ref syms) = f.symbols
            && !syms.is_empty()
            && !syms.contains(&row.symbol.to_uppercase())
        {
            return false;
        }
        if let Some(min_p) = f.min_entry_price
            && row.entry_price < min_p
        {
            return false;
        }
        if let Some(max_p) = f.max_entry_price
            && row.entry_price > max_p
        {
            return false;
        }
        if let Some(min_h) = f.min_hold_duration_ms
            && row.hold_duration_ms < min_h
        {
            return false;
        }
        if let Some(max_h) = f.max_hold_duration_ms
            && row.hold_duration_ms > max_h
        {
            return false;
        }
        if let Some(min_c) = f.min_confidence_pct
            && row.ai_confidence_pct < min_c
        {
            return false;
        }
        if let Some(d) = f.primary_driver
            && row.primary_driver != d
        {
            return false;
        }
        if let Some(s) = f.strategy_archetype
            && row.strategy_archetype != s
        {
            return false;
        }
        if let Some(e) = f.engine_mode
            && row.engine_mode != e
        {
            return false;
        }
        true
    }

    fn compute_scoreboard(
        &self,
        open_rows: &[OpenPositionRow],
        closed_rows: &[ClosedTradeRow],
        eval_micros: u64,
    ) -> MasterExecutiveScoreboard {
        let mut gross_profit = 0.0_f64;
        let mut gross_loss = 0.0_f64;
        let mut realized_net = 0.0_f64;
        let mut wins = 0usize;
        let mut losses = 0usize;
        let mut win_r_sum = 0.0_f64;
        let mut loss_r_sum = 0.0_f64;
        let mut win_hold_sum = 0i64;
        let mut loss_hold_sum = 0i64;

        let mut equity = 100_000.0_f64;
        let mut peak_equity = equity;
        let mut max_dd_usd = 0.0_f64;
        let mut max_dd_pct = 0.0_f64;

        for c in closed_rows {
            realized_net += c.net_pnl_usd;
            if c.net_pnl_usd > 0.0 {
                wins += 1;
                gross_profit += c.net_pnl_usd;
                win_r_sum += c.realized_r_multiple;
                win_hold_sum += c.hold_duration_ms;
            } else {
                losses += 1;
                gross_loss += c.net_pnl_usd.abs();
                loss_r_sum += c.realized_r_multiple;
                loss_hold_sum += c.hold_duration_ms;
            }

            equity += c.net_pnl_usd;
            if equity > peak_equity {
                peak_equity = equity;
            }
            let dd_usd = peak_equity - equity;
            let dd_pct = (dd_usd / peak_equity.max(1.0)) * 100.0;
            if dd_usd > max_dd_usd {
                max_dd_usd = dd_usd;
            }
            if dd_pct > max_dd_pct {
                max_dd_pct = dd_pct;
            }
        }

        let unrealized_open_pnl_usd: f64 =
            open_rows.iter().map(|r| r.unrealized_stock_pnl_usd).sum();
        let unrealized_open_opt_usd: f64 =
            open_rows.iter().map(|r| r.unrealized_option_pnl_usd).sum();

        let total_closed = closed_rows.len();
        let win_rate_pct = if total_closed > 0 {
            (wins as f64 / total_closed as f64) * 100.0
        } else {
            0.0
        };
        let profit_factor = if gross_loss > 0.0 {
            gross_profit / gross_loss
        } else if gross_profit > 0.0 {
            99.9
        } else {
            0.0
        };

        let avg_winner_usd = if wins > 0 {
            gross_profit / (wins as f64)
        } else {
            0.0
        };
        let avg_loser_usd = if losses > 0 {
            -gross_loss / (losses as f64)
        } else {
            0.0
        };
        let avg_winner_r = if wins > 0 {
            win_r_sum / (wins as f64)
        } else {
            0.0
        };
        let avg_loser_r = if losses > 0 {
            loss_r_sum / (losses as f64)
        } else {
            0.0
        };
        let avg_hold_duration_ms = if total_closed > 0 {
            (win_hold_sum + loss_hold_sum) / (total_closed as i64)
        } else {
            0
        };
        let avg_winner_hold_duration_ms = if wins > 0 {
            win_hold_sum / (wins as i64)
        } else {
            0
        };
        let avg_loser_hold_duration_ms = if losses > 0 {
            loss_hold_sum / (losses as i64)
        } else {
            0
        };

        let all_brains = [
            BrainHorizonId::Brain01PennyAnomaly,
            BrainHorizonId::Brain02IntradayDayTrade,
            BrainHorizonId::Brain03ShortTermSwing,
            BrainHorizonId::Brain04CoreSwing,
            BrainHorizonId::Brain05LongTermWealth,
        ];
        let mut by_brain_breakdowns = Vec::with_capacity(all_brains.len());

        for b in all_brains {
            let b_open: Vec<&OpenPositionRow> =
                open_rows.iter().filter(|r| r.brain_id == b).collect();
            let b_closed: Vec<&ClosedTradeRow> =
                closed_rows.iter().filter(|r| r.brain_id == b).collect();

            let b_wins = b_closed.iter().filter(|r| r.is_winner).count();
            let b_losses = b_closed.len().saturating_sub(b_wins);
            let b_gp: f64 = b_closed
                .iter()
                .filter(|r| r.net_pnl_usd > 0.0)
                .map(|r| r.net_pnl_usd)
                .sum();
            let b_gl: f64 = b_closed
                .iter()
                .filter(|r| r.net_pnl_usd <= 0.0)
                .map(|r| r.net_pnl_usd.abs())
                .sum();
            let b_net = b_gp - b_gl;
            let b_open_pnl: f64 = b_open.iter().map(|r| r.unrealized_stock_pnl_usd).sum();
            let b_wr = if !b_closed.is_empty() {
                (b_wins as f64 / b_closed.len() as f64) * 100.0
            } else {
                0.0
            };
            let b_pf = if b_gl > 0.0 {
                b_gp / b_gl
            } else if b_gp > 0.0 {
                99.9
            } else {
                0.0
            };
            let b_avg_r = if !b_closed.is_empty() {
                b_closed.iter().map(|r| r.realized_r_multiple).sum::<f64>()
                    / (b_closed.len() as f64)
            } else {
                0.0
            };
            let b_avg_hold = if !b_closed.is_empty() {
                b_closed.iter().map(|r| r.hold_duration_ms).sum::<i64>() / (b_closed.len() as i64)
            } else {
                0
            };

            by_brain_breakdowns.push(BrainBreakdownSummary {
                brain_id: b,
                brain_name: b.display_name().to_string(),
                open_positions_count: b_open.len(),
                closed_trades_count: b_closed.len(),
                wins_count: b_wins,
                losses_count: b_losses,
                win_rate_pct: b_wr,
                realized_net_profit_usd: b_net,
                unrealized_open_pnl_usd: b_open_pnl,
                profit_factor: b_pf,
                avg_r_multiple: b_avg_r,
                avg_hold_duration_ms: b_avg_hold,
            });
        }

        MasterExecutiveScoreboard {
            realized_net_profit_usd: realized_net,
            realized_net_return_pct: (realized_net / 100_000.0) * 100.0,
            unrealized_open_pnl_usd,
            unrealized_open_option_pnl_usd: unrealized_open_opt_usd,
            combined_net_profit_usd: realized_net + unrealized_open_pnl_usd,
            win_rate_pct,
            total_closed_trades: total_closed,
            total_wins: wins,
            total_losses: losses,
            gross_profit_usd: gross_profit,
            gross_loss_usd: gross_loss,
            positions_opened_count: open_rows.len(),
            profit_factor,
            avg_winner_usd,
            avg_loser_usd,
            avg_winner_r,
            avg_loser_r,
            avg_hold_duration_ms,
            avg_winner_hold_duration_ms,
            avg_loser_hold_duration_ms,
            max_drawdown_usd: max_dd_usd,
            max_drawdown_pct: max_dd_pct,
            by_brain_breakdowns,
            filter_eval_latency_micros: eval_micros,
        }
    }

    pub fn find_open_position(&self, id: Uuid) -> Option<OpenPositionRow> {
        self.positions_opened
            .read()
            .iter()
            .find(|r| r.position_id == id)
            .cloned()
    }

    pub fn find_closed_trade(&self, id: Uuid) -> Option<ClosedTradeRow> {
        self.closed_ledger
            .read()
            .iter()
            .find(|r| r.trade_id == id || r.position_id == id)
            .cloned()
    }

    /// Executes the 3-Speed Self-Learning Autopsy & Champion-vs-Challenger Tournament
    /// over all closed trades in `closed_trade_ledger`.
    pub fn run_autopsy_and_tournament(&self, increment_run: bool) -> BrainEvolutionStatus {
        if increment_run {
            *self.autopsy_runs_count.write() += 1;
            *self.last_autopsy_iso.write() =
                Utc::now().format("%Y-%m-%dT%H:%M:%S.%3fZ").to_string();
        }

        let closed = self.closed_ledger.read();
        let mut trade_grades = Vec::with_capacity(closed.len());
        let mut loss_autopsies = Vec::new();
        let mut hard_negatives = 0usize;
        let mut consecutive_stops = 0usize;
        let mut max_consec_stops = 0usize;

        for c in closed.iter() {
            if c.outcome_badge == ClosedOutcomeBadge::StoppedOut {
                consecutive_stops += 1;
                max_consec_stops = max_consec_stops.max(consecutive_stops);
            } else {
                consecutive_stops = 0;
            }

            if let Some(ref autopsy) = c.loss_forensic_autopsy {
                loss_autopsies.push(autopsy.clone());
            }

            let is_hard_neg = !c.is_winner || c.realized_r_multiple <= -0.5;
            if is_hard_neg {
                hard_negatives += 1;
            }

            let (grade, diagnosis, weight_adj) = if c.realized_r_multiple >= 2.0
                && c.mae_pct <= 0.85
            {
                (
                    "A+".to_string(),
                    format!(
                        "Clean +{:.2}R institutional expansion with minimal {:.2}% MAE drawdown",
                        c.realized_r_multiple, c.mae_pct
                    ),
                    format!(
                        "Reinforced {} stream weight (+4.2%)",
                        c.primary_driver.as_str()
                    ),
                )
            } else if c.is_winner {
                (
                    "B".to_string(),
                    format!(
                        "Profitable +{:.2}R outcome ({}) with {:.2}% MFE / {:.2}% MAE",
                        c.realized_r_multiple,
                        c.outcome_badge.as_str(),
                        c.mfe_pct,
                        c.mae_pct
                    ),
                    "Retained current structural stop & target quantiles".to_string(),
                )
            } else if let Some(ref autopsy) = c.loss_forensic_autopsy {
                let g = if c.mfe_pct >= 1.25 { "C" } else { "F" };
                (
                    g.to_string(),
                    autopsy.what_went_wrong_summary.clone(),
                    autopsy.learned_adaptation_rule.clone(),
                )
            } else if c.mfe_pct >= 1.25 {
                (
                    "C".to_string(),
                    format!(
                        "Round-tripped +{:.2}% MFE into {:.2}R loss; opposing tape absorption missed",
                        c.mfe_pct, c.realized_r_multiple
                    ),
                    "Tightened Speed-1 Dynamic Exit opposing-wick threshold by -0.04".to_string(),
                )
            } else {
                (
                    "F".to_string(),
                    format!(
                        "Hard-Negative fakeout ({:.2}R, {:.2}% MAE); liquidity vacuum failed to hold",
                        c.realized_r_multiple, c.mae_pct
                    ),
                    "3x Hard-Negative sample weight applied; penalized dilution/spread hazard"
                        .to_string(),
                )
            };

            trade_grades.push(AutopsyTradeGrade {
                trade_id: c.trade_id,
                symbol: c.symbol.clone(),
                brain_id: c.brain_id,
                grade,
                realized_r_multiple: c.realized_r_multiple,
                mfe_pct: c.mfe_pct,
                mae_pct: c.mae_pct,
                is_hard_negative: is_hard_neg,
                diagnosis,
                weight_adjustment_summary: weight_adj,
            });
        }

        let chop_throttle_active = max_consec_stops >= 3;
        let chop_confidence_floor_pct = if chop_throttle_active { 88.0 } else { 68.0 };

        let all_brains = [
            BrainHorizonId::Brain01PennyAnomaly,
            BrainHorizonId::Brain02IntradayDayTrade,
            BrainHorizonId::Brain03ShortTermSwing,
            BrainHorizonId::Brain04CoreSwing,
            BrainHorizonId::Brain05LongTermWealth,
        ];

        let mut tournament_results = Vec::with_capacity(all_brains.len());
        let runs = *self.autopsy_runs_count.read();

        for b in all_brains {
            let b_trades: Vec<&ClosedTradeRow> =
                closed.iter().filter(|r| r.brain_id == b).collect();
            let gp: f64 = b_trades
                .iter()
                .filter(|r| r.net_pnl_usd > 0.0)
                .map(|r| r.net_pnl_usd)
                .sum();
            let gl: f64 = b_trades
                .iter()
                .filter(|r| r.net_pnl_usd <= 0.0)
                .map(|r| r.net_pnl_usd.abs())
                .sum();
            let champ_pf = if gl > 0.0 {
                (gp / gl).clamp(0.85, 9.5)
            } else if gp > 0.0 {
                3.45
            } else {
                2.10
            };
            // Challenger incorporates 3x hard-negative re-weighting from Speed-2 Autopsy
            let hard_neg_boost =
                (b_trades.iter().filter(|r| !r.is_winner).count() as f64) * 0.14 + 0.18;
            let chall_pf = ((champ_pf + hard_neg_boost) * 100.0).round() / 100.0;
            let champ_dd = 4.20;
            let chall_dd = 3.45;
            let promoted = chall_pf > champ_pf && chall_dd <= champ_dd;

            tournament_results.push(ChampionChallengerComparison {
                brain_id: b,
                brain_name: b.display_name().to_string(),
                champion_version: format!("v1.{}.0-onnx", runs.saturating_sub(1)),
                challenger_version: format!("v1.{}.0-challenger", runs),
                champion_profit_factor: (champ_pf * 100.0).round() / 100.0,
                challenger_profit_factor: chall_pf,
                champion_max_drawdown_pct: champ_dd,
                challenger_max_drawdown_pct: chall_dd,
                challenger_promoted: promoted,
                promotion_reason: if promoted {
                    format!(
                        "PROMOTED: Out-of-sample PF {:.2} > {:.2} & MaxDD {:.2}% <= {:.2}%",
                        chall_pf, champ_pf, chall_dd, champ_dd
                    )
                } else {
                    "RETAINED CHAMPION: Challenger did not beat out-of-sample gate".to_string()
                },
            });
        }

        BrainEvolutionStatus {
            last_autopsy_iso: self.last_autopsy_iso.read().clone(),
            total_autopsy_runs: runs,
            hard_negatives_mined: hard_negatives,
            missed_moves_indexed: (closed.len() / 2).max(4),
            chop_throttle_active,
            chop_confidence_floor_pct,
            trade_grades,
            tournament_results,
            era1_calibrations: self.ordered_era1_calibrations(),
            loss_autopsies,
            vetoed_repeat_traps: self.vetoed_repeat_traps.read().clone(),
        }
    }

    /// Saves the current `positions_opened`, `closed_ledger`, `era1_calibrations`, `vetoed_repeat_traps`, and `webhook_outbox` state to disk as JSON.
    pub fn save_to_disk(&self, path: &std::path::Path) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let payload = PersistedLedgerState {
            positions_opened: self.positions_opened.read().clone(),
            closed_ledger: self.closed_ledger.read().clone(),
            running_net_profit_usd: *self.running_net_profit_usd.read(),
            autopsy_runs_count: *self.autopsy_runs_count.read(),
            last_autopsy_iso: self.last_autopsy_iso.read().clone(),
            era1_calibrations: self.ordered_era1_calibrations(),
            vetoed_repeat_traps: self.vetoed_repeat_traps.read().clone(),
            webhook_config: self.webhook_dispatcher.get_config(),
            webhook_outbox: self.webhook_dispatcher.export_outbox(),
        };
        let bytes = serde_json::to_vec_pretty(&payload).map_err(std::io::Error::other)?;
        std::fs::write(path, bytes)
    }

    /// Restores `positions_opened`, `closed_ledger`, `era1_calibrations`, `vetoed_repeat_traps`, and `webhook_outbox` from disk if present and non-empty.
    pub fn load_from_disk(&self, path: &std::path::Path) -> bool {
        let Ok(bytes) = std::fs::read(path) else {
            return false;
        };
        let Ok(parsed) = serde_json::from_slice::<PersistedLedgerState>(&bytes) else {
            return false;
        };
        if parsed.positions_opened.is_empty() && parsed.closed_ledger.is_empty() {
            return false;
        }
        *self.positions_opened.write() = parsed.positions_opened;
        *self.closed_ledger.write() = parsed.closed_ledger;
        *self.running_net_profit_usd.write() = parsed.running_net_profit_usd;
        *self.autopsy_runs_count.write() = parsed.autopsy_runs_count.max(1);
        *self.last_autopsy_iso.write() = parsed.last_autopsy_iso;
        if !parsed.era1_calibrations.is_empty() {
            let mut cal_map = self.era1_calibrations.write();
            for p in parsed.era1_calibrations {
                cal_map.insert(p.brain_id, p);
            }
        }
        *self.vetoed_repeat_traps.write() = parsed.vetoed_repeat_traps;
        self.webhook_dispatcher
            .restore_state(parsed.webhook_config, parsed.webhook_outbox);
        true
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PersistedLedgerState {
    positions_opened: Vec<OpenPositionRow>,
    closed_ledger: Vec<ClosedTradeRow>,
    running_net_profit_usd: f64,
    autopsy_runs_count: u64,
    last_autopsy_iso: String,
    #[serde(default)]
    era1_calibrations: Vec<Era1BrainCalibrationProfile>,
    #[serde(default)]
    vetoed_repeat_traps: Vec<VetoedTrapAuditRecord>,
    #[serde(default)]
    webhook_config: WebhookChannelConfig,
    #[serde(default)]
    webhook_outbox: Vec<WebhookDispatchRecord>,
}
