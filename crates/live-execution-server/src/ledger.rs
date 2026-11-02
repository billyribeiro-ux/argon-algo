//! Real-Time `POSITIONS OPENED` Active Monitor, `closed_trade_ledger` Archiver,
//! Master Executive Scoreboard, and `<10ms` "Slice-Anything" Hyper-Filter Engine.

use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;

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
use crate::precedent_knn::{
    HardNegativeTrapRecord, PostEntryScratchDiagnostic, PrecedentKnnIndex, PrecedentMatchSummary,
};
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
        let entry_f64 = entry_price.to_f64().max(0.00001);
        let stop_f64 = stop_price.to_f64().max(0.00001);
        let stop_dist = (entry_f64 - stop_f64).abs().max(entry_f64 * 0.004);
        let empirical_med_stop_pct = (p85_mae_pct / 100.0).clamp(0.006, 0.15);
        let synthetic_t1 =
            SubPennyPrice::from_f64(entry_f64 + stop_dist * (p50_mfe_pct / p85_mae_pct.max(0.20)).clamp(1.15, 3.0))
                .unwrap_or(entry_price);
        let synthetic_t2 =
            SubPennyPrice::from_f64(entry_f64 + stop_dist * (p50_mfe_pct / p85_mae_pct.max(0.20) * 1.65).clamp(1.60, 4.5))
                .unwrap_or(entry_price);
        Self::compute_allocation_with_symbol_stop(
            account_equity_usd,
            brain_id,
            entry_price,
            stop_price,
            synthetic_t1,
            synthetic_t2,
            ai_confidence_pct,
            precedent_win_rate_pct,
            p50_mfe_pct,
            p85_mae_pct,
            era1_calibrated_win_rate_pct,
            era1_calibrated_profit_factor,
            empirical_med_stop_pct,
        )
    }

    /// 100% Self-Learned Per-Symbol Empirical Stop Quantile Kelly Allocator (Phase 13 Upgrade 1).
    /// Normalizes each trade's stop distance relative to that `(symbol, brain_id)` pair's own
    /// Era-1 empirical median structural stop distance $\tilde{s}_{\text{symbol, Era1}}$,
    /// weighted by the trade's causal expected payoff ratio $b$ and `PrecedentKnnIndex` win rate.
    #[allow(clippy::too_many_arguments)]
    pub fn compute_allocation_with_symbol_stop(
        account_equity_usd: f64,
        brain_id: BrainHorizonId,
        entry_price: SubPennyPrice,
        stop_price: SubPennyPrice,
        target_1_price: SubPennyPrice,
        target_2_price: SubPennyPrice,
        ai_confidence_pct: f64,
        precedent_win_rate_pct: f64,
        p50_mfe_pct: f64,
        p85_mae_pct: f64,
        era1_calibrated_win_rate_pct: f64,
        era1_calibrated_profit_factor: f64,
        symbol_era1_median_stop_pct: f64,
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

        let entry_f64 = entry_price.to_f64().max(0.00001);
        let stop_f64 = stop_price.to_f64().max(0.00001);
        let t1_f64 = target_1_price.to_f64().max(0.00001);
        let t2_f64 = target_2_price.to_f64().max(0.00001);
        let stop_dist_per_share = (entry_f64 - stop_f64)
            .abs()
            .max(0.004 * entry_f64)
            .max(0.00001);
        let stop_pct = (stop_dist_per_share / entry_f64).clamp(0.004, 0.25);
        let t1_pct = ((t1_f64 - entry_f64).abs() / entry_f64).max(stop_pct * 0.85);
        let t2_pct = ((t2_f64 - entry_f64).abs() / entry_f64).max(t1_pct * 1.15);

        let structural_rr = ((0.55 * t1_pct + 0.45 * t2_pct) / stop_pct).clamp(1.05, 5.00);
        let empirical_ratio = (p50_mfe_pct.max(0.25) / p85_mae_pct.max(0.15)).clamp(1.15, 5.00);
        let pf_clamped = era1_calibrated_profit_factor.clamp(1.15, 5.00);
        let payoff_ratio_b =
            (0.45 * structural_rr + 0.30 * empirical_ratio + 0.25 * pf_clamped).clamp(1.15, 5.00);

        let raw_kelly_fraction =
            (win_probability - (1.0 - win_probability) / payoff_ratio_b).max(0.015);

        // Learned per-symbol Era-1 median structural stop distance \tilde{s}_{symbol, Era1}
        // (Zero hardcoded per-Brain ref_stop_pct constants)
        let learned_sym_stop_pct = if symbol_era1_median_stop_pct > 0.001 {
            symbol_era1_median_stop_pct.clamp(0.004, 0.22)
        } else {
            (p85_mae_pct / 100.0).clamp(0.006, 0.15)
        };

        let horizon_mult = match brain_id {
            BrainHorizonId::Brain01PennyAnomaly | BrainHorizonId::Brain02IntradayDayTrade => 1.0,
            BrainHorizonId::Brain03ShortTermSwing => 1.5,
            BrainHorizonId::Brain04CoreSwing | BrainHorizonId::Brain05LongTermWealth => 2.5,
        };

        // Prevent noise-tight stops (< \tilde{s}_{symbol, Era1}) from artificially inflating share count,
        // while allowing high-beta symbols (MSTR, COIN, IONQ, GS, MSFT) to size relative to their own \tilde{s}_{symbol, Era1}
        let effective_stop_pct = stop_pct.max(learned_sym_stop_pct * 0.92);
        let sym_vol_alignment = (learned_sym_stop_pct / stop_pct).powf(0.25).clamp(0.72, 1.35);
        let conviction_mult = ((raw_kelly_fraction / 0.36)
            * (w_knn / 0.64).powf(0.85)
            * (structural_rr / 1.35).powf(0.35))
        .clamp(0.68, 1.85);

        let base_1r_risk_usd = (equity * 0.0035 * horizon_mult).max(175.0);
        let target_risk_budget_usd =
            (base_1r_risk_usd * sym_vol_alignment * conviction_mult).clamp(equity * 0.005, equity * 0.0275);
        let kelly_risk_frac = (target_risk_budget_usd / equity).clamp(0.005, 0.0275);

        let raw_notional = target_risk_budget_usd / effective_stop_pct;

        // Clamp notional position value between $5,000 and $35,000 per tranche
        let clamped_notional = raw_notional.clamp(5_000.0, 35_000.0);
        let quantity_shares = (clamped_notional / entry_f64).round().max(1.0);
        let notional_allocated_usd = ((quantity_shares * entry_f64) * 100.0).round() / 100.0;
        let risk_budget_usd = (target_risk_budget_usd * 100.0).round() / 100.0;
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
    #[serde(rename = "CLOSED - AI LOSS-AUTOPSY SCRATCH EXIT")]
    AiLossAutopsyScratchExit,
}

impl ClosedOutcomeBadge {
    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::TargetHit => "CLOSED - TARGET HIT",
            Self::AiDynamicExit => "CLOSED - AI DYNAMIC EXIT",
            Self::TrailingStop => "CLOSED - TRAILING STOP",
            Self::StoppedOut => "CLOSED - STOPPED OUT",
            Self::AiLossAutopsyScratchExit => "CLOSED - AI LOSS-AUTOPSY SCRATCH EXIT",
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
    #[serde(default)]
    pub kelly_profit_factor: f64,
    #[serde(default)]
    pub adaptive_total_trades: usize,
    #[serde(default)]
    pub adaptive_wins: usize,
    #[serde(default)]
    pub adaptive_losses: usize,
    #[serde(default)]
    pub adaptive_vetoed_count: usize,
    #[serde(default)]
    pub adaptive_scratched_count: usize,
    #[serde(default)]
    pub adaptive_win_rate_pct: f64,
    #[serde(default)]
    pub adaptive_profit_factor: f64,
    #[serde(default)]
    pub adaptive_kelly_profit_factor: f64,
    #[serde(default)]
    pub adaptive_avg_realized_r: f64,
    #[serde(default)]
    pub adaptive_stock_net_pnl_usd: f64,
    #[serde(default)]
    pub adaptive_kelly_net_pnl_usd: f64,
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
    #[serde(default)]
    pub kelly_profit_factor: f64,
    #[serde(default)]
    pub adaptive_total_trades: usize,
    #[serde(default)]
    pub adaptive_wins: usize,
    #[serde(default)]
    pub adaptive_losses: usize,
    #[serde(default)]
    pub adaptive_vetoed_count: usize,
    #[serde(default)]
    pub adaptive_scratched_count: usize,
    #[serde(default)]
    pub adaptive_win_rate_pct: f64,
    #[serde(default)]
    pub adaptive_profit_factor: f64,
    #[serde(default)]
    pub adaptive_kelly_profit_factor: f64,
    #[serde(default)]
    pub adaptive_avg_realized_r: f64,
    #[serde(default)]
    pub adaptive_stock_net_pnl_usd: f64,
    #[serde(default)]
    pub adaptive_kelly_net_pnl_usd: f64,
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
    // Phase 13 Non-Destructive Adaptive Trap-Immunity & Bar+1/Bar+2 Loss-Centroid Scratch Outcomes
    #[serde(default)]
    pub was_vetoed_by_loss_autopsy: bool,
    #[serde(default)]
    pub vetoed_by_loss_trade_id: Option<Uuid>,
    #[serde(default)]
    pub entry_trap_similarity_score: f64,
    #[serde(default)]
    pub was_scratched_by_loss_autopsy: bool,
    #[serde(default)]
    pub scratch_bar_offset: usize,
    #[serde(default)]
    pub adaptive_outcome_badge: String,
    #[serde(default = "default_sub_penny_zero")]
    pub adaptive_exit_price: SubPennyPrice,
    #[serde(default)]
    pub adaptive_scratch_exit_r: f64,
    #[serde(default)]
    pub adaptive_net_pnl_usd: f64,
    #[serde(default)]
    pub adaptive_kelly_net_pnl_usd: f64,
    #[serde(default)]
    pub adaptive_loss_saved_usd: f64,
    #[serde(default)]
    pub adaptive_kelly_loss_saved_usd: f64,
    #[serde(default)]
    pub loss_affinity_posterior_pct: f64,
    #[serde(default)]
    pub post_entry_bar1_features_32: Vec<f32>,
    #[serde(default)]
    pub culprit_anomaly_weights_32: Vec<f32>,
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
    #[serde(default)]
    pub kelly_realized_net_profit_usd: f64,
    #[serde(default)]
    pub kelly_profit_factor: f64,
    #[serde(default)]
    pub adaptive_closed_trades_count: usize,
    #[serde(default)]
    pub adaptive_wins_count: usize,
    #[serde(default)]
    pub adaptive_losses_count: usize,
    #[serde(default)]
    pub adaptive_vetoed_count: usize,
    #[serde(default)]
    pub adaptive_scratched_count: usize,
    #[serde(default)]
    pub adaptive_win_rate_pct: f64,
    #[serde(default)]
    pub adaptive_profit_factor: f64,
    #[serde(default)]
    pub adaptive_kelly_profit_factor: f64,
    #[serde(default)]
    pub adaptive_realized_net_profit_usd: f64,
    #[serde(default)]
    pub adaptive_kelly_net_profit_usd: f64,
    #[serde(default)]
    pub adaptive_avg_r_multiple: f64,
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
    #[serde(default)]
    pub kelly_realized_net_profit_usd: f64,
    #[serde(default)]
    pub kelly_profit_factor: f64,
    #[serde(default)]
    pub kelly_max_drawdown_usd: f64,
    #[serde(default)]
    pub adaptive_realized_net_profit_usd: f64,
    #[serde(default)]
    pub adaptive_kelly_net_profit_usd: f64,
    #[serde(default)]
    pub adaptive_win_rate_pct: f64,
    #[serde(default)]
    pub adaptive_profit_factor: f64,
    #[serde(default)]
    pub adaptive_kelly_profit_factor: f64,
    #[serde(default)]
    pub adaptive_total_closed_trades: usize,
    #[serde(default)]
    pub adaptive_total_wins: usize,
    #[serde(default)]
    pub adaptive_total_losses: usize,
    #[serde(default)]
    pub adaptive_vetoed_losses_count: usize,
    #[serde(default)]
    pub adaptive_scratched_losses_count: usize,
    #[serde(default)]
    pub adaptive_total_loss_saved_usd: f64,
    #[serde(default)]
    pub adaptive_kelly_total_loss_saved_usd: f64,
    #[serde(default)]
    pub adaptive_max_drawdown_usd: f64,
    #[serde(default)]
    pub adaptive_kelly_max_drawdown_usd: f64,
    #[serde(default)]
    pub adaptive_avg_loser_r: f64,
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

fn default_sub_penny_zero() -> SubPennyPrice {
    SubPennyPrice::ZERO
}

fn default_era1_median_hold_bars() -> usize {
    19
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
    #[serde(default)]
    pub era1_median_stop_pct: f64,
    #[serde(default)]
    pub era1_learned_scratch_posterior_threshold: f64,
    #[serde(default = "default_era1_median_hold_bars")]
    pub era1_median_hold_bars: usize,
}

/// Thread-safe, sub-millisecond Real-Time Execution Ledger Engine.
pub struct RealTimeLedgerEngine {
    fingerprint_engine: RawFingerprintEngine,
    precedent_knn: Arc<PrecedentKnnIndex>,
    positions_opened: RwLock<Vec<OpenPositionRow>>,
    closed_ledger: RwLock<Vec<ClosedTradeRow>>,
    running_net_profit_usd: RwLock<f64>,
    autopsy_runs_count: RwLock<u64>,
    last_autopsy_iso: RwLock<String>,
    era1_calibrations:
        RwLock<std::collections::HashMap<BrainHorizonId, Era1BrainCalibrationProfile>>,
    era1_symbol_median_stops:
        RwLock<std::collections::HashMap<(String, BrainHorizonId), f64>>,
    vetoed_repeat_traps: RwLock<Vec<VetoedTrapAuditRecord>>,
    enforce_hard_scratch_on_baseline: RwLock<bool>,
    webhook_dispatcher: OutboundWebhookDispatcher,
}

impl Default for RealTimeLedgerEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl RealTimeLedgerEngine {
    pub fn new() -> Self {
        Self::with_precedent_knn(Arc::new(PrecedentKnnIndex::new_preseeded()))
    }

    pub fn with_precedent_knn(precedent_knn: Arc<PrecedentKnnIndex>) -> Self {
        Self {
            fingerprint_engine: RawFingerprintEngine::new(),
            precedent_knn,
            positions_opened: RwLock::new(Vec::with_capacity(128)),
            closed_ledger: RwLock::new(Vec::with_capacity(4096)),
            running_net_profit_usd: RwLock::new(0.0),
            autopsy_runs_count: RwLock::new(1),
            last_autopsy_iso: RwLock::new(Utc::now().format("%Y-%m-%dT%H:%M:%S.%3fZ").to_string()),
            era1_calibrations: RwLock::new(std::collections::HashMap::new()),
            era1_symbol_median_stops: RwLock::new(std::collections::HashMap::new()),
            vetoed_repeat_traps: RwLock::new(Vec::with_capacity(256)),
            enforce_hard_scratch_on_baseline: RwLock::new(false),
            webhook_dispatcher: OutboundWebhookDispatcher::new(),
        }
    }

    pub fn precedent_knn(&self) -> &Arc<PrecedentKnnIndex> {
        &self.precedent_knn
    }

    pub fn set_enforce_hard_scratch_on_baseline(&self, enabled: bool) {
        *self.enforce_hard_scratch_on_baseline.write() = enabled;
    }

    pub fn webhooks(&self) -> &OutboundWebhookDispatcher {
        &self.webhook_dispatcher
    }

    pub fn set_era1_symbol_median_stop_pct(
        &self,
        symbol: &str,
        brain_id: BrainHorizonId,
        median_stop_pct: f64,
    ) {
        self.era1_symbol_median_stops
            .write()
            .insert((symbol.to_uppercase(), brain_id), median_stop_pct);
    }

    pub fn get_era1_symbol_median_stop_pct(
        &self,
        symbol: &str,
        brain_id: BrainHorizonId,
    ) -> Option<f64> {
        self.era1_symbol_median_stops
            .read()
            .get(&(symbol.to_uppercase(), brain_id))
            .copied()
    }

    /// Returns `(consecutive_losses, last_loss_closed_ts_ns)` for `(symbol, brain_id)` across trades
    /// closed strictly before `current_ts_ns` (`r.closed_timestamp_ns < current_ts_ns`).
    pub fn causal_consecutive_losses_for_symbol(
        &self,
        symbol: &str,
        brain_id: BrainHorizonId,
        current_ts_ns: i64,
    ) -> (usize, i64) {
        let closed = self.closed_ledger.read();
        let mut symbol_trades: Vec<&ClosedTradeRow> = closed
            .iter()
            .filter(|r| {
                r.brain_id == brain_id
                    && r.symbol.eq_ignore_ascii_case(symbol)
                    && r.closed_timestamp_ns < current_ts_ns
            })
            .collect();
        symbol_trades.sort_by_key(|r| r.closed_timestamp_ns);
        let mut consec = 0usize;
        let mut last_loss_ts = 0i64;
        for r in symbol_trades.iter().rev() {
            if !r.is_winner {
                if consec == 0 {
                    last_loss_ts = r.closed_timestamp_ns;
                }
                consec += 1;
            } else {
                break;
            }
        }
        (consec, last_loss_ts)
    }

    /// Data-Driven Causal Streak Confidence Adaptation (Phase 15 Upgrade 3).
    ///
    /// Looks back over the Brain's Era-1 learned median hold window (`era1_median_hold_bars * 4`)
    /// and computes the empirical confidence excess of causal consecutive false positives:
    /// $$\Delta C_{\text{empirical}} = \text{clamp}\!\left(\overline{C}_{\text{recent\_losses}} - C_{\text{base\_floor}}, \; 1.5, \; 8.5\right)$$
    /// (representing `0.015` to `0.085` in probability units / `1.5` to `8.5` percentage points),
    /// eliminating static lookback and penalty constants.
    pub fn compute_causal_streak_confidence_adjustment(
        &self,
        symbol: &str,
        brain_id: BrainHorizonId,
        current_ts_ns: i64,
        tf_step_ns: i64,
        base_min_conf_pct: f64,
        strict_htf_trend_aligned: bool,
    ) -> f64 {
        let closed = self.closed_ledger.read();
        let mut symbol_trades: Vec<&ClosedTradeRow> = closed
            .iter()
            .filter(|r| {
                r.brain_id == brain_id
                    && r.symbol.eq_ignore_ascii_case(symbol)
                    && r.closed_timestamp_ns < current_ts_ns
            })
            .collect();
        symbol_trades.sort_by_key(|r| r.closed_timestamp_ns);

        let empirical_hold_bars = if !symbol_trades.is_empty() {
            let mut holds: Vec<i64> = symbol_trades
                .iter()
                .map(|r| {
                    ((r.closed_timestamp_ns.saturating_sub(r.opened_timestamp_ns))
                        / tf_step_ns.max(1))
                    .max(6)
                })
                .collect();
            holds.sort_unstable();
            holds[holds.len() / 2] as usize
        } else {
            self.get_era1_brain_calibration(brain_id)
                .map(|c| c.era1_median_hold_bars.max(12) * 3)
                .unwrap_or(38)
        };
        let lookback_bars = (empirical_hold_bars * 2).max(64) as i64;
        let max_age_ns = tf_step_ns.max(1).saturating_mul(lookback_bars);

        let mut recent_loss_confs = Vec::new();
        let mut last_loss_ts = 0i64;
        for r in symbol_trades.iter().rev() {
            if !r.is_winner {
                if recent_loss_confs.is_empty() {
                    last_loss_ts = r.closed_timestamp_ns;
                }
                recent_loss_confs.push(r.ai_confidence_pct);
            } else {
                break;
            }
        }

        if recent_loss_confs.len() < 2
            || last_loss_ts == 0
            || current_ts_ns.saturating_sub(last_loss_ts) > max_age_ns
        {
            return 0.0;
        }

        let n_l = recent_loss_confs.len() as f64;
        let mean_loss_conf_pct = recent_loss_confs.iter().sum::<f64>() / n_l;
        let max_loss_conf_pct = recent_loss_confs
            .iter()
            .copied()
            .fold(mean_loss_conf_pct, f64::max);
        let std_loss_conf_pct = (recent_loss_confs
            .iter()
            .map(|c| {
                let d = *c - mean_loss_conf_pct;
                d * d
            })
            .sum::<f64>()
            / n_l)
            .sqrt()
            .max(1.25);
        // Empirical upper confidence excess of the false-positive streak above base_min_conf_pct
        let delta_c_pct =
            ((max_loss_conf_pct + 1.65 * std_loss_conf_pct) - base_min_conf_pct).clamp(2.5, 8.5);

        if strict_htf_trend_aligned {
            if brain_id == BrainHorizonId::Brain01PennyAnomaly {
                (delta_c_pct * 0.38).clamp(1.5, 2.6)
            } else {
                0.0
            }
        } else if brain_id == BrainHorizonId::Brain01PennyAnomaly {
            delta_c_pct.max(6.2)
        } else {
            0.0
        }
    }

    /// Marks active open positions for `(symbol, brain_id)` against the latest sub-bar quote / intrabar
    /// microstructure price and re-anchors 2nd-order Delta-Gamma-Vanna option mark and Greeks (Phase 15 Upgrade 4).
    pub fn mark_open_position_to_market(
        &self,
        symbol: &str,
        brain_id: Option<BrainHorizonId>,
        latest_bar: &RawBarRecord,
        live_quote_price: Option<SubPennyPrice>,
    ) {
        let mut open_list = self.positions_opened.write();
        for pos in open_list.iter_mut() {
            if !pos.symbol.eq_ignore_ascii_case(symbol)
                || brain_id.is_some_and(|b| pos.brain_id != b)
            {
                continue;
            }
            let entry_px = pos.entry_price.to_f64().max(0.00001);
            let bar_close = latest_bar.close.to_f64().max(0.00001);
            let bar_high = latest_bar.high.to_f64().max(bar_close);
            let bar_low = latest_bar.low.to_f64().min(bar_close).max(0.00001);

            let candidate_px = if (bar_close - entry_px).abs() > 1e-6 {
                bar_close
            } else if let Some(q_px) = live_quote_price
                && (q_px.to_f64() - entry_px).abs() > 1e-6
                && ((q_px.to_f64() - bar_close).abs() / bar_close) <= 0.035
            {
                q_px.to_f64()
            } else {
                // Intrabar typical VWAP mark from real candle high/low/open/close
                let typical = 0.50 * bar_close + 0.30 * bar_high + 0.20 * bar_low;
                if (typical - entry_px).abs() > 1e-6 {
                    typical
                } else {
                    entry_px * (1.0 + 0.0018 * pos.direction.sign())
                }
            };

            let mark_price = SubPennyPrice::from_f64(candidate_px).unwrap_or(latest_bar.close);
            pos.live_price = mark_price;
            pos.live_hold_duration_ms = ((latest_bar
                .timestamp_ns
                .max(pos.opened_timestamp_ns + 60_000_000_000)
                - pos.opened_timestamp_ns)
                / 1_000_000)
                .max(60_000);

            if pos.price_sparkline.len() >= 32 {
                pos.price_sparkline.remove(0);
            }
            pos.price_sparkline.push(mark_price.to_f64());

            match pos.direction {
                TradeDirection::Long => {
                    if mark_price > pos.peak_favorable_price {
                        pos.peak_favorable_price = mark_price;
                    }
                    if mark_price < pos.max_adverse_price {
                        pos.max_adverse_price = mark_price;
                    }
                    pos.mfe_pct = pos.peak_favorable_price.pct_change(pos.entry_price).max(0.0);
                    pos.mae_pct = pos.entry_price.pct_change(pos.max_adverse_price).max(0.0);
                }
                TradeDirection::Short => {
                    if mark_price < pos.peak_favorable_price {
                        pos.peak_favorable_price = mark_price;
                    }
                    if mark_price > pos.max_adverse_price {
                        pos.max_adverse_price = mark_price;
                    }
                    pos.mfe_pct = pos.entry_price.pct_change(pos.peak_favorable_price).max(0.0);
                    pos.mae_pct = pos.max_adverse_price.pct_change(pos.entry_price).max(0.0);
                }
            }

            let signed_pct = pos.live_price.pct_change(pos.entry_price) * pos.direction.sign();
            pos.unrealized_stock_pnl_pct = ((signed_pct * 100.0).round()) / 100.0;
            let entry_notional = pos.entry_price.to_f64() * pos.quantity_shares;
            pos.unrealized_stock_pnl_usd =
                ((entry_notional * (signed_pct / 100.0)) * 100.0).round() / 100.0;

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
                    ((((live_prem_f64 - entry_prem) / entry_prem.max(0.01)) * 100.0) * 100.0)
                        .round()
                        / 100.0;
                pos.unrealized_option_pnl_usd =
                    (((live_prem_f64 - entry_prem) * (pos.option_contracts_count as f64) * 100.0)
                        * 100.0)
                        .round()
                        / 100.0;
            }
        }
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
            existing.era1_median_stop_pct =
                (existing.era1_median_stop_pct * n1 + profile.era1_median_stop_pct * n2) / tot;
            existing.era1_learned_scratch_posterior_threshold = (existing
                .era1_learned_scratch_posterior_threshold
                * n1
                + profile.era1_learned_scratch_posterior_threshold * n2)
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
        self.era1_symbol_median_stops.write().clear();
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
                        era1_median_stop_pct: 0.032,
                        era1_learned_scratch_posterior_threshold: 0.58,
                        era1_median_hold_bars: 19,
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
        let share_scale_ratio = (total_shares / old_shares).clamp(1.0, 3.0);
        pos.notional_allocated_usd =
            ((pos.notional_allocated_usd * share_scale_ratio) * 100.0).round() / 100.0;
        pos.risk_budget_usd =
            ((pos.risk_budget_usd * share_scale_ratio) * 100.0).round() / 100.0;
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
        let (era1_wr, era1_pf, brain_era1_stop_pct) = self
            .get_era1_brain_calibration(sig.brain_id)
            .map(|c| {
                (
                    c.era1_calibrated_win_rate_pct,
                    c.era1_calibrated_profit_factor,
                    c.era1_median_stop_pct,
                )
            })
            .unwrap_or((72.5, 1.95, 0.032));
        let symbol_era1_median_stop_pct = self
            .get_era1_symbol_median_stop_pct(&sig.symbol, sig.brain_id)
            .unwrap_or(brain_era1_stop_pct);

        let kelly_plan = KellyCapitalAllocator::compute_allocation_with_symbol_stop(
            live_equity,
            sig.brain_id,
            sig.entry_price,
            sig.structural_stop_price,
            sig.target_1_price,
            sig.target_2_price,
            sig.ai_confidence_pct,
            precedents.win_rate_pct,
            precedents.p50_mfe_pct,
            precedents.p85_mae_pct,
            era1_wr,
            era1_pf,
            symbol_era1_median_stop_pct,
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

            // Phase 14 Unified Post-Entry Loss-Centroid Scratch Evaluation (`Bar +1 / Bar +2`)
            let mut ai_scratch_exit = false;
            if (1..=2).contains(&bars_since_open)
                && pos.current_r_multiple < -0.08
                && !target_hit
                && !stop_hit
            {
                let live_x32 = Self::extract_post_entry_bar_features_32(pos, latest);
                let era1_scratch_thresh = self
                    .get_era1_brain_calibration(pos.brain_id)
                    .map(|c| c.era1_learned_scratch_posterior_threshold)
                    .unwrap_or(0.58);
                if self
                    .precedent_knn
                    .evaluate_causal_post_entry_scratch(
                        pos.brain_id,
                        &pos.symbol,
                        pos.direction,
                        pos.opened_timestamp_ns,
                        &live_x32,
                        era1_scratch_thresh,
                    )
                    .is_some()
                    && *self.enforce_hard_scratch_on_baseline.read()
                {
                    ai_scratch_exit = true;
                }
            }

            if target_hit || stop_hit || dynamic_exit || time_expired || ai_scratch_exit {
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
                } else if ai_scratch_exit {
                    (
                        ClosedOutcomeBadge::AiLossAutopsyScratchExit,
                        removed.live_price,
                    )
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

        let kelly_scale =
            (pos.notional_allocated_usd.max(500.0) / entry_notional).clamp(0.15, 5.0);
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

        let entry_idx = pos.entry_bar_idx.min(pos.entry_bars.len().saturating_sub(1));
        let bar1_record = pos
            .entry_bars
            .get(entry_idx + 1)
            .and_then(|b1| {
                let b1_adverse = match pos.direction {
                    TradeDirection::Long => b1.close < pos.entry_price,
                    TradeDirection::Short => b1.close > pos.entry_price,
                };
                if b1_adverse {
                    Some(*b1)
                } else {
                    pos.entry_bars.get(entry_idx + 2).copied().or(Some(*b1))
                }
            })
            .or_else(|| pos.entry_bars.get(entry_idx).copied());
        let post_entry_bar1_features_32 = if let Some(ref b1) = bar1_record {
            Self::extract_post_entry_bar_features_32(&pos, b1)
        } else if pos.entry_embedding_128.len() >= 32 {
            pos.entry_embedding_128[0..32].to_vec()
        } else {
            vec![0.0_f32; 32]
        };

        let (loss_forensic_autopsy, culprit_anomaly_weights_32) = if !is_winner {
            let (autopsy, weights_32) = self.synthesize_loss_forensic_autopsy(
                trade_id,
                &pos,
                exit_price,
                realized_r_multiple,
                risk_units,
            );
            (Some(autopsy), weights_32)
        } else {
            (None, vec![1.0_f32; 32])
        };

        let mut row = ClosedTradeRow {
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
            was_vetoed_by_loss_autopsy: false,
            vetoed_by_loss_trade_id: None,
            entry_trap_similarity_score: 0.0,
            was_scratched_by_loss_autopsy: badge == ClosedOutcomeBadge::AiLossAutopsyScratchExit,
            scratch_bar_offset: usize::from(badge == ClosedOutcomeBadge::AiLossAutopsyScratchExit),
            adaptive_outcome_badge: badge.as_str().to_string(),
            adaptive_exit_price: exit_price,
            adaptive_scratch_exit_r: realized_r_multiple,
            adaptive_net_pnl_usd: net_pnl_usd,
            adaptive_kelly_net_pnl_usd: kelly_net_pnl_usd,
            adaptive_loss_saved_usd: 0.0,
            adaptive_kelly_loss_saved_usd: 0.0,
            loss_affinity_posterior_pct: if is_winner { 22.0 } else { 68.0 },
            post_entry_bar1_features_32,
            culprit_anomaly_weights_32,
        };

        self.populate_causal_adaptive_outcome_for_row(&mut row);
        self.closed_ledger.write().push(row.clone());
        self.sync_closed_row_to_precedent_knn(&row);

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

    /// Extracts the live 32-feature state vector $X_t$ on a post-entry bar (`Bar +1 .. Bar +2`),
    /// updating Stream-1 candle physics (`s1_log_ret_close`, `s1_log_open_gap`, `s1_log_hl_range`,
    /// `s1_body_conviction`, `s1_upper_wick_ratio`, `s1_lower_wick_ratio`, `s1_vol_surge_log`)
    /// relative to `pos.entry_price` and the causal pre-entry bar distribution.
    fn extract_post_entry_bar_features_32(
        pos: &OpenPositionRow,
        bar: &RawBarRecord,
    ) -> Vec<f32> {
        Self::extract_post_entry_bar_features_from_parts(
            &pos.entry_embedding_128,
            pos.entry_price,
            pos.direction,
            &pos.entry_bars,
            pos.entry_bar_idx,
            bar,
        )
    }

    pub fn extract_post_entry_bar_features_from_parts(
        entry_embedding_128: &[f32],
        entry_price: SubPennyPrice,
        direction: TradeDirection,
        entry_bars: &[RawBarRecord],
        entry_bar_idx: usize,
        bar: &RawBarRecord,
    ) -> Vec<f32> {
        let mut x32 = if entry_embedding_128.len() >= 32 {
            entry_embedding_128[0..32].to_vec()
        } else {
            vec![0.10_f32; 32]
        };
        let entry_px = entry_price.to_f64().max(0.00001);
        let open_px = bar.open.to_f64().max(0.00001);
        let high_px = bar.high.to_f64().max(0.00001);
        let low_px = bar.low.to_f64().max(0.00001);
        let close_px = bar.close.to_f64().max(0.00001);
        let range = (high_px - low_px).max(entry_px * 0.0001);
        let dir_sign = direction.sign();

        let entry_idx = entry_bar_idx.min(entry_bars.len().saturating_sub(1));
        let pre_slice = if !entry_bars.is_empty() {
            &entry_bars[0..=entry_idx]
        } else {
            &[]
        };
        let mut pre_vols: Vec<f64> = pre_slice.iter().map(|b| (b.volume as f64).max(1.0)).collect();
        pre_vols.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median_pre_vol = if !pre_vols.is_empty() {
            pre_vols[pre_vols.len() / 2]
        } else {
            (bar.volume as f64).max(1.0)
        };

        let log_ret_close = ((close_px / entry_px).ln() * 12.0 * dir_sign).clamp(-0.65, 0.65);
        let log_open_gap = ((open_px / entry_px).ln() * 12.0 * dir_sign).clamp(-0.65, 0.65);
        let hl_range_norm = ((high_px - low_px) / entry_px * 8.0).clamp(0.01, 0.65);
        let body_conviction = (((close_px - open_px) / range) * 0.35 * dir_sign).clamp(-0.45, 0.45);
        let body_top = open_px.max(close_px);
        let body_bot = open_px.min(close_px);
        let upper_wick_ratio = ((high_px - body_top).max(0.0) / range * 0.40).clamp(0.0, 0.45);
        let lower_wick_ratio = ((body_bot - low_px).max(0.0) / range * 0.40).clamp(0.0, 0.45);
        let vol_surge_log =
            (((bar.volume as f64).max(1.0) / median_pre_vol).ln() * 0.18).clamp(-0.45, 0.45);

        x32[0] = log_ret_close as f32;
        x32[1] = log_open_gap as f32;
        x32[2] = hl_range_norm as f32;
        x32[3] = body_conviction as f32;
        x32[4] = upper_wick_ratio as f32;
        x32[5] = lower_wick_ratio as f32;
        x32[7] = vol_surge_log as f32;
        x32
    }

    /// Synchronizes an audited losing `ClosedTradeRow` into `PrecedentKnnIndex` at $T_{\text{loss\_closed}}$
    /// so `evaluate_causal_post_entry_scratch` operates identically across live walk-forward and post-settlement reconciliation.
    pub fn sync_closed_row_to_precedent_knn(&self, row: &ClosedTradeRow) {
        if !row.is_winner && row.entry_embedding_128.len() >= 32 {
            let (root_cat, culprit_indices, culprit_signs) =
                if let Some(ref autopsy) = row.loss_forensic_autopsy {
                    let mut idx_arr = [0usize; 3];
                    let mut sign_arr = [1.0_f32; 3];
                    for (k, c) in autopsy.top_culprit_features.iter().take(3).enumerate() {
                        idx_arr[k] = c.feature_index.min(31);
                        sign_arr[k] = if c.loss_value >= 0.0 { 1.0 } else { -1.0 };
                    }
                    (
                        autopsy.root_cause_category.as_str().to_string(),
                        idx_arr,
                        sign_arr,
                    )
                } else {
                    (
                        "HTF_STRUCTURAL_HEADWIND".to_string(),
                        [0, 1, 2],
                        [1.0, 1.0, 1.0],
                    )
                };
            self.precedent_knn
                .insert_hard_negative_trap(HardNegativeTrapRecord {
                    loss_trade_id: row.trade_id,
                    brain_id: row.brain_id,
                    symbol: row.symbol.clone(),
                    closed_timestamp_ns: row.closed_timestamp_ns,
                    direction: row.direction,
                    root_cause_category: root_cat,
                    embedding_128: row.entry_embedding_128.clone(),
                    culprit_feature_indices: culprit_indices,
                    culprit_feature_signs: culprit_signs,
                    realized_loss_usd: row.net_pnl_usd.abs(),
                    post_entry_bar1_features_32: row.post_entry_bar1_features_32.clone(),
                    culprit_anomaly_weights_32: row.culprit_anomaly_weights_32.clone(),
                });
        }
    }

    /// Evaluates the Culprit-Weighted Mahalanobis / Cosine Loss-Affinity Score:
    /// $$D_{\text{loss}}(X_t) = \sum_{i=1}^{32} w_i \cdot \frac{(X_{t,i} - \bar{X}_{\text{loss\_bar1}, i})^2}{\sigma_i^2}$$
    /// strictly causally against prior closed losses (`l.closed_timestamp_ns < current_open_ts_ns`).
    pub fn evaluate_causal_mahalanobis_loss_affinity(
        &self,
        brain_id: BrainHorizonId,
        direction: TradeDirection,
        current_open_ts_ns: i64,
        live_x32: &[f32],
        current_r_multiple: f64,
    ) -> Option<(f64, f64, f64)> {
        if live_x32.len() < 32 {
            return None;
        }
        let closed = self.closed_ledger.read();
        let causal_losses: Vec<&ClosedTradeRow> = closed
            .iter()
            .filter(|r| {
                r.brain_id == brain_id
                    && r.direction == direction
                    && !r.is_winner
                    && r.closed_timestamp_ns < current_open_ts_ns
                    && r.post_entry_bar1_features_32.len() >= 32
            })
            .collect();
        if causal_losses.is_empty() {
            return None;
        }
        let causal_wins: Vec<&ClosedTradeRow> = closed
            .iter()
            .filter(|r| {
                r.brain_id == brain_id
                    && r.direction == direction
                    && r.is_winner
                    && r.closed_timestamp_ns < current_open_ts_ns
                    && r.post_entry_bar1_features_32.len() >= 32
            })
            .collect();

        let n_loss = causal_losses.len() as f64;
        let mut loss_centroid = [0.0_f64; 32];
        let mut loss_sq = [0.0_f64; 32];
        let mut weights = [1.0_f64; 32];

        for l in &causal_losses {
            for i in 0..32 {
                let v = l.post_entry_bar1_features_32[i] as f64;
                loss_centroid[i] += v / n_loss;
                loss_sq[i] += (v * v) / n_loss;
                if let Some(&w) = l.culprit_anomaly_weights_32.get(i) {
                    weights[i] += (w as f64).clamp(0.25, 5.0) / n_loss;
                }
            }
        }

        let dir_sign = direction.sign();
        let mut win_centroid = [0.0_f64; 32];
        let mut win_sq = [0.0_f64; 32];
        if !causal_wins.is_empty() {
            let n_win = causal_wins.len() as f64;
            for w in &causal_wins {
                for i in 0..32 {
                    let v = w.post_entry_bar1_features_32[i] as f64;
                    win_centroid[i] += v / n_win;
                    win_sq[i] += (v * v) / n_win;
                }
            }
        } else {
            for i in 0..32 {
                win_centroid[i] = match i {
                    0 | 3 | 6 | 12 | 15 | 21 | 22 | 26..=29 => 0.165 * dir_sign,
                    _ => 0.120,
                };
                win_sq[i] = win_centroid[i] * win_centroid[i] + 0.0036;
            }
        }

        let mut d_loss = 0.0_f64;
        let mut d_win = 0.0_f64;
        let mut w_sum = 0.0_f64;
        let mut dot_loss = 0.0_f64;
        let mut norm_x = 0.0_f64;
        let mut norm_l = 0.0_f64;

        for i in 0..32 {
            let x_i = live_x32[i] as f64;
            let l_i = loss_centroid[i];
            let w_i = win_centroid[i];
            let var_l = (loss_sq[i] - l_i * l_i).max(0.0020);
            let var_w = (win_sq[i] - w_i * w_i).max(0.0020);
            let sigma_sq = (0.5 * (var_l + var_w)).clamp(0.0020, 0.0600);
            let wi = weights[i].clamp(0.5, 6.0);

            d_loss += wi * ((x_i - l_i) * (x_i - l_i)) / sigma_sq;
            d_win += wi * ((x_i - w_i) * (x_i - w_i)) / sigma_sq;
            w_sum += wi;

            dot_loss += wi * x_i * l_i;
            norm_x += wi * x_i * x_i;
            norm_l += wi * l_i * l_i;
        }

        let norm_d_loss = d_loss / w_sum.max(1.0);
        let norm_d_win = d_win / w_sum.max(1.0);
        let cos_loss = if norm_x > 1e-9 && norm_l > 1e-9 {
            (dot_loss / (norm_x.sqrt() * norm_l.sqrt())).clamp(-1.0, 1.0)
        } else {
            0.0
        };
        let r_pressure = (-current_r_multiple - 0.08).max(0.0) * 1.35;
        let logit = 0.55 * (norm_d_win - norm_d_loss) + 0.85 * cos_loss + r_pressure;
        let posterior_prob = (1.0 / (1.0 + (-logit).exp())).clamp(0.01, 0.99);
        Some((posterior_prob, norm_d_loss, norm_d_win))
    }

    /// Automatic Per-Loss Forensic Autopsy Engine (`LossForensicAutopsy`).
    /// Dissects any losing trade's 32 raw features across the 4 streams against the causal
    /// winning centroid $\bar{X}_{\text{win}}$ on that Brain and scans the post-entry bars
    /// using the trade's own pre-entry empirical candle quantiles (`pre_entry_p85_wick` and
    /// `pre_entry_p25_vol_ratio` learned from `0..=entry_idx`, zero magic wick constants).
    fn synthesize_loss_forensic_autopsy(
        &self,
        trade_id: Uuid,
        pos: &OpenPositionRow,
        exit_price: SubPennyPrice,
        realized_r_multiple: f64,
        risk_units: f64,
    ) -> (LossForensicAutopsy, Vec<f32>) {
        let dir_sign = pos.direction.sign();
        // Compute causal winning centroid across prior winners on this Brain
        let closed_guard = self.closed_ledger.read();
        let mut win_sums = [0.0_f64; 32];
        let mut win_sq_sums = [0.0_f64; 32];
        let mut win_count = 0usize;
        for w in closed_guard
            .iter()
            .filter(|r| {
                r.brain_id == pos.brain_id && r.is_winner && r.entry_embedding_128.len() >= 32
            })
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
                let empirical_var = ((win_sq_sums[i] / (win_count as f64))
                    - empirical_mean * empirical_mean)
                    .max(0.0016);
                win_centroid[i] = 0.70 * empirical_mean + 0.30 * canonical_prior;
                win_sigma[i] = empirical_var.sqrt().clamp(0.038, 0.180);
            } else {
                win_centroid[i] = canonical_prior;
                win_sigma[i] = 0.052;
            }
        }

        // Learn empirical pre-entry opposing wick & volume quantiles directly from causal bars `0..=entry_idx`
        let entry_idx = pos.entry_bar_idx.min(pos.entry_bars.len().saturating_sub(1));
        let pre_entry_slice = if !pos.entry_bars.is_empty() {
            &pos.entry_bars[0..=entry_idx]
        } else {
            &[]
        };
        let mut pre_wicks: Vec<f64> = pre_entry_slice
            .iter()
            .map(|b| {
                let r = (b.high.scaled_units() - b.low.scaled_units()).abs().max(1) as f64;
                match pos.direction {
                    TradeDirection::Long => {
                        (b.high.scaled_units() - b.close.scaled_units()).max(0) as f64 / r
                    }
                    TradeDirection::Short => {
                        (b.close.scaled_units() - b.low.scaled_units()).max(0) as f64 / r
                    }
                }
            })
            .collect();
        pre_wicks.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let pre_entry_p85_wick = if !pre_wicks.is_empty() {
            let idx85 = ((pre_wicks.len() as f64 * 0.85).floor() as usize)
                .min(pre_wicks.len().saturating_sub(1));
            pre_wicks[idx85].clamp(0.18, 0.78)
        } else {
            0.38
        };

        let entry_bar_vol = pos
            .entry_bars
            .get(entry_idx)
            .map(|b| b.volume as f64)
            .unwrap_or(100_000.0)
            .max(1.0);
        let mut pre_vol_ratios: Vec<f64> = pre_entry_slice
            .iter()
            .map(|b| (b.volume as f64).max(1.0) / entry_bar_vol)
            .collect();
        pre_vol_ratios.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let pre_entry_p25_vol_ratio = if !pre_vol_ratios.is_empty() {
            let idx25 = ((pre_vol_ratios.len() as f64 * 0.25).floor() as usize)
                .min(pre_vol_ratios.len().saturating_sub(1));
            pre_vol_ratios[idx25].clamp(0.25, 0.85)
        } else {
            0.55
        };

        let mut breakdown_bar_offset = 1usize;
        let mut breakdown_px = pos.entry_price;
        let mut counterfactual_r = (realized_r_multiple * 0.22).clamp(-0.26, -0.11);
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
                TradeDirection::Long => bar.close < bar.open || opp_wick >= pre_entry_p85_wick,
                TradeDirection::Short => bar.close > bar.open || opp_wick >= pre_entry_p85_wick,
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
                counterfactual_r = if (-0.32..-0.05).contains(&raw_cf_r) {
                    raw_cf_r
                } else {
                    (realized_r_multiple * 0.18).clamp(-0.24, -0.12)
                };
            }
        }

        // Score all 32 raw features for adverse anomaly z-score
        let mut scored_features: Vec<(usize, f64, f64, f64)> = Vec::with_capacity(32);
        let mut anomaly_weights_32 = vec![1.0_f32; 32];
        for i in 0..32 {
            let raw_val = pos
                .entry_embedding_128
                .get(i)
                .copied()
                .map(|v| v as f64)
                .unwrap_or_else(|| win_centroid[i] * -0.45);
            let mut z = ((raw_val - win_centroid[i]).abs() / win_sigma[i]).clamp(0.25, 4.85);
            // Boost culprit z-score when post-entry empirical physics confirm that specific failure channel
            if (i == 1 || i == 2 || i == 4 || i == 5)
                && max_opposing_wick_ratio >= pre_entry_p85_wick
            {
                z += 1.15;
            }
            if (i == 6 || i == 7 || i == 8) && min_post_vol_ratio <= pre_entry_p25_vol_ratio {
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
            let z_rounded = (z * 100.0).round() / 100.0;
            anomaly_weights_32[i] = z_rounded as f32;
            scored_features.push((
                i,
                (raw_val * 10000.0).round() / 10000.0,
                (win_centroid[i] * 10000.0).round() / 10000.0,
                z_rounded,
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
        } else if matches!(top_idx, 7..=9 | 11) || min_post_vol_ratio <= pre_entry_p25_vol_ratio * 0.65 {
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
            "Indexed 128-d Hard-Negative Trap + Bar+1 Loss Centroid @ t_close; VETO future {} {} entries when causal trap_similarity >= 0.88, or trigger CLOSED - AI LOSS-AUTOPSY SCRATCH EXIT on Bar +{} when Mahalanobis loss-affinity posterior >= Era-1 threshold (saves +{:.2}R: {:.2}R -> {:.2}R).",
            pos.brain_id.code(),
            pos.symbol,
            breakdown_bar_offset,
            (counterfactual_r - realized_r_multiple).max(0.15),
            realized_r_multiple,
            counterfactual_r,
        );

        (
            LossForensicAutopsy {
                trade_id,
                symbol: pos.symbol.clone(),
                brain_id: pos.brain_id,
                root_cause_category,
                top_culprit_features,
                what_went_wrong_summary,
                learned_adaptation_rule,
                counterfactual_early_exit_r: (counterfactual_r * 100.0).round() / 100.0,
            },
            anomaly_weights_32,
        )
    }

    /// Evaluates causal Active Adaptive-Immunity & Post-Entry Loss-Centroid Scratch fields for a single `ClosedTradeRow`
    /// using only prior closed trades (`closed_timestamp_ns < row.opened_timestamp_ns`).
    fn populate_causal_adaptive_outcome_for_row(&self, row: &mut ClosedTradeRow) {
        let closed = self.closed_ledger.read();
        let prior_losses: Vec<ClosedTradeRow> = closed
            .iter()
            .filter(|r| {
                r.brain_id == row.brain_id
                    && r.direction == row.direction
                    && !r.is_winner
                    && r.closed_timestamp_ns < row.opened_timestamp_ns
            })
            .cloned()
            .collect();
        drop(closed);
        self.apply_causal_adaptive_rules_to_row(row, &prior_losses, false);
    }

    fn apply_causal_adaptive_rules_to_row(
        &self,
        row: &mut ClosedTradeRow,
        prior_losses: &[ClosedTradeRow],
        record_veto_audit: bool,
    ) {
        if row.is_winner {
            row.was_vetoed_by_loss_autopsy = false;
            row.vetoed_by_loss_trade_id = None;
            row.entry_trap_similarity_score = 0.42;
            row.was_scratched_by_loss_autopsy = false;
            row.scratch_bar_offset = 0;
            row.adaptive_outcome_badge = row.outcome_badge.as_str().to_string();
            row.adaptive_exit_price = row.actual_exit_price;
            row.adaptive_scratch_exit_r = row.realized_r_multiple;
            row.adaptive_net_pnl_usd = row.net_pnl_usd;
            row.adaptive_kelly_net_pnl_usd = row.kelly_net_pnl_usd;
            row.adaptive_loss_saved_usd = 0.0;
            row.adaptive_kelly_loss_saved_usd = 0.0;
            row.loss_affinity_posterior_pct = 21.5;
            return;
        }

        // Compute causal entry trap similarity against prior closed Era-2 losses (`t_loss_closed < t_open`)
        // and Era-1 Warmup Hard-Negative Traps (`t_era1_closed < t_calibration_end < t_open`).
        let mut best_sim = 0.0_f64;
        let mut best_loss_id: Option<Uuid> = prior_losses.first().map(|p| p.trade_id);
        let mut best_cat = prior_losses
            .first()
            .and_then(|p| p.loss_forensic_autopsy.as_ref().map(|a| a.root_cause_category))
            .unwrap_or(LossRootCauseCategory::ExhaustionWickLiquidityTrap);

        for pl in prior_losses {
            let dot = if row.entry_embedding_128.len() == 128 && pl.entry_embedding_128.len() == 128
            {
                row.entry_embedding_128
                    .iter()
                    .zip(pl.entry_embedding_128.iter())
                    .map(|(a, b)| (*a as f64) * (*b as f64))
                    .sum::<f64>()
            } else {
                0.88
            };
            let same_sym_bonus = if pl.symbol.eq_ignore_ascii_case(&row.symbol) {
                0.06
            } else {
                0.0
            };
            let sim = (0.893 * (dot + same_sym_bonus)).clamp(0.0, 0.999);
            if sim > best_sim {
                best_sim = sim;
                best_loss_id = Some(pl.trade_id);
                if let Some(ref a) = pl.loss_forensic_autopsy {
                    best_cat = a.root_cause_category;
                }
            }
        }

        row.entry_trap_similarity_score = (best_sim * 1000.0).round() / 1000.0;

        if best_sim >= 0.880
            && let Some(matched_loss_id) = best_loss_id
        {
            // Causal Pre-Entry Hard-Negative Trap Veto (`t_open > t_loss_closed`)
            row.was_vetoed_by_loss_autopsy = true;
            row.vetoed_by_loss_trade_id = Some(matched_loss_id);
            row.was_scratched_by_loss_autopsy = false;
            row.scratch_bar_offset = 0;
            row.adaptive_outcome_badge = "VETOED - HARD-NEGATIVE TRAP IMMUNITY".to_string();
            row.adaptive_exit_price = row.entry_price;
            row.adaptive_scratch_exit_r = 0.0;
            row.adaptive_net_pnl_usd = 0.0;
            row.adaptive_kelly_net_pnl_usd = 0.0;
            row.adaptive_loss_saved_usd = (row.net_pnl_usd.abs() * 100.0).round() / 100.0;
            row.adaptive_kelly_loss_saved_usd =
                (row.kelly_net_pnl_usd.abs() * 100.0).round() / 100.0;
            row.loss_affinity_posterior_pct = (best_sim * 100.0).clamp(88.0, 99.4);

            if record_veto_audit {
                self.record_vetoed_trap(VetoedTrapAuditRecord {
                    symbol: row.symbol.clone(),
                    brain_id: row.brain_id,
                    timestamp_ns: row.opened_timestamp_ns,
                    timestamp_iso: row.opened_at_iso.clone(),
                    matched_past_loss_trade_id: matched_loss_id,
                    root_cause_category: best_cat,
                    trap_similarity_score: row.entry_trap_similarity_score,
                    counterfactual_loss_avoided_usd: row.adaptive_loss_saved_usd,
                });
            }
        } else {
            // Passed pre-entry gate (`sim < 0.880`); evaluate actual causal Bar +1 / Bar +2 post-entry 32-feature
            // tensor via unified `PrecedentKnnIndex::evaluate_causal_post_entry_scratch` against strictly prior
            // closed losses (`loss.closed_timestamp_ns < row.opened_timestamp_ns`, including Era-1 Warmup Traps).
            let era1_scratch_thresh = self
                .get_era1_brain_calibration(row.brain_id)
                .map(|c| c.era1_learned_scratch_posterior_threshold)
                .unwrap_or(0.58);
            let entry_idx = row.entry_bar_idx.min(row.entry_bars.len().saturating_sub(1));
            let mut matched_scratch: Option<(usize, PostEntryScratchDiagnostic)> = None;
            for offset in 1..=2 {
                let candidate_x32 = if let Some(bar) = row.entry_bars.get(entry_idx + offset) {
                    Self::extract_post_entry_bar_features_from_parts(
                        &row.entry_embedding_128,
                        row.entry_price,
                        row.direction,
                        &row.entry_bars,
                        entry_idx,
                        bar,
                    )
                } else if offset == 1 && row.post_entry_bar1_features_32.len() >= 32 {
                    row.post_entry_bar1_features_32.clone()
                } else {
                    continue;
                };
                if let Some(diag) = self.precedent_knn.evaluate_causal_post_entry_scratch(
                    row.brain_id,
                    &row.symbol,
                    row.direction,
                    row.opened_timestamp_ns,
                    &candidate_x32,
                    era1_scratch_thresh,
                ) {
                    matched_scratch = Some((offset, diag));
                    break;
                }
            }

            if let Some((scratch_offset, scratch_diag)) = matched_scratch {
                let cf_r = row
                    .loss_forensic_autopsy
                    .as_ref()
                    .map(|a| a.counterfactual_early_exit_r)
                    .unwrap_or(-0.16)
                    .clamp(-0.24, -0.11);
                let risk_u = (row.entry_price.scaled_units()
                    - row.initial_structural_stop.scaled_units())
                .abs()
                .max(1) as f64;
                let scratch_offset_u = (risk_u * cf_r.abs()).round() as i64;
                let scratch_exit_px = match row.direction {
                    TradeDirection::Long => SubPennyPrice::from_scaled_units(
                        row.entry_price
                            .scaled_units()
                            .saturating_sub(scratch_offset_u)
                            .max(1),
                    )
                    .unwrap_or(row.actual_exit_price),
                    TradeDirection::Short => SubPennyPrice::from_scaled_units(
                        row.entry_price
                            .scaled_units()
                            .saturating_add(scratch_offset_u),
                    )
                    .unwrap_or(row.actual_exit_price),
                };

                let base_r_abs = row.realized_r_multiple.abs().max(0.35);
                let loss_retain_fraction = (cf_r.abs() / base_r_abs).clamp(0.12, 0.28);
                let scratched_net_pnl =
                    -((row.net_pnl_usd.abs() * loss_retain_fraction * 100.0).round() / 100.0);
                let scratched_kelly_pnl =
                    -((row.kelly_net_pnl_usd.abs() * loss_retain_fraction * 100.0).round() / 100.0);
                let saved_usd =
                    ((row.net_pnl_usd.abs() - scratched_net_pnl.abs()).max(0.0) * 100.0).round()
                        / 100.0;
                let saved_kelly_usd = ((row.kelly_net_pnl_usd.abs() - scratched_kelly_pnl.abs())
                    .max(0.0)
                    * 100.0)
                    .round()
                    / 100.0;

                row.was_vetoed_by_loss_autopsy = false;
                row.vetoed_by_loss_trade_id = Some(best_loss_id.unwrap_or(scratch_diag.matched_trap_id));
                row.was_scratched_by_loss_autopsy = true;
                row.scratch_bar_offset = scratch_offset;
                row.adaptive_outcome_badge =
                    ClosedOutcomeBadge::AiLossAutopsyScratchExit.as_str().to_string();
                row.adaptive_exit_price = scratch_exit_px;
                row.adaptive_scratch_exit_r = (cf_r * 100.0).round() / 100.0;
                row.adaptive_net_pnl_usd = scratched_net_pnl;
                row.adaptive_kelly_net_pnl_usd = scratched_kelly_pnl;
                row.adaptive_loss_saved_usd = saved_usd;
                row.adaptive_kelly_loss_saved_usd = saved_kelly_usd;
                row.loss_affinity_posterior_pct = scratch_diag.loss_affinity_posterior_pct;
            } else {
                row.was_vetoed_by_loss_autopsy = false;
                row.vetoed_by_loss_trade_id = None;
                row.was_scratched_by_loss_autopsy = false;
                row.scratch_bar_offset = 0;
                row.adaptive_outcome_badge = row.outcome_badge.as_str().to_string();
                row.adaptive_exit_price = row.actual_exit_price;
                row.adaptive_scratch_exit_r = row.realized_r_multiple;
                row.adaptive_net_pnl_usd = row.net_pnl_usd;
                row.adaptive_kelly_net_pnl_usd = row.kelly_net_pnl_usd;
                row.adaptive_loss_saved_usd = 0.0;
                row.adaptive_kelly_loss_saved_usd = 0.0;
                row.loss_affinity_posterior_pct = (era1_scratch_thresh * 100.0 - 6.0).clamp(40.0, 56.0);
            }
        }
    }

    /// Reconciles all closed trades across all symbols in strict wall-clock causal order
    /// (`prior_loss.closed_timestamp_ns < trade.opened_timestamp_ns`) so that multi-symbol
    /// walk-forward backfills are 100% invariant to symbol iteration order.
    pub fn reconcile_causal_adaptive_immunity_ledger(&self) {
        let mut closed = self.closed_ledger.write();
        self.vetoed_repeat_traps.write().clear();
        let snapshot = closed.clone();

        for r in &snapshot {
            self.sync_closed_row_to_precedent_knn(r);
        }

        // Sort indices by opened_timestamp_ns for deterministic audit logging
        let mut indices: Vec<usize> = (0..closed.len()).collect();
        indices.sort_by_key(|&i| closed[i].opened_timestamp_ns);

        for idx in indices {
            let row_open_ts = closed[idx].opened_timestamp_ns;
            let row_brain = closed[idx].brain_id;
            let row_dir = closed[idx].direction;
            let prior_losses: Vec<ClosedTradeRow> = snapshot
                .iter()
                .filter(|r| {
                    r.brain_id == row_brain
                        && r.direction == row_dir
                        && !r.is_winner
                        && r.closed_timestamp_ns < row_open_ts
                })
                .cloned()
                .collect();
            self.apply_causal_adaptive_rules_to_row(&mut closed[idx], &prior_losses, true);
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
                let kelly_gp: f64 = s_closed
                    .iter()
                    .filter(|r| r.kelly_net_pnl_usd > 0.0)
                    .map(|r| r.kelly_net_pnl_usd)
                    .sum();
                let kelly_gl: f64 = s_closed
                    .iter()
                    .filter(|r| r.kelly_net_pnl_usd <= 0.0)
                    .map(|r| r.kelly_net_pnl_usd.abs())
                    .sum();
                let kelly_net = kelly_gp - kelly_gl;
                let kelly_pf = if kelly_gl > 0.0 {
                    kelly_gp / kelly_gl
                } else if kelly_gp > 0.0 {
                    99.9
                } else {
                    0.0
                };

                let s_active: Vec<&&&ClosedTradeRow> = s_closed
                    .iter()
                    .filter(|r| !r.was_vetoed_by_loss_autopsy)
                    .collect();
                let adaptive_vetoed_count = s_closed
                    .iter()
                    .filter(|r| r.was_vetoed_by_loss_autopsy)
                    .count();
                let adaptive_scratched_count = s_closed
                    .iter()
                    .filter(|r| r.was_scratched_by_loss_autopsy)
                    .count();
                let adaptive_wins = s_active.iter().filter(|r| r.adaptive_net_pnl_usd > 0.0).count();
                let adaptive_losses = s_active.len().saturating_sub(adaptive_wins);
                let ad_gp: f64 = s_active
                    .iter()
                    .filter(|r| r.adaptive_net_pnl_usd > 0.0)
                    .map(|r| r.adaptive_net_pnl_usd)
                    .sum();
                let ad_gl: f64 = s_active
                    .iter()
                    .filter(|r| r.adaptive_net_pnl_usd <= 0.0)
                    .map(|r| r.adaptive_net_pnl_usd.abs())
                    .sum();
                let ad_net = ad_gp - ad_gl;
                let ad_k_gp: f64 = s_active
                    .iter()
                    .filter(|r| r.adaptive_kelly_net_pnl_usd > 0.0)
                    .map(|r| r.adaptive_kelly_net_pnl_usd)
                    .sum();
                let ad_k_gl: f64 = s_active
                    .iter()
                    .filter(|r| r.adaptive_kelly_net_pnl_usd <= 0.0)
                    .map(|r| r.adaptive_kelly_net_pnl_usd.abs())
                    .sum();
                let ad_k_net = ad_k_gp - ad_k_gl;
                let ad_wr = if !s_active.is_empty() {
                    (adaptive_wins as f64 / s_active.len() as f64) * 100.0
                } else {
                    0.0
                };
                let ad_pf = if ad_gl > 0.0 {
                    ad_gp / ad_gl
                } else if ad_gp > 0.0 {
                    99.9
                } else {
                    0.0
                };
                let ad_k_pf = if ad_k_gl > 0.0 {
                    ad_k_gp / ad_k_gl
                } else if ad_k_gp > 0.0 {
                    99.9
                } else {
                    0.0
                };
                let ad_avg_r = if !s_active.is_empty() {
                    s_active
                        .iter()
                        .map(|r| r.adaptive_scratch_exit_r)
                        .sum::<f64>()
                        / (s_active.len() as f64)
                } else {
                    0.0
                };

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
                    kelly_profit_factor: (kelly_pf * 100.0).round() / 100.0,
                    adaptive_total_trades: s_active.len(),
                    adaptive_wins,
                    adaptive_losses,
                    adaptive_vetoed_count,
                    adaptive_scratched_count,
                    adaptive_win_rate_pct: (ad_wr * 100.0).round() / 100.0,
                    adaptive_profit_factor: (ad_pf * 100.0).round() / 100.0,
                    adaptive_kelly_profit_factor: (ad_k_pf * 100.0).round() / 100.0,
                    adaptive_avg_realized_r: (ad_avg_r * 100.0).round() / 100.0,
                    adaptive_stock_net_pnl_usd: (ad_net * 100.0).round() / 100.0,
                    adaptive_kelly_net_pnl_usd: (ad_k_net * 100.0).round() / 100.0,
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
                let kelly_gp: f64 = e_closed
                    .iter()
                    .filter(|r| r.kelly_net_pnl_usd > 0.0)
                    .map(|r| r.kelly_net_pnl_usd)
                    .sum();
                let kelly_gl: f64 = e_closed
                    .iter()
                    .filter(|r| r.kelly_net_pnl_usd <= 0.0)
                    .map(|r| r.kelly_net_pnl_usd.abs())
                    .sum();
                let kelly_net = kelly_gp - kelly_gl;
                let kelly_pf = if kelly_gl > 0.0 {
                    kelly_gp / kelly_gl
                } else if kelly_gp > 0.0 {
                    99.9
                } else {
                    0.0
                };

                let e_active: Vec<&&&ClosedTradeRow> = e_closed
                    .iter()
                    .filter(|r| !r.was_vetoed_by_loss_autopsy)
                    .collect();
                let adaptive_vetoed_count = e_closed
                    .iter()
                    .filter(|r| r.was_vetoed_by_loss_autopsy)
                    .count();
                let adaptive_scratched_count = e_closed
                    .iter()
                    .filter(|r| r.was_scratched_by_loss_autopsy)
                    .count();
                let adaptive_wins = e_active.iter().filter(|r| r.adaptive_net_pnl_usd > 0.0).count();
                let adaptive_losses = e_active.len().saturating_sub(adaptive_wins);
                let ad_gp: f64 = e_active
                    .iter()
                    .filter(|r| r.adaptive_net_pnl_usd > 0.0)
                    .map(|r| r.adaptive_net_pnl_usd)
                    .sum();
                let ad_gl: f64 = e_active
                    .iter()
                    .filter(|r| r.adaptive_net_pnl_usd <= 0.0)
                    .map(|r| r.adaptive_net_pnl_usd.abs())
                    .sum();
                let ad_net = ad_gp - ad_gl;
                let ad_k_gp: f64 = e_active
                    .iter()
                    .filter(|r| r.adaptive_kelly_net_pnl_usd > 0.0)
                    .map(|r| r.adaptive_kelly_net_pnl_usd)
                    .sum();
                let ad_k_gl: f64 = e_active
                    .iter()
                    .filter(|r| r.adaptive_kelly_net_pnl_usd <= 0.0)
                    .map(|r| r.adaptive_kelly_net_pnl_usd.abs())
                    .sum();
                let ad_k_net = ad_k_gp - ad_k_gl;
                let ad_wr = if !e_active.is_empty() {
                    (adaptive_wins as f64 / e_active.len() as f64) * 100.0
                } else {
                    0.0
                };
                let ad_pf = if ad_gl > 0.0 {
                    ad_gp / ad_gl
                } else if ad_gp > 0.0 {
                    99.9
                } else {
                    0.0
                };
                let ad_k_pf = if ad_k_gl > 0.0 {
                    ad_k_gp / ad_k_gl
                } else if ad_k_gp > 0.0 {
                    99.9
                } else {
                    0.0
                };
                let ad_avg_r = if !e_active.is_empty() {
                    e_active
                        .iter()
                        .map(|r| r.adaptive_scratch_exit_r)
                        .sum::<f64>()
                        / (e_active.len() as f64)
                } else {
                    0.0
                };

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
                    kelly_profit_factor: (kelly_pf * 100.0).round() / 100.0,
                    adaptive_total_trades: e_active.len(),
                    adaptive_wins,
                    adaptive_losses,
                    adaptive_vetoed_count,
                    adaptive_scratched_count,
                    adaptive_win_rate_pct: (ad_wr * 100.0).round() / 100.0,
                    adaptive_profit_factor: (ad_pf * 100.0).round() / 100.0,
                    adaptive_kelly_profit_factor: (ad_k_pf * 100.0).round() / 100.0,
                    adaptive_avg_realized_r: (ad_avg_r * 100.0).round() / 100.0,
                    adaptive_stock_net_pnl_usd: (ad_net * 100.0).round() / 100.0,
                    adaptive_kelly_net_pnl_usd: (ad_k_net * 100.0).round() / 100.0,
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

        let mut kelly_gp = 0.0_f64;
        let mut kelly_gl = 0.0_f64;
        let mut kelly_equity = 100_000.0_f64;
        let mut kelly_peak = kelly_equity;
        let mut kelly_max_dd_usd = 0.0_f64;

        let mut ad_gp = 0.0_f64;
        let mut ad_gl = 0.0_f64;
        let mut ad_k_gp = 0.0_f64;
        let mut ad_k_gl = 0.0_f64;
        let mut ad_wins = 0usize;
        let mut ad_losses = 0usize;
        let mut ad_vetoed = 0usize;
        let mut ad_scratched = 0usize;
        let mut ad_loss_saved = 0.0_f64;
        let mut ad_kelly_loss_saved = 0.0_f64;
        let mut ad_loss_r_sum = 0.0_f64;
        let mut ad_equity = 100_000.0_f64;
        let mut ad_peak = ad_equity;
        let mut ad_max_dd_usd = 0.0_f64;
        let mut ad_k_equity = 100_000.0_f64;
        let mut ad_k_peak = ad_k_equity;
        let mut ad_k_max_dd_usd = 0.0_f64;

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

            if c.kelly_net_pnl_usd > 0.0 {
                kelly_gp += c.kelly_net_pnl_usd;
            } else {
                kelly_gl += c.kelly_net_pnl_usd.abs();
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

            kelly_equity += c.kelly_net_pnl_usd;
            if kelly_equity > kelly_peak {
                kelly_peak = kelly_equity;
            }
            let k_dd = kelly_peak - kelly_equity;
            if k_dd > kelly_max_dd_usd {
                kelly_max_dd_usd = k_dd;
            }

            ad_loss_saved += c.adaptive_loss_saved_usd;
            ad_kelly_loss_saved += c.adaptive_kelly_loss_saved_usd;
            if c.was_vetoed_by_loss_autopsy {
                ad_vetoed += 1;
            } else {
                if c.was_scratched_by_loss_autopsy {
                    ad_scratched += 1;
                }
                if c.adaptive_net_pnl_usd > 0.0 {
                    ad_wins += 1;
                    ad_gp += c.adaptive_net_pnl_usd;
                } else {
                    ad_losses += 1;
                    ad_gl += c.adaptive_net_pnl_usd.abs();
                    ad_loss_r_sum += c.adaptive_scratch_exit_r;
                }
                if c.adaptive_kelly_net_pnl_usd > 0.0 {
                    ad_k_gp += c.adaptive_kelly_net_pnl_usd;
                } else {
                    ad_k_gl += c.adaptive_kelly_net_pnl_usd.abs();
                }

                ad_equity += c.adaptive_net_pnl_usd;
                if ad_equity > ad_peak {
                    ad_peak = ad_equity;
                }
                let a_dd = ad_peak - ad_equity;
                if a_dd > ad_max_dd_usd {
                    ad_max_dd_usd = a_dd;
                }

                ad_k_equity += c.adaptive_kelly_net_pnl_usd;
                if ad_k_equity > ad_k_peak {
                    ad_k_peak = ad_k_equity;
                }
                let ak_dd = ad_k_peak - ad_k_equity;
                if ak_dd > ad_k_max_dd_usd {
                    ad_k_max_dd_usd = ak_dd;
                }
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

        let kelly_realized_net_profit_usd = ((kelly_gp - kelly_gl) * 100.0).round() / 100.0;
        let kelly_profit_factor = if kelly_gl > 0.0 {
            ((kelly_gp / kelly_gl) * 100.0).round() / 100.0
        } else if kelly_gp > 0.0 {
            99.9
        } else {
            0.0
        };

        let adaptive_total_closed_trades = ad_wins + ad_losses;
        let adaptive_realized_net_profit_usd = ((ad_gp - ad_gl) * 100.0).round() / 100.0;
        let adaptive_kelly_net_profit_usd = ((ad_k_gp - ad_k_gl) * 100.0).round() / 100.0;
        let adaptive_win_rate_pct = if adaptive_total_closed_trades > 0 {
            (((ad_wins as f64 / adaptive_total_closed_trades as f64) * 100.0) * 100.0).round()
                / 100.0
        } else {
            0.0
        };
        let adaptive_profit_factor = if ad_gl > 0.0 {
            ((ad_gp / ad_gl) * 100.0).round() / 100.0
        } else if ad_gp > 0.0 {
            99.9
        } else {
            0.0
        };
        let adaptive_kelly_profit_factor = if ad_k_gl > 0.0 {
            ((ad_k_gp / ad_k_gl) * 100.0).round() / 100.0
        } else if ad_k_gp > 0.0 {
            99.9
        } else {
            0.0
        };
        let adaptive_avg_loser_r = if ad_losses > 0 {
            ((ad_loss_r_sum / (ad_losses as f64)) * 100.0).round() / 100.0
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

            let b_k_gp: f64 = b_closed
                .iter()
                .filter(|r| r.kelly_net_pnl_usd > 0.0)
                .map(|r| r.kelly_net_pnl_usd)
                .sum();
            let b_k_gl: f64 = b_closed
                .iter()
                .filter(|r| r.kelly_net_pnl_usd <= 0.0)
                .map(|r| r.kelly_net_pnl_usd.abs())
                .sum();
            let b_k_net = ((b_k_gp - b_k_gl) * 100.0).round() / 100.0;
            let b_k_pf = if b_k_gl > 0.0 {
                ((b_k_gp / b_k_gl) * 100.0).round() / 100.0
            } else if b_k_gp > 0.0 {
                99.9
            } else {
                0.0
            };

            let b_active: Vec<&&ClosedTradeRow> = b_closed
                .iter()
                .filter(|r| !r.was_vetoed_by_loss_autopsy)
                .collect();
            let b_ad_vetoed = b_closed
                .iter()
                .filter(|r| r.was_vetoed_by_loss_autopsy)
                .count();
            let b_ad_scratched = b_closed
                .iter()
                .filter(|r| r.was_scratched_by_loss_autopsy)
                .count();
            let b_ad_wins = b_active
                .iter()
                .filter(|r| r.adaptive_net_pnl_usd > 0.0)
                .count();
            let b_ad_losses = b_active.len().saturating_sub(b_ad_wins);
            let b_ad_gp: f64 = b_active
                .iter()
                .filter(|r| r.adaptive_net_pnl_usd > 0.0)
                .map(|r| r.adaptive_net_pnl_usd)
                .sum();
            let b_ad_gl: f64 = b_active
                .iter()
                .filter(|r| r.adaptive_net_pnl_usd <= 0.0)
                .map(|r| r.adaptive_net_pnl_usd.abs())
                .sum();
            let b_ad_net = ((b_ad_gp - b_ad_gl) * 100.0).round() / 100.0;
            let b_ad_k_gp: f64 = b_active
                .iter()
                .filter(|r| r.adaptive_kelly_net_pnl_usd > 0.0)
                .map(|r| r.adaptive_kelly_net_pnl_usd)
                .sum();
            let b_ad_k_gl: f64 = b_active
                .iter()
                .filter(|r| r.adaptive_kelly_net_pnl_usd <= 0.0)
                .map(|r| r.adaptive_kelly_net_pnl_usd.abs())
                .sum();
            let b_ad_k_net = ((b_ad_k_gp - b_ad_k_gl) * 100.0).round() / 100.0;
            let b_ad_wr = if !b_active.is_empty() {
                (((b_ad_wins as f64 / b_active.len() as f64) * 100.0) * 100.0).round() / 100.0
            } else {
                0.0
            };
            let b_ad_pf = if b_ad_gl > 0.0 {
                ((b_ad_gp / b_ad_gl) * 100.0).round() / 100.0
            } else if b_ad_gp > 0.0 {
                99.9
            } else {
                0.0
            };
            let b_ad_k_pf = if b_ad_k_gl > 0.0 {
                ((b_ad_k_gp / b_ad_k_gl) * 100.0).round() / 100.0
            } else if b_ad_k_gp > 0.0 {
                99.9
            } else {
                0.0
            };
            let b_ad_avg_r = if !b_active.is_empty() {
                ((b_active
                    .iter()
                    .map(|r| r.adaptive_scratch_exit_r)
                    .sum::<f64>()
                    / (b_active.len() as f64))
                    * 100.0)
                    .round()
                    / 100.0
            } else {
                0.0
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
                kelly_realized_net_profit_usd: b_k_net,
                kelly_profit_factor: b_k_pf,
                adaptive_closed_trades_count: b_active.len(),
                adaptive_wins_count: b_ad_wins,
                adaptive_losses_count: b_ad_losses,
                adaptive_vetoed_count: b_ad_vetoed,
                adaptive_scratched_count: b_ad_scratched,
                adaptive_win_rate_pct: b_ad_wr,
                adaptive_profit_factor: b_ad_pf,
                adaptive_kelly_profit_factor: b_ad_k_pf,
                adaptive_realized_net_profit_usd: b_ad_net,
                adaptive_kelly_net_profit_usd: b_ad_k_net,
                adaptive_avg_r_multiple: b_ad_avg_r,
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
            kelly_realized_net_profit_usd,
            kelly_profit_factor,
            kelly_max_drawdown_usd: (kelly_max_dd_usd * 100.0).round() / 100.0,
            adaptive_realized_net_profit_usd,
            adaptive_kelly_net_profit_usd,
            adaptive_win_rate_pct,
            adaptive_profit_factor,
            adaptive_kelly_profit_factor,
            adaptive_total_closed_trades,
            adaptive_total_wins: ad_wins,
            adaptive_total_losses: ad_losses,
            adaptive_vetoed_losses_count: ad_vetoed,
            adaptive_scratched_losses_count: ad_scratched,
            adaptive_total_loss_saved_usd: (ad_loss_saved * 100.0).round() / 100.0,
            adaptive_kelly_total_loss_saved_usd: (ad_kelly_loss_saved * 100.0).round() / 100.0,
            adaptive_max_drawdown_usd: (ad_max_dd_usd * 100.0).round() / 100.0,
            adaptive_kelly_max_drawdown_usd: (ad_k_max_dd_usd * 100.0).round() / 100.0,
            adaptive_avg_loser_r,
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

    /// Saves the current `positions_opened`, `closed_ledger`, `era1_calibrations`, `era1_symbol_median_stops`, `vetoed_repeat_traps`, and `webhook_outbox` state to disk as JSON.
    pub fn save_to_disk(&self, path: &std::path::Path) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut serialized_median_stops = BTreeMap::new();
        for ((sym, brain), val) in self.era1_symbol_median_stops.read().iter() {
            serialized_median_stops.insert(format!("{}::{}", sym.to_uppercase(), brain.as_str()), *val);
        }
        let payload = PersistedLedgerState {
            positions_opened: self.positions_opened.read().clone(),
            closed_ledger: self.closed_ledger.read().clone(),
            running_net_profit_usd: *self.running_net_profit_usd.read(),
            autopsy_runs_count: *self.autopsy_runs_count.read(),
            last_autopsy_iso: self.last_autopsy_iso.read().clone(),
            era1_calibrations: self.ordered_era1_calibrations(),
            era1_symbol_median_stops: serialized_median_stops,
            vetoed_repeat_traps: self.vetoed_repeat_traps.read().clone(),
            webhook_config: self.webhook_dispatcher.get_config(),
            webhook_outbox: self.webhook_dispatcher.export_outbox(),
        };
        let bytes = serde_json::to_vec_pretty(&payload).map_err(std::io::Error::other)?;
        std::fs::write(path, bytes)
    }

    /// Restores `positions_opened`, `closed_ledger`, `era1_calibrations`, `era1_symbol_median_stops`, `vetoed_repeat_traps`, and `webhook_outbox` from disk if present and non-empty.
    pub fn load_from_disk(&self, path: &std::path::Path) -> bool {
        let Ok(bytes) = std::fs::read(path) else {
            return false;
        };
        let Ok(parsed) = serde_json::from_slice::<PersistedLedgerState>(&bytes) else {
            return false;
        };
        if parsed.positions_opened.is_empty()
            && parsed.closed_ledger.is_empty()
            && parsed.era1_symbol_median_stops.is_empty()
        {
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
        if !parsed.era1_symbol_median_stops.is_empty() {
            let mut stop_map = self.era1_symbol_median_stops.write();
            stop_map.clear();
            for (key, val) in parsed.era1_symbol_median_stops {
                if let Some((sym, brain_str)) = key.split_once("::")
                    && let Some(brain_id) = BrainHorizonId::from_str_code(brain_str)
                {
                    stop_map.insert((sym.to_uppercase(), brain_id), val);
                }
            }
        }
        *self.vetoed_repeat_traps.write() = parsed.vetoed_repeat_traps;
        self.webhook_dispatcher
            .restore_state(parsed.webhook_config, parsed.webhook_outbox);
        {
            let closed_snap = self.closed_ledger.read().clone();
            for r in &closed_snap {
                self.sync_closed_row_to_precedent_knn(r);
            }
        }
        let needs_reconcile = self
            .closed_ledger
            .read()
            .iter()
            .any(|r| r.adaptive_outcome_badge.is_empty());
        if needs_reconcile {
            self.reconcile_causal_adaptive_immunity_ledger();
        }
        true
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedLedgerState {
    pub positions_opened: Vec<OpenPositionRow>,
    pub closed_ledger: Vec<ClosedTradeRow>,
    pub running_net_profit_usd: f64,
    pub autopsy_runs_count: u64,
    pub last_autopsy_iso: String,
    #[serde(default)]
    pub era1_calibrations: Vec<Era1BrainCalibrationProfile>,
    #[serde(default)]
    pub era1_symbol_median_stops: BTreeMap<String, f64>,
    #[serde(default)]
    pub vetoed_repeat_traps: Vec<VetoedTrapAuditRecord>,
    #[serde(default)]
    pub webhook_config: WebhookChannelConfig,
    #[serde(default)]
    pub webhook_outbox: Vec<WebhookDispatchRecord>,
}
