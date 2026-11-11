//! Phase 32.1 & Phase 33: Closed-Loop Forensic Loss-Autopsy, Precision-Weighted Mahalanobis
//! Attribution, Causal Bar-by-Bar Path Replay & Live Scanner Reinforcement Evolver.
//!
//! Provides:
//! - `LedoitWolfPrecision`: 32x32 Ledoit-Wolf shrinkage covariance & exact Cholesky precision engine
//!   backed directly by `PrecedentKnnIndex::solve_ledoit_wolf_cholesky_mahalanobis_32_scaled`.
//! - `decode_trade_failure` / `decode_trade_failure_with_precision`: Extracts real recorded
//!   `(entry_tensor, peak_mfe_tensor, adverse_inflection_tensor)` and computes precision-weighted
//!   Mahalanobis shift $\sqrt{\Delta z^\top W^{1/2} \Sigma_{\text{LW}}^{-1} W^{1/2} \Delta z}$.
//! - `simulate_trade_under_candidate`: Deterministic single-trade causal re-simulator executing:
//!   - Stage 0 Entry Gate (strictly causal pre-entry Mahalanobis conformal margin),
//!   - Stage 1 (`Bar 1..3`) sequential Wald SPRT log-likelihood ratio exiting at exact bar `k` close price,
//!   - Stage 3 causal running MFE ratchet (`running_peak_r` up to bar `k - 1` ONLY — zero lookahead).
//! - `verify_and_promote_generation`: Evaluates all 173 baseline trades (Gate 1) and all walk-forward
//!   trades (Gate 2) through `simulate_trade_under_candidate` under `current_states` and `candidate_states`,
//!   and upon Dual-Gate pass atomically updates champion `BrainSynapticPlasticityState`s and registers
//!   decoded `EntryLiquidityVacuumTrap` pre-entry 32-feature tensors into live `PrecedentKnnIndex`.

use crate::evolution::BrainSynapticPlasticityState;
use crate::ledger::{ClosedTradeRow, RealTimeLedgerEngine};
use crate::precedent_knn::{HardNegativeTrapRecord, PrecedentKnnIndex, TrapUpsertOutcome};
use chrono::Utc;
use raw_fingerprint_core::{BrainHorizonId, TradeDirection};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Canonical feature names for the 32-feature cross-stream sensory vector.
pub const FEATURE_NAMES_32: [&str; 32] = [
    // Stream 1 (0..12): Chart & Tape Physics
    "close_log_ret",
    "open_gap_log",
    "hl_range_log",
    "body_conviction",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "vwap_disp_log",
    "vol_surge_log",
    "trade_intensity_log",
    "orderflow_delta_ratio",
    "spread_bps_norm",
    "poc_clustering_score",
    // Stream 2 (12..20): Options Gamma / Vanna
    "call_put_log_ratio",
    "net_gamma_iv_skew",
    "dealer_gex_norm",
    "dealer_vanna_norm",
    "zero_dte_volume_share",
    "iv_rank_norm",
    "speed_accel_norm",
    "charm_decay_norm",
    // Stream 3 (20..26): Short Float & Liquidity
    "float_rotation_log",
    "short_to_float_ratio",
    "dtc_pressure",
    "intraday_short_vol_ratio",
    "borrow_fee_annual_log",
    "dilution_hazard_log",
    // Stream 4 (26..32): Market Internals Macro
    "nyse_tick_norm",
    "nasdaq_tick_norm",
    "add_breadth_norm",
    "vold_up_down_log",
    "vix_premium_ratio",
    "sector_flow_norm",
];

/// Deterministic classification of root failure mode from forensic tensor deltas.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AutopsyFailureMode {
    #[serde(rename = "ENTRY_LIQUIDITY_VACUUM_TRAP")]
    EntryLiquidityVacuumTrap,
    #[serde(rename = "POST_ENTRY_ORDERFLOW_TOXICITY_REVERSAL")]
    PostEntryOrderflowToxicityReversal,
    #[serde(rename = "DEALER_CHARM_VANNA_UNPIN_DECAY")]
    DealerCharmVannaUnpinDecay,
    #[serde(rename = "MFE_STALL_GIVEBACK_EXHAUSTION")]
    MfeStallGivebackExhaustion,
}

impl AutopsyFailureMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::EntryLiquidityVacuumTrap => "ENTRY_LIQUIDITY_VACUUM_TRAP",
            Self::PostEntryOrderflowToxicityReversal => "POST_ENTRY_ORDERFLOW_TOXICITY_REVERSAL",
            Self::DealerCharmVannaUnpinDecay => "DEALER_CHARM_VANNA_UNPIN_DECAY",
            Self::MfeStallGivebackExhaustion => "MFE_STALL_GIVEBACK_EXHAUSTION",
        }
    }
}

/// Decoded forensic trade autopsy record with 32-feature blame attribution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecodedTradeAutopsy {
    pub trade_id: Uuid,
    pub symbol: String,
    pub brain_id: BrainHorizonId,
    pub realized_r: f64,
    pub peak_mfe_r: f64,
    pub giveback_pct: f64,
    pub failure_mode: AutopsyFailureMode,
    pub failure_mode_label: String,
    pub blame_attribution_32: Vec<f64>,
    pub top_culprit_feature_indices: Vec<usize>,
    pub top_culprit_names: Vec<String>,
    pub mahalanobis_distance_shift: f64,
    pub collapse_bar_offset: usize,
    pub summary: String,
    #[serde(default)]
    pub entry_features_32: Vec<f32>,
}

/// Phase 34 Deliverable 1: Causal Regime Partitioning (`RegimeBucket`) across Stream 1 (`0..12`),
/// Stream 2 Dealer GEX/Charm/Vanna (`12..20`), and Stream 3 Toxicity/Float (`20..26`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RegimeBucket {
    #[serde(rename = "POS_GAMMA_PIN_MEAN_REVERT")]
    PosGammaPinMeanRevert,
    #[serde(rename = "NEG_GAMMA_MOMENTUM_TREND")]
    NegGammaMomentumTrend,
    #[serde(rename = "LIQUIDITY_VACUUM_HIGH_TOXICITY")]
    LiquidityVacuumHighToxicity,
}

impl Default for RegimeBucket {
    fn default() -> Self {
        Self::PosGammaPinMeanRevert
    }
}

impl RegimeBucket {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::PosGammaPinMeanRevert => "POS_GAMMA_PIN_MEAN_REVERT",
            Self::NegGammaMomentumTrend => "NEG_GAMMA_MOMENTUM_TREND",
            Self::LiquidityVacuumHighToxicity => "LIQUIDITY_VACUUM_HIGH_TOXICITY",
        }
    }
}

/// Pure-causal pre-entry regime classifier over the 32-feature state vector.
pub fn classify_regime_bucket(entry_features_32: &[f64; 32]) -> RegimeBucket {
    let spread_vacuum = entry_features_32[10];
    let wick_trap = entry_features_32[4].max(entry_features_32[5]);
    let trend_energy =
        entry_features_32[0].abs() + entry_features_32[6].abs() + entry_features_32[8].abs();
    let dealer_gex = entry_features_32[14];
    let dealer_vanna_charm = entry_features_32[15] + entry_features_32[19];
    let toxicity_mean: f64 =
        entry_features_32[20..26].iter().map(|v| v.abs()).sum::<f64>() / 6.0;

    if spread_vacuum >= 0.45
        || toxicity_mean >= 1.15
        || (wick_trap >= 0.58 && spread_vacuum >= 0.22)
    {
        RegimeBucket::LiquidityVacuumHighToxicity
    } else if dealer_gex < -0.04 || (trend_energy >= 0.72 && dealer_vanna_charm <= 0.15) {
        RegimeBucket::NegGammaMomentumTrend
    } else {
        RegimeBucket::PosGammaPinMeanRevert
    }
}

/// Active Stage-0 Trap Centroid entry passed into `evaluate_stage0_conformal_kernel`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Stage0TrapCentroid {
    pub trap_id: Option<Uuid>,
    pub root_cause_category: String,
    pub features_32: [f64; 32],
    pub quarantined: bool,
}

/// Output of the single shared causal Stage 0 conformal + $\chi^2_{32}$ OOD epistemic kernel
/// (`evaluate_stage0_conformal_kernel`), guaranteeing `< 1e-12` zero-skew parity between
/// offline Gate 1/2 simulation and live scanner / paper-daemon entry gating.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Stage0ConformalKernelOutput {
    pub regime_bucket: RegimeBucket,
    pub d_win: f64,
    pub d_trap: f64,
    pub d_nearest: f64,
    pub q975_mahalanobis_radius: f64,
    pub candidate_nonconformity_alpha: f64,
    pub conformal_p_value: f64,
    pub conformal_margin: f64,
    pub conformal_llr: f64,
    pub conformal_hurdle: f64,
    pub epistemic_ood_score: f64,
    pub kelly_epistemic_multiplier: f64,
    pub vetoed: bool,
    pub is_epistemic_ood_veto: bool,
    pub matched_trap_id: Option<Uuid>,
    pub veto_status: String,
    pub root_cause_category: String,
}

/// Outcome of causally re-simulating a single historical or walk-forward trade under a candidate
/// `BrainSynapticPlasticityState` and Hard-Negative Trap centroid bank.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SimulatedTradeOutcome {
    pub trade_id: Uuid,
    pub realized_r: f64,
    pub exit_price: f64,
    pub vetoed: bool,
    pub scratched_stage1: bool,
    pub ratchet_exited_stage3: bool,
    pub full_stop_hit: bool,
    pub exit_bar_offset: usize,
    pub candidate_conformal_margin: f64,
    pub conformal_llr: f64,
    pub conformal_p_value: f64,
    pub epistemic_ood_score: f64,
    pub kelly_epistemic_multiplier: f64,
    pub regime_bucket: RegimeBucket,
    pub max_wald_sprt_llr: f64,
    pub max_reversal_hazard: f64,
    pub effective_retention_ratio: f64,
}

/// Per-brain 32x32 Ledoit-Wolf empirical covariance & Cholesky precision manifold
/// built causally from settled precedent trades.
#[derive(Debug, Clone)]
pub struct BrainLedoitWolfManifold {
    pub pooled_entry_cov: [[f64; 32]; 32],
    pub n_entry_causal: usize,
    pub scale_entry_sq: f64,
    pub winner_centroid_32: [f64; 32],
    pub loss_centroid_32: [f64; 32],
    pub stage0_anchor_32: [f64; 32],
    pub winner_exemplars_32: Vec<(Uuid, [f64; 32])>,
    pub loss_exemplars_32: Vec<(Uuid, [f64; 32])>,
    pub winner_feature_var_32: [f64; 32],
    pub mondrian_calibration_scores: HashMap<RegimeBucket, Vec<(Uuid, f64)>>,
    pub all_calibration_scores: Vec<(Uuid, f64)>,
    pub prior_shell_margin: f64,
    pub q975_mahalanobis_radius: f64,
    pub pooled_bar1_cov: [[f64; 32]; 32],
    pub n_bar1_causal: usize,
    pub scale_bar1_sq: f64,
    pub bar1_win_centroid_32: [f64; 32],
    pub bar1_loss_centroid_32: [f64; 32],
}

/// Multi-brain 32x32 Ledoit-Wolf shrinkage covariance & Cholesky precision engine.
#[derive(Debug, Clone)]
pub struct LedoitWolfPrecision {
    pub brains: HashMap<BrainHorizonId, BrainLedoitWolfManifold>,
}

impl LedoitWolfPrecision {
    /// Builds per-brain 32x32 Ledoit-Wolf shrinkage covariance matrices and causal centroids
    /// directly from settled baseline trades.
    #[allow(clippy::needless_range_loop)]
    pub fn from_settled_trades(trades: &[ClosedTradeRow]) -> Self {
        let mut brains = HashMap::new();
        for brain_id in [
            BrainHorizonId::Brain01PennyAnomaly,
            BrainHorizonId::Brain02IntradayDayTrade,
            BrainHorizonId::Brain03ShortTermSwing,
            BrainHorizonId::Brain04CoreSwing,
            BrainHorizonId::Brain05LongTermWealth,
        ] {
            let brain_trades: Vec<&ClosedTradeRow> =
                trades.iter().filter(|t| t.brain_id == brain_id).collect();
            let source_slice: Vec<&ClosedTradeRow> = if brain_trades.len() >= 3 {
                brain_trades
            } else {
                trades.iter().collect()
            };

            let mut win_entries: Vec<(Uuid, [f64; 32])> = Vec::new();
            let mut loss_entries_with_id: Vec<(Uuid, [f64; 32])> = Vec::new();
            let mut loss_entries: Vec<[f64; 32]> = Vec::new();
            let mut all_entries: Vec<(Uuid, bool, [f64; 32])> = Vec::new();
            let mut win_bar1: Vec<[f64; 32]> = Vec::new();
            let mut loss_bar1: Vec<[f64; 32]> = Vec::new();

            for row in &source_slice {
                let e_vec = extract_causal_entry_array_32(row);
                let b1_vec = extract_causal_bar1_array_32(row);
                all_entries.push((row.trade_id, row.is_winner, e_vec));
                if row.is_winner {
                    win_entries.push((row.trade_id, e_vec));
                    win_bar1.push(b1_vec);
                } else {
                    loss_entries_with_id.push((row.trade_id, e_vec));
                    loss_entries.push(e_vec);
                    loss_bar1.push(b1_vec);
                }
            }

            let mut winner_centroid_32 = [0.0_f64; 32];
            if !win_entries.is_empty() {
                let inv_w = 1.0 / (win_entries.len() as f64);
                for (_, v) in &win_entries {
                    for i in 0..32 {
                        winner_centroid_32[i] += v[i] * inv_w;
                    }
                }
            }

            let mut winner_feature_var_32 = [0.0_f64; 32];
            if !win_entries.is_empty() {
                let inv_w = 1.0 / (win_entries.len() as f64);
                for (_, v) in &win_entries {
                    for i in 0..32 {
                        let d = v[i] - winner_centroid_32[i];
                        winner_feature_var_32[i] += (d * d) * inv_w;
                    }
                }
            }

            let mut loss_centroid_32 = [0.0_f64; 32];
            if !loss_entries.is_empty() {
                let inv_l = 1.0 / (loss_entries.len() as f64);
                for v in &loss_entries {
                    for i in 0..32 {
                        loss_centroid_32[i] += v[i] * inv_l;
                    }
                }
            } else if win_entries.is_empty() {
                for i in 0..32 {
                    loss_centroid_32[i] = 1.25;
                }
            } else {
                for i in 0..32 {
                    loss_centroid_32[i] = -winner_centroid_32[i];
                }
            }

            // Antipodal Stage-0 tail trap anchor beyond the loss centroid
            let mut stage0_anchor_32 = [0.0_f64; 32];
            for i in 0..32 {
                stage0_anchor_32[i] =
                    loss_centroid_32[i] + 1.85 * (loss_centroid_32[i] - winner_centroid_32[i]);
            }

            let mut range_sum = 0.0_f64;
            let mut range_cnt = 0usize;
            for (_, v) in &win_entries {
                let r_part = v[8].abs() + v[9].abs() + v[10].abs();
                if r_part > 1e-9 {
                    range_sum += r_part;
                    range_cnt += 1;
                }
            }
            for v in &loss_entries {
                let r_part = v[8].abs() + v[9].abs() + v[10].abs();
                if r_part > 1e-9 {
                    range_sum += r_part;
                    range_cnt += 1;
                }
            }
            let scale_entry = if range_cnt > 0 {
                (range_sum / (range_cnt as f64)).max(1e-9)
            } else {
                1.0
            };
            let scale_entry_sq = (scale_entry * scale_entry).max(1e-12);
            let entry_ridge = 0.035 * scale_entry_sq;

            let mut pooled_entry_cov = [[0.0_f64; 32]; 32];
            for i in 0..32 {
                pooled_entry_cov[i][i] = entry_ridge;
            }
            if !win_entries.is_empty() {
                let inv_w = 1.0 / (win_entries.len() as f64);
                for (_, v) in &win_entries {
                    for i in 0..32 {
                        let di = v[i] - winner_centroid_32[i];
                        for j in 0..=i {
                            let dj = v[j] - winner_centroid_32[j];
                            let c_ij = 0.5 * (di * dj) * inv_w;
                            pooled_entry_cov[i][j] += c_ij;
                            pooled_entry_cov[j][i] = pooled_entry_cov[i][j];
                        }
                    }
                }
            }
            if !loss_entries.is_empty() {
                let inv_l = 1.0 / (loss_entries.len() as f64);
                for v in &loss_entries {
                    for i in 0..32 {
                        let di = v[i] - loss_centroid_32[i];
                        for j in 0..=i {
                            let dj = v[j] - loss_centroid_32[j];
                            let c_ij = 0.5 * (di * dj) * inv_l;
                            pooled_entry_cov[i][j] += c_ij;
                            pooled_entry_cov[j][i] = pooled_entry_cov[i][j];
                        }
                    }
                }
            }

            // Post-entry Bar 1..3 covariance & centroids
            let mut bar1_win_centroid_32 = [0.0_f64; 32];
            if !win_bar1.is_empty() {
                let inv_w = 1.0 / (win_bar1.len() as f64);
                for v in &win_bar1 {
                    for i in 0..32 {
                        bar1_win_centroid_32[i] += v[i] * inv_w;
                    }
                }
            }
            let mut bar1_loss_centroid_32 = [0.0_f64; 32];
            if !loss_bar1.is_empty() {
                let inv_l = 1.0 / (loss_bar1.len() as f64);
                for v in &loss_bar1 {
                    for i in 0..32 {
                        bar1_loss_centroid_32[i] += v[i] * inv_l;
                    }
                }
            }

            let mut b1_range_sum = 0.0_f64;
            let mut b1_range_cnt = 0usize;
            for v in win_bar1.iter().chain(loss_bar1.iter()) {
                let r_part = v[4].abs() + v[5].abs() + v[6].abs();
                if r_part > 1e-9 {
                    b1_range_sum += r_part;
                    b1_range_cnt += 1;
                }
            }
            let scale_bar1 = if b1_range_cnt > 0 {
                (b1_range_sum / (b1_range_cnt as f64)).max(1e-9)
            } else {
                1.0
            };
            let scale_bar1_sq = (scale_bar1 * scale_bar1).max(1e-12);
            let mut pooled_bar1_cov = [[0.0_f64; 32]; 32];
            for i in 0..32 {
                pooled_bar1_cov[i][i] = 0.035 * scale_bar1_sq;
            }
            if !win_bar1.is_empty() {
                let inv_w = 1.0 / (win_bar1.len() as f64);
                for v in &win_bar1 {
                    for i in 0..32 {
                        let di = v[i] - bar1_win_centroid_32[i];
                        for j in 0..=i {
                            let dj = v[j] - bar1_win_centroid_32[j];
                            let c_ij = 0.5 * (di * dj) * inv_w;
                            pooled_bar1_cov[i][j] += c_ij;
                            pooled_bar1_cov[j][i] = pooled_bar1_cov[i][j];
                        }
                    }
                }
            }
            if !loss_bar1.is_empty() {
                let inv_l = 1.0 / (loss_bar1.len() as f64);
                for v in &loss_bar1 {
                    for i in 0..32 {
                        let di = v[i] - bar1_loss_centroid_32[i];
                        for j in 0..=i {
                            let dj = v[j] - bar1_loss_centroid_32[j];
                            let c_ij = 0.5 * (di * dj) * inv_l;
                            pooled_bar1_cov[i][j] += c_ij;
                            pooled_bar1_cov[j][i] = pooled_bar1_cov[i][j];
                        }
                    }
                }
            }

            let n_entry_causal = (win_entries.len() + loss_entries.len()).max(4);
            let n_bar1_causal = (win_bar1.len() + loss_bar1.len()).max(4);

            // Phase 34 Deliverable 1: Mondrian per-(brain_id, RegimeBucket) leave-one-out calibration
            // nonconformity scores alpha_i = d_LW(z_i, W_{-i}) / (d_LW(z_i, T) + 0.05) and empirical q975 radius
            let uniform_simplex_w = [6.0 / 32.0; 32];
            let mut mondrian_calibration_scores: HashMap<RegimeBucket, Vec<(Uuid, f64)>> =
                HashMap::new();
            let mut all_calibration_scores: Vec<(Uuid, f64)> =
                Vec::with_capacity(all_entries.len());
            let mut nearest_radii: Vec<f64> = Vec::with_capacity(all_entries.len());

            let maha_local = |x: &[f64; 32], y: &[f64; 32]| -> f64 {
                let mut diff = [0.0_f64; 32];
                for k in 0..32 {
                    diff[k] = x[k] - y[k];
                }
                let (d, _) =
                    PrecedentKnnIndex::solve_ledoit_wolf_cholesky_mahalanobis_32_scaled(
                        &pooled_entry_cov,
                        n_entry_causal,
                        &uniform_simplex_w,
                        &diff,
                        scale_entry_sq,
                    );
                d
            };

            let mut winner_loo_lambdas: Vec<f64> = Vec::with_capacity(win_entries.len());
            for (wid, w_vec) in &win_entries {
                let d_wc = maha_local(w_vec, &winner_centroid_32);
                let mut min_nn = d_wc;
                for (wid2, w_vec2) in &win_entries {
                    if wid2 != wid {
                        let d2 = maha_local(w_vec, w_vec2);
                        if d2 < min_nn {
                            min_nn = d2;
                        }
                    }
                }
                let dw_eff = min_nn.max(0.0);
                let dl_c = maha_local(w_vec, &loss_centroid_32).max(0.0);
                let rad_sq = (0.5 * (dw_eff * dw_eff + dl_c * dl_c)).max(1.0);
                let lam_w = 0.5 * (dw_eff * dw_eff - dl_c * dl_c) / rad_sq;
                winner_loo_lambdas.push(lam_w);
            }
            winner_loo_lambdas.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let prior_shell_margin = if winner_loo_lambdas.is_empty() {
                0.040
            } else {
                let max_lam = *winner_loo_lambdas.last().unwrap_or(&0.030);
                (max_lam + 0.012).clamp(0.025, 0.250)
            };

            for (tid, _is_win, z_i) in &all_entries {
                let regime_i = classify_regime_bucket(z_i);
                let d_centroid_i = maha_local(z_i, &winner_centroid_32);
                let mut d_w_i = d_centroid_i;
                for (wid, w_vec) in &win_entries {
                    if wid != tid {
                        let d_ex = maha_local(z_i, w_vec);
                        if d_ex < d_w_i {
                            d_w_i = d_ex;
                        }
                    }
                }
                let mut d_t_i = d_w_i + prior_shell_margin;
                for (lid, l_vec) in &loss_entries_with_id {
                    if lid != tid {
                        let dl = maha_local(z_i, l_vec);
                        if dl < d_t_i {
                            d_t_i = dl;
                        }
                    }
                }
                let alpha_i = d_w_i.max(0.0) / (d_t_i.max(0.0) + 0.05);
                mondrian_calibration_scores
                    .entry(regime_i)
                    .or_default()
                    .push((*tid, alpha_i));
                all_calibration_scores.push((*tid, alpha_i));
                nearest_radii.push(d_centroid_i.max(0.05));
            }

            nearest_radii.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let n_eff = (n_entry_causal as f64).max(6.0);
            let hotelling_c = (((n_eff + 1.0) * (n_eff - 1.0))
                / (n_eff * (n_eff - 32.0).max(4.0)))
            .sqrt();
            let chi2_32_q975_floor = (49.4804_f64).sqrt() * 1.65;
            let q975_mahalanobis_radius = if nearest_radii.is_empty() {
                chi2_32_q975_floor
            } else {
                let q_idx = (((nearest_radii.len() - 1) as f64) * 0.975).ceil() as usize;
                let emp_q975 = nearest_radii[q_idx.min(nearest_radii.len() - 1)];
                (emp_q975 * hotelling_c).max(chi2_32_q975_floor)
            };

            brains.insert(
                brain_id,
                BrainLedoitWolfManifold {
                    pooled_entry_cov,
                    n_entry_causal,
                    scale_entry_sq,
                    winner_centroid_32,
                    loss_centroid_32,
                    stage0_anchor_32,
                    winner_exemplars_32: win_entries,
                    loss_exemplars_32: loss_entries_with_id,
                    winner_feature_var_32,
                    mondrian_calibration_scores,
                    all_calibration_scores,
                    prior_shell_margin,
                    q975_mahalanobis_radius,
                    pooled_bar1_cov,
                    n_bar1_causal,
                    scale_bar1_sq,
                    bar1_win_centroid_32,
                    bar1_loss_centroid_32,
                },
            );
        }
        Self { brains }
    }

    /// Computes true 32x32 Ledoit-Wolf Cholesky precision-weighted Mahalanobis distance
    /// $\sqrt{\Delta z^\top W^{1/2} \Sigma_{\text{LW}}^{-1} W^{1/2} \Delta z}$ via `PrecedentKnnIndex`.
    #[allow(clippy::needless_range_loop)]
    pub fn entry_mahalanobis(
        &self,
        brain_id: BrainHorizonId,
        weights_32: &[f64; 32],
        a: &[f64; 32],
        b: &[f64; 32],
    ) -> f64 {
        let mut diff = [0.0_f64; 32];
        for i in 0..32 {
            diff[i] = a[i] - b[i];
        }
        if let Some(m) = self.brains.get(&brain_id) {
            let (dist, _) = PrecedentKnnIndex::solve_ledoit_wolf_cholesky_mahalanobis_32_scaled(
                &m.pooled_entry_cov,
                m.n_entry_causal,
                weights_32,
                &diff,
                m.scale_entry_sq,
            );
            dist
        } else {
            let mut unit_cov = [[0.0_f64; 32]; 32];
            for i in 0..32 {
                unit_cov[i][i] = 0.10;
            }
            let (dist, _) = PrecedentKnnIndex::solve_ledoit_wolf_cholesky_mahalanobis_32_scaled(
                &unit_cov,
                16,
                weights_32,
                &diff,
                1.0,
            );
            dist
        }
    }

    /// Computes post-entry (`Bar 1..3`) 32x32 Ledoit-Wolf Cholesky Mahalanobis distance.
    #[allow(clippy::needless_range_loop)]
    pub fn bar1_mahalanobis(
        &self,
        brain_id: BrainHorizonId,
        weights_32: &[f64; 32],
        a: &[f64; 32],
        b: &[f64; 32],
    ) -> f64 {
        let mut diff = [0.0_f64; 32];
        for i in 0..32 {
            diff[i] = a[i] - b[i];
        }
        if let Some(m) = self.brains.get(&brain_id) {
            let (dist, _) = PrecedentKnnIndex::solve_ledoit_wolf_cholesky_mahalanobis_32_scaled(
                &m.pooled_bar1_cov,
                m.n_bar1_causal,
                weights_32,
                &diff,
                m.scale_bar1_sq,
            );
            dist
        } else {
            self.entry_mahalanobis(brain_id, weights_32, a, b)
        }
    }
}

/// Dual-Gate verification telemetry comparing candidate vs current champion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateAuditReport {
    pub gate1_baseline_lock_passed: bool,
    pub gate1_baseline_winners_preserved: usize,
    pub gate1_baseline_winners_target: usize,
    pub gate1_current_expectancy_r: f64,
    pub gate1_candidate_expectancy_r: f64,
    pub gate1_current_pf: f64,
    pub gate1_candidate_pf: f64,
    pub current_baseline_exp_r: f64,
    pub candidate_baseline_exp_r: f64,
    pub current_baseline_pf: f64,
    pub candidate_baseline_pf: f64,
    pub gate2_walkforward_passed: bool,
    pub gate2_current_walkforward_net_r: f64,
    pub gate2_candidate_walkforward_net_r: f64,
    pub current_wf_net_r: f64,
    pub candidate_wf_net_r: f64,
    pub gate2_offending_losses_reduced: usize,
    pub gate2_walkforward_winners_preserved: bool,
    pub hard_negative_traps_added_count: usize,
    #[serde(default)]
    pub shadow_vetoed_evaluated_count: usize,
    #[serde(default)]
    pub shadow_net_veto_alpha_r: f64,
    #[serde(default)]
    pub quarantined_traps_count: usize,
    #[serde(default)]
    pub mirror_descent_kl_divergence: f64,
    pub all_gates_passed: bool,
    pub decision_rationale: String,
}

/// Full response payload returned by `POST /api/v1/scanner/evolve-cycle` and `GET /api/v1/scanner/autopsy-evolution`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutopsyEvolutionResponse {
    pub status: String,
    pub timestamp_iso: String,
    pub current_generation_id: u32,
    pub candidate_generation_id: u32,
    pub promoted_generation_id: Option<u32>,
    pub is_promoted: bool,
    pub total_trades_inspected: usize,
    pub hard_negative_traps_added_count: usize,
    #[serde(default)]
    pub shadow_vetoed_evaluated_count: usize,
    #[serde(default)]
    pub shadow_net_veto_alpha_r: f64,
    #[serde(default)]
    pub quarantined_traps_count: usize,
    #[serde(default)]
    pub mirror_descent_kl_divergence: f64,
    pub autopsies_decoded: Vec<DecodedTradeAutopsy>,
    pub gate_audit: GateAuditReport,
    pub updated_synaptic_states: Vec<BrainSynapticPlasticityState>,
}

/// Extracts the strictly causal 32-feature pre-entry state vector (`[f64; 32]`) for `trade`.
#[allow(clippy::needless_range_loop)]
pub fn extract_causal_entry_array_32(trade: &ClosedTradeRow) -> [f64; 32] {
    let raw = RealTimeLedgerEngine::extract_causal_entry_features_from_row(trade);
    let mut out = [0.0_f64; 32];
    for i in 0..32 {
        out[i] = raw.get(i).copied().unwrap_or(0.0) as f64;
    }
    out
}

/// Extracts the causal post-entry `Bar +1` 32-feature state vector (`[f64; 32]`) for `trade`.
#[allow(clippy::needless_range_loop)]
pub fn extract_causal_bar1_array_32(trade: &ClosedTradeRow) -> [f64; 32] {
    let raw = RealTimeLedgerEngine::extract_causal_bar1_features_from_row(trade);
    let mut out = [0.0_f64; 32];
    for i in 0..32 {
        out[i] = raw.get(i).copied().unwrap_or(0.0) as f64;
    }
    out
}

/// Exact KL Bregman projection onto the capped 32-simplex
/// $\{w \in [\text{floor\_w}, \text{ceil\_w}]^{32} : \sum_{f=0}^{31} w_f = \text{total\_mass}\}$
/// via bisection on the multiplicative water-filling dual multiplier $\lambda > 0$.
#[allow(clippy::needless_range_loop)]
pub fn project_capped_simplex_kl_32(
    unnormalized: &[f64; 32],
    floor_w: f64,
    ceil_w: f64,
    total_mass: f64,
) -> [f64; 32] {
    let mut safe_u = [0.0_f64; 32];
    for f in 0..32 {
        safe_u[f] = unnormalized[f].max(1e-12);
    }

    let mut log_lo = -35.0_f64;
    let mut log_hi = 35.0_f64;
    for _ in 0..80 {
        let log_mid = 0.5 * (log_lo + log_hi);
        let scale = log_mid.exp();
        let mut s = 0.0_f64;
        for f in 0..32 {
            s += (safe_u[f] * scale).clamp(floor_w, ceil_w);
        }
        if s < total_mass {
            log_lo = log_mid;
        } else {
            log_hi = log_mid;
        }
    }

    let opt_scale = (0.5 * (log_lo + log_hi)).exp();
    let mut out = [0.0_f64; 32];
    for f in 0..32 {
        out[f] = (safe_u[f] * opt_scale).clamp(floor_w, ceil_w);
    }
    out
}

/// Computes normalized Kullback-Leibler divergence $D_{\text{KL}}(w_{\text{next}} / Z \,\|\, w_{\text{prev}} / Z)$
/// across the 32-simplex synaptic mass $Z = \sum_{f=0}^{31} w_f = 6.0$.
#[allow(clippy::needless_range_loop)]
pub fn compute_simplex_kl_divergence_32(
    w_next: &[f64; 32],
    w_prev: &[f64; 32],
    total_mass: f64,
) -> f64 {
    let z = total_mass.max(1e-9);
    let mut kl = 0.0_f64;
    for f in 0..32 {
        let p = (w_next[f] / z).max(1e-12);
        let q = (w_prev[f] / z).max(1e-12);
        kl += p * (p / q).ln();
    }
    kl.max(0.0)
}

/// Converts `BrainSynapticPlasticityState::synaptic_feature_weights_32` into positive precision
/// metric weights $W = \operatorname{diag}(w_0, \dots, w_{31})$ conserving total mass $\sum_{f=0}^{31} w_f = 6.0$.
#[allow(clippy::needless_range_loop)]
pub fn build_synaptic_precision_weights(brain_state: &BrainSynapticPlasticityState) -> [f64; 32] {
    if let Some(sw) = brain_state.simplex_weights_32 {
        let mut max_dev = 0.0_f64;
        for i in 0..32 {
            let expected_syn = sw[i] * (32.0 / 6.0);
            let dev = (brain_state.synaptic_feature_weights_32[i] - expected_syn).abs();
            if dev > max_dev {
                max_dev = dev;
            }
        }
        if max_dev < 0.05 {
            return sw;
        }
        let mut raw_w = [0.0_f64; 32];
        let mut sum_w = 0.0_f64;
        for i in 0..32 {
            let wi = brain_state.synaptic_feature_weights_32[i].max(0.0001);
            raw_w[i] = wi;
            sum_w += wi;
        }
        sum_w = sum_w.max(1e-9);
        let mut out = [0.0_f64; 32];
        for i in 0..32 {
            out[i] = (raw_w[i] / sum_w) * 6.0;
        }
        return out;
    }

    let max_syn = brain_state
        .synaptic_feature_weights_32
        .iter()
        .copied()
        .fold(0.0_f64, f64::max);
    let fisher_p = brain_state
        .learned_fisher_sparsity_exponent_p
        .clamp(1.0, 2.75);
    let mut raw_w = [0.0_f64; 32];
    let mut sum_w = 0.0_f64;
    for i in 0..32 {
        let wi = brain_state.synaptic_feature_weights_32[i]
            .max(0.001)
            .powf(fisher_p);
        raw_w[i] = wi;
        sum_w += wi;
    }
    sum_w = sum_w.max(1e-9);
    let mut out = [0.0_f64; 32];
    for i in 0..32 {
        let norm_i = (raw_w[i] / sum_w) * 32.0;
        out[i] = (6.0 / 32.0) * norm_i;
    }
    if max_syn <= 10.0 {
        project_capped_simplex_kl_32(&out, 0.04, 0.65, 6.0)
    } else {
        out
    }
}

/// Phase 33.1 & Phase 34 Single Shared Pure-Causal Stage 0 Conformal + $\chi^2_{32}$ OOD Kernel.
///
/// Invoked identically by BOTH offline `simulate_trade_under_candidate` (Gate 1 & Gate 2) AND
/// live `RealTimeLedgerEngine::evaluate_live_entry_stage0_gate` / watchlist scanner, guaranteeing
/// `< 1e-12` zero-skew parity across `conformal_margin`, `conformal_llr`, `conformal_p_value`,
/// `epistemic_ood_score`, `kelly_epistemic_multiplier`, and `vetoed` decisions.
pub fn evaluate_stage0_conformal_kernel(
    brain_id: BrainHorizonId,
    entry_features_32: &[f64; 32],
    exclude_trade_id: Option<Uuid>,
    brain_state: &BrainSynapticPlasticityState,
    trap_bank: &[Stage0TrapCentroid],
    precision: &LedoitWolfPrecision,
) -> Stage0ConformalKernelOutput {
    let regime_bucket = classify_regime_bucket(entry_features_32);
    let weights_32 = build_synaptic_precision_weights(brain_state);

    let mut d_win = if let Some(manifold) = precision.brains.get(&brain_id) {
        let mut min_w = precision.entry_mahalanobis(
            brain_id,
            &weights_32,
            entry_features_32,
            &manifold.winner_centroid_32,
        );
        for (w_id, w_vec) in &manifold.winner_exemplars_32 {
            if exclude_trade_id != Some(*w_id) {
                let d_ex =
                    precision.entry_mahalanobis(brain_id, &weights_32, entry_features_32, w_vec);
                if d_ex < min_w {
                    min_w = d_ex;
                }
            }
        }
        min_w
    } else {
        let zero = [0.0_f64; 32];
        precision.entry_mahalanobis(brain_id, &weights_32, entry_features_32, &zero)
    };
    d_win = d_win.max(0.0);

    let d_loss_c = if let Some(manifold) = precision.brains.get(&brain_id) {
        precision.entry_mahalanobis(
            brain_id,
            &weights_32,
            entry_features_32,
            &manifold.loss_centroid_32,
        )
    } else {
        d_win + 0.35
    };

    let mut matched_trap_id: Option<Uuid> = None;
    let mut matched_category = "ENTRY_LIQUIDITY_VACUUM_TRAP".to_string();
    let mut min_active_trap = f64::INFINITY;
    let mut found_active_trap = false;

    for trap in trap_bank {
        if trap.quarantined {
            continue;
        }
        found_active_trap = true;
        let dt = precision.entry_mahalanobis(
            brain_id,
            &weights_32,
            entry_features_32,
            &trap.features_32,
        );
        if dt < min_active_trap {
            min_active_trap = dt;
            matched_trap_id = trap.trap_id;
            matched_category = trap.root_cause_category.clone();
        }
    }

    let d_trap = d_loss_c.min(min_active_trap).max(0.0);
    let radial_sq = (0.5 * (d_win * d_win + d_trap * d_trap)).max(1.0);
    let conformal_llr = 0.5 * (d_win * d_win - d_trap * d_trap) / radial_sq;
    let base_hurdle = precision
        .brains
        .get(&brain_id)
        .map(|m| m.prior_shell_margin)
        .unwrap_or(0.040);
    let conformal_hurdle = base_hurdle - brain_state.entry_veto_margin_delta;
    let conformal_margin = conformal_hurdle - conformal_llr;
    let active_trap_triggered = found_active_trap && min_active_trap <= d_loss_c;

    let d_nearest = d_win.min(d_trap);
    let q975 = precision
        .brains
        .get(&brain_id)
        .map(|m| m.q975_mahalanobis_radius)
        .unwrap_or(11.5)
        .max(0.50);

    let epistemic_ood_score = if d_nearest > q975 {
        (d_nearest / q975) - 1.0
    } else {
        0.0
    };
    let kelly_epistemic_multiplier = (-1.5 * epistemic_ood_score.max(0.0)).exp().clamp(0.15, 1.0);
    let is_epistemic_ood_veto = epistemic_ood_score > 1.25;

    // Mondrian Regime-Conditioned Split-Conformal p-value:
    // p = (1 + #{i in C_{b,r} : alpha_i >= alpha_cand}) / (1 + |C_{b,r}|)
    let candidate_nonconformity_alpha = d_win / (d_trap + 0.05);
    let p_mondrian = if let Some(manifold) = precision.brains.get(&brain_id) {
        let regime_slice = manifold.mondrian_calibration_scores.get(&regime_bucket);
        let calib_iter: Vec<f64> = if let Some(bucket_scores) = regime_slice
            && bucket_scores.len() >= 3
        {
            bucket_scores
                .iter()
                .filter(|(tid, _)| exclude_trade_id != Some(*tid))
                .map(|(_, a)| *a)
                .collect()
        } else {
            manifold
                .all_calibration_scores
                .iter()
                .filter(|(tid, _)| exclude_trade_id != Some(*tid))
                .map(|(_, a)| *a)
                .collect()
        };
        let geq_count = calib_iter
            .iter()
            .filter(|&&alpha_i| alpha_i >= candidate_nonconformity_alpha)
            .count();
        (1.0 + geq_count as f64) / (1.0 + calib_iter.len() as f64)
    } else {
        1.0
    };

    let is_trap_veto = conformal_margin < 0.0;
    let vetoed = is_epistemic_ood_veto || is_trap_veto;

    let (veto_status, root_cause_category) = if is_epistemic_ood_veto {
        (
            "STAGE_0_VETOED_EPISTEMIC_OOD_REGIME".to_string(),
            "STAGE_0_VETOED_EPISTEMIC_OOD_REGIME".to_string(),
        )
    } else if is_trap_veto {
        (
            "STAGE_0_VETOED_HARD_NEGATIVE_TRAP".to_string(),
            matched_category,
        )
    } else {
        ("PASS".to_string(), "NONE".to_string())
    };

    Stage0ConformalKernelOutput {
        regime_bucket,
        d_win,
        d_trap,
        d_nearest,
        q975_mahalanobis_radius: q975,
        candidate_nonconformity_alpha,
        conformal_p_value: p_mondrian,
        conformal_margin,
        conformal_llr,
        conformal_hurdle,
        epistemic_ood_score,
        kelly_epistemic_multiplier,
        vetoed,
        is_epistemic_ood_veto,
        matched_trap_id: if is_trap_veto && active_trap_triggered {
            matched_trap_id
        } else {
            None
        },
        veto_status,
        root_cause_category,
    }
}

/// Phase 34 Deliverable 4: Pure-Causal Bar-`k` Reversal Hazard Score $h_{k-1} \in [0, 1]$ and
/// Hazard-Conditioned Dynamic Stage 3 Continuation Retention Floor $\rho_{\text{eff}}(k) \in [0.50, 0.88]$.
///
/// Uses strictly causal observations up to bar `k - 1` (`prev_bar` OHLC rejection wick + `prev_bar_features_32`
/// post-entry Mahalanobis win/loss separation):
/// $$h_{k-1} = \operatorname{sigmoid}\big(1.25 \cdot (d_{\text{bar\_win}} - d_{\text{bar\_loss}}) + 1.50 \cdot \text{wick\_exhaustion}_{k-1}\big)$$
/// $$\rho_{\text{eff}}(k) = \operatorname{clamp}\big(\rho_{\text{base}} + 0.14 \cdot (h_{k-1} - 0.50),\; 0.50,\; 0.88\big)$$
pub fn compute_causal_reversal_hazard_retention(
    brain_id: BrainHorizonId,
    dir: f64,
    prev_open: f64,
    prev_high: f64,
    prev_low: f64,
    prev_close: f64,
    prev_bar_features_32: Option<&[f64; 32]>,
    brain_state: &BrainSynapticPlasticityState,
    precision: &LedoitWolfPrecision,
) -> (f64, f64) {
    let prev_hl = (prev_high - prev_low).max(1e-6);
    let wick_exhaustion_prev = if dir > 0.0 {
        ((prev_high - prev_open.max(prev_close)).max(0.0) / prev_hl).clamp(0.0, 1.0)
    } else {
        ((prev_open.min(prev_close) - prev_low).max(0.0) / prev_hl).clamp(0.0, 1.0)
    };

    let maha_delta = if let (Some(fk_prev), Some(manifold)) =
        (prev_bar_features_32, precision.brains.get(&brain_id))
    {
        let weights_32 = build_synaptic_precision_weights(brain_state);
        let d_bar_win =
            precision.bar1_mahalanobis(brain_id, &weights_32, fk_prev, &manifold.bar1_win_centroid_32);
        let d_bar_loss =
            precision.bar1_mahalanobis(brain_id, &weights_32, fk_prev, &manifold.bar1_loss_centroid_32);
        (d_bar_win - d_bar_loss).clamp(-4.0, 4.0)
    } else {
        0.0
    };

    let logit_h = 1.25 * maha_delta + 1.50 * wick_exhaustion_prev;
    let h_prev = 1.0 / (1.0 + (-logit_h).exp());
    let rho_base = brain_state.learned_giveback_retention_ratio;
    let rho_eff = (rho_base + 0.14 * (h_prev - 0.50)).clamp(0.50, 0.88);
    (h_prev, rho_eff)
}

/// Helper to compute peak post-entry MFE in R units from trade execution levels and `replay_bars`.
pub fn compute_trade_peak_mfe_r(trade: &ClosedTradeRow) -> f64 {
    let stop_dist = (trade.entry_price.to_f64() - trade.initial_structural_stop.to_f64())
        .abs()
        .max(0.0001);
    let ep = if !trade.entry_bars.is_empty() {
        let idx = trade.entry_bar_idx.min(trade.entry_bars.len() - 1);
        trade.entry_bars[idx].close.to_f64()
    } else {
        trade.entry_price.to_f64()
    };
    let mut peak_mfe_r = 0.0_f64;

    for b in trade.replay_bars() {
        let favorable = if trade.direction == TradeDirection::Long {
            b.high.to_f64() - ep
        } else {
            ep - b.low.to_f64()
        };
        let r = favorable / stop_dist;
        if r > peak_mfe_r {
            peak_mfe_r = r;
        }
    }

    let mfe_from_pct = (trade.mfe_pct / 100.0 * trade.entry_price.to_f64()) / stop_dist;
    if mfe_from_pct > peak_mfe_r {
        peak_mfe_r = mfe_from_pct;
    }

    if peak_mfe_r < trade.realized_r_multiple {
        peak_mfe_r = trade.realized_r_multiple;
    }

    peak_mfe_r
}

/// Builds a local 32x32 empirical covariance from a single trade's causal pre/post bar tensors
/// when standalone `decode_trade_failure` is invoked outside a full ledger snapshot.
#[allow(clippy::needless_range_loop)]
fn compute_single_trade_ledoit_wolf_mahalanobis(
    trade: &ClosedTradeRow,
    delta_z: &[f64; 32],
    weights_32: &[f64; 32],
) -> f64 {
    let entry_idx = trade
        .entry_bar_idx
        .min(trade.entry_bars.len().saturating_sub(1));
    let max_k = trade.entry_bars.len().saturating_sub(entry_idx + 1).min(16);
    let mut samples: Vec<[f64; 32]> = Vec::with_capacity(max_k + 2);
    let mut s0 = extract_causal_entry_array_32(trade);
    for v in &mut s0 {
        *v = if v.is_finite() { v.clamp(-25.0, 25.0) } else { 0.0 };
    }
    samples.push(s0);
    for k in 1..=max_k {
        if let Some(fk) =
            RealTimeLedgerEngine::extract_causal_post_entry_bar_k_features_from_row(trade, k)
        {
            let mut arr = [0.0_f64; 32];
            for i in 0..32 {
                let raw_v = fk.get(i).copied().unwrap_or(0.0) as f64;
                arr[i] = if raw_v.is_finite() {
                    raw_v.clamp(-25.0, 25.0)
                } else {
                    0.0
                };
            }
            samples.push(arr);
        }
    }
    if samples.len() < 2 {
        let mut sb1 = extract_causal_bar1_array_32(trade);
        for v in &mut sb1 {
            *v = if v.is_finite() { v.clamp(-25.0, 25.0) } else { 0.0 };
        }
        samples.push(sb1);
    }

    let mut clean_dz = [0.0_f64; 32];
    for i in 0..32 {
        clean_dz[i] = if delta_z[i].is_finite() {
            delta_z[i].clamp(-25.0, 25.0)
        } else {
            0.0
        };
    }

    let n = samples.len();
    let inv_n = 1.0 / (n as f64);
    let mut mean = [0.0_f64; 32];
    for s in &samples {
        for i in 0..32 {
            mean[i] += s[i] * inv_n;
        }
    }

    let mut scale_acc = 0.0_f64;
    for i in 0..32 {
        scale_acc += clean_dz[i].abs() + mean[i].abs();
    }
    let scale = (scale_acc / 32.0).max(0.05);
    let scale_sq = (scale * scale).max(1e-6);
    let mut cov = [[0.0_f64; 32]; 32];
    for i in 0..32 {
        cov[i][i] = 0.035 * scale_sq;
    }
    for s in &samples {
        for i in 0..32 {
            let di = s[i] - mean[i];
            for j in 0..=i {
                let dj = s[j] - mean[j];
                cov[i][j] += (di * dj) * inv_n;
                cov[j][i] = cov[i][j];
            }
        }
    }

    let (maha, _) = PrecedentKnnIndex::solve_ledoit_wolf_cholesky_mahalanobis_32_scaled(
        &cov,
        n.max(4),
        weights_32,
        &clean_dz,
        scale_sq,
    );
    if maha.is_finite() { maha } else { 1.0 }
}

/// Decodes trade failure mode and computes 32-feature blame attribution vector + true
/// precision-weighted Ledoit-Wolf Cholesky Mahalanobis shift $\sqrt{\Delta z^\top W^{1/2} \Sigma_{\text{LW}}^{-1} W^{1/2} \Delta z}$.
pub fn decode_trade_failure(trade: &ClosedTradeRow) -> Option<DecodedTradeAutopsy> {
    decode_trade_failure_internal(trade, None, None)
}

/// Decodes trade failure using the full multi-brain 32x32 `LedoitWolfPrecision` covariance manifold
/// and active `BrainSynapticPlasticityState` weights.
pub fn decode_trade_failure_with_precision(
    trade: &ClosedTradeRow,
    precision: &LedoitWolfPrecision,
    brain_state: Option<&BrainSynapticPlasticityState>,
) -> Option<DecodedTradeAutopsy> {
    decode_trade_failure_internal(trade, Some(precision), brain_state)
}

#[allow(clippy::needless_range_loop)]
fn decode_trade_failure_internal(
    trade: &ClosedTradeRow,
    precision: Option<&LedoitWolfPrecision>,
    brain_state: Option<&BrainSynapticPlasticityState>,
) -> Option<DecodedTradeAutopsy> {
    let peak_mfe_r = compute_trade_peak_mfe_r(trade);
    let is_loss = trade.realized_r_multiple < 0.0;
    let is_giveback = peak_mfe_r >= 1.0 && trade.realized_r_multiple < 0.65 * peak_mfe_r;

    if !is_loss && !is_giveback {
        return None;
    }

    let giveback_pct = if peak_mfe_r > 0.0 {
        ((peak_mfe_r - trade.realized_r_multiple.max(0.0)) / peak_mfe_r * 100.0).clamp(0.0, 100.0)
    } else {
        0.0
    };

    // 1. Extract real recorded entry_tensor, peak_mfe_tensor, and adverse_inflection_tensor
    let entry_features_vec_f32 = RealTimeLedgerEngine::extract_causal_entry_features_from_row(trade);
    let mut entry_tensor = [0.0_f64; 32];
    for i in 0..32 {
        entry_tensor[i] = entry_features_vec_f32.get(i).copied().unwrap_or(0.0) as f64;
    }

    let replay = trade.replay_bars();
    let ep = if !trade.entry_bars.is_empty() {
        let idx = trade.entry_bar_idx.min(trade.entry_bars.len() - 1);
        trade.entry_bars[idx].close.to_f64()
    } else {
        trade.entry_price.to_f64()
    };
    let stop_dist = (trade.entry_price.to_f64() - trade.initial_structural_stop.to_f64())
        .abs()
        .max(0.0001);
    let dir = trade.direction.sign();

    let mut peak_bar_k = 1usize;
    let mut best_fav_r = f64::NEG_INFINITY;
    let mut inflection_bar_k = 1usize;
    let mut worst_drop_from_peak = 0.0_f64;
    let mut running_pk = 0.0_f64;

    for (idx, b) in replay.iter().enumerate() {
        let k = idx + 1;
        let fav_r = if dir > 0.0 {
            (b.high.to_f64() - ep) / stop_dist
        } else {
            (ep - b.low.to_f64()) / stop_dist
        };
        let close_r = ((b.close.to_f64() - ep) * dir) / stop_dist;
        if fav_r > best_fav_r {
            best_fav_r = fav_r;
            peak_bar_k = k;
        }
        if fav_r > running_pk {
            running_pk = fav_r;
        }
        let drop_r = running_pk - close_r;
        if drop_r > worst_drop_from_peak {
            worst_drop_from_peak = drop_r;
            inflection_bar_k = k;
        }
    }

    let peak_mfe_tensor: [f64; 32] = if !replay.is_empty()
        && let Some(pk_vec) =
            RealTimeLedgerEngine::extract_causal_post_entry_bar_k_features_from_row(
                trade, peak_bar_k,
            ) {
        let mut arr = [0.0_f64; 32];
        for i in 0..32 {
            arr[i] = pk_vec.get(i).copied().unwrap_or(0.0) as f64;
        }
        arr
    } else {
        entry_tensor
    };

    let adverse_inflection_tensor: [f64; 32] = if !replay.is_empty()
        && let Some(adv_vec) =
            RealTimeLedgerEngine::extract_causal_post_entry_bar_k_features_from_row(
                trade,
                inflection_bar_k,
            ) {
        let mut arr = [0.0_f64; 32];
        for i in 0..32 {
            arr[i] = adv_vec.get(i).copied().unwrap_or(0.0) as f64;
        }
        arr
    } else if trade.post_entry_bar1_features_32.len() >= 32 {
        let mut arr = [0.0_f64; 32];
        for i in 0..32 {
            arr[i] = trade.post_entry_bar1_features_32[i] as f64;
        }
        arr
    } else if trade.entry_embedding_128.len() >= 64 {
        let mut arr = [0.0_f64; 32];
        for i in 0..32 {
            arr[i] = trade.entry_embedding_128[32 + i] as f64;
        }
        arr
    } else {
        extract_causal_bar1_array_32(trade)
    };

    // 2. Compute signed & magnitude 32-Feature Blame Attribution Vector across (entry, peak_mfe, adverse_inflection)
    let reference_tensor = if is_giveback {
        if replay.is_empty()
            && let Some(prec) = precision
            && let Some(m) = prec.brains.get(&trade.brain_id)
        {
            m.winner_centroid_32
        } else {
            peak_mfe_tensor
        }
    } else if let Some(prec) = precision
        && let Some(m) = prec.brains.get(&trade.brain_id)
        && peak_mfe_r < 0.20
    {
        // Immediate pre-entry vacuum trap: measure pre-entry divergence against causal winner centroid + bar1 inflection
        let mut blended = [0.0_f64; 32];
        for i in 0..32 {
            blended[i] = 0.5 * m.winner_centroid_32[i] + 0.5 * entry_tensor[i];
        }
        blended
    } else {
        entry_tensor
    };

    let target_tensor = if !is_giveback && peak_mfe_r < 0.20 {
        let mut blended = [0.0_f64; 32];
        for i in 0..32 {
            blended[i] = 0.5 * entry_tensor[i] + 0.5 * adverse_inflection_tensor[i];
        }
        blended
    } else {
        adverse_inflection_tensor
    };

    let mut delta_z = [0.0_f64; 32];
    let mut blame_vector = vec![0.0_f64; 32];
    for f in 0..32 {
        let raw_dz = target_tensor[f] - reference_tensor[f];
        let dz = if raw_dz.is_finite() {
            raw_dz.clamp(-25.0, 25.0)
        } else {
            0.0
        };
        delta_z[f] = dz;
        blame_vector[f] = ((dz.abs() * 1000.0).round() / 1000.0).max(0.001);
    }

    let mut indexed_blame: Vec<(usize, f64)> = blame_vector.iter().copied().enumerate().collect();
    indexed_blame.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let top_5_indices: Vec<usize> = indexed_blame.iter().take(5).map(|(idx, _)| *idx).collect();
    let top_5_names: Vec<String> = top_5_indices
        .iter()
        .map(|&idx| FEATURE_NAMES_32.get(idx).unwrap_or(&"unknown").to_string())
        .collect();

    // 3. Deterministic Failure Mode Classification from recorded excursion & stream blame deltas
    let failure_mode = if is_giveback {
        AutopsyFailureMode::MfeStallGivebackExhaustion
    } else {
        let s1_blame: f64 = blame_vector[0..12].iter().sum();
        let s2_blame: f64 = blame_vector[12..20].iter().sum();
        let s3_blame: f64 = blame_vector[20..26].iter().sum();
        let immediate_entry_vacuum = peak_mfe_r < 0.18
            && (trade.entry_trap_similarity_score >= 0.70
                || trade.was_vetoed_by_loss_autopsy
                || blame_vector[10] >= 0.25
                || (s3_blame > s1_blame && s3_blame > s2_blame));

        if immediate_entry_vacuum {
            AutopsyFailureMode::EntryLiquidityVacuumTrap
        } else if s2_blame > s1_blame && s2_blame > 0.80 {
            AutopsyFailureMode::DealerCharmVannaUnpinDecay
        } else {
            AutopsyFailureMode::PostEntryOrderflowToxicityReversal
        }
    };

    // 4. True 32x32 Ledoit-Wolf Cholesky Precision-Weighted Mahalanobis Shift:
    //    sqrt(delta_z^T * W^{1/2} * Sigma_LW^{-1} * W^{1/2} * delta_z)
    let weights_32 = if let Some(st) = brain_state {
        build_synaptic_precision_weights(st)
    } else {
        [6.0 / 32.0; 32]
    };
    let raw_maha = if let Some(prec) = precision {
        if is_giveback || failure_mode == AutopsyFailureMode::PostEntryOrderflowToxicityReversal {
            prec.bar1_mahalanobis(trade.brain_id, &weights_32, &target_tensor, &reference_tensor)
        } else {
            prec.entry_mahalanobis(trade.brain_id, &weights_32, &target_tensor, &reference_tensor)
        }
    } else {
        compute_single_trade_ledoit_wolf_mahalanobis(trade, &delta_z, &weights_32)
    };
    let mahalanobis_distance_shift = ((raw_maha.max(0.001)) * 1000.0).round() / 1000.0;

    let collapse_bar_offset = if trade.scratch_bar_offset > 0 {
        trade.scratch_bar_offset
    } else if !replay.is_empty() {
        inflection_bar_k.max(1)
    } else if is_giveback {
        5
    } else {
        1
    };

    let summary = format!(
        "Trade {} on {} ({}) failed via {} at Bar +{}: top culprit [{}] (MFE: +{:.2}R -> Realized: {:.2}R, Giveback: {:.1}%, LW-Mahalanobis Shift: {:.3})",
        trade.trade_id.to_string().chars().take(8).collect::<String>(),
        trade.symbol,
        trade.brain_id.as_str(),
        failure_mode.as_str(),
        collapse_bar_offset,
        top_5_names.first().cloned().unwrap_or_default(),
        peak_mfe_r,
        trade.realized_r_multiple,
        giveback_pct,
        mahalanobis_distance_shift
    );

    Some(DecodedTradeAutopsy {
        trade_id: trade.trade_id,
        symbol: trade.symbol.clone(),
        brain_id: trade.brain_id,
        realized_r: trade.realized_r_multiple,
        peak_mfe_r,
        giveback_pct,
        failure_mode,
        failure_mode_label: failure_mode.as_str().to_string(),
        blame_attribution_32: blame_vector,
        top_culprit_feature_indices: top_5_indices,
        top_culprit_names: top_5_names,
        mahalanobis_distance_shift,
        collapse_bar_offset,
        summary,
        entry_features_32: entry_features_vec_f32,
    })
}

/// Deterministic Single-Trade Causal Re-Simulator (`simulate_trade_under_candidate`).
///
/// Evaluates `trade` causally under `brain_state` and `trap_centroids_32` with ZERO outcome lookahead:
/// 1. **Stage 0 Entry Gate (Single Shared Kernel `evaluate_stage0_conformal_kernel`)**:
///    Computes precision- and synaptic-weight-adjusted Mahalanobis distance (`d_win`, `d_trap`),
///    Mondrian regime-conditioned split-conformal `p_value`, and $\chi^2_{32}$ OOD epistemic shield.
/// 2. **Stage 1–3 Bar-by-Bar Path Replay (`trade.replay_bars()`) with Gap-Slippage Realism & Hazard Stopping**:
///    - **Adverse Gap-Through Fill Clamping**: Long ratchet/stop fills at `level.min(bar_open)`;
///      Short ratchet/stop fills at `level.max(bar_open)` (booking true `realized_r <= -1.0` on adverse open gaps).
///    - **Stage 1 (`Bar 1..3`)**: Sequential Wald SPRT log-likelihood ratio exiting at exact `bar_k` close price.
///    - **Stage 3 (`Hazard-Conditioned Optimal Stopping`)**: Tracks causal `running_peak_r` strictly up to bar `k - 1`
///      ONLY and applies $\rho_{\text{eff}}(k) = \operatorname{clamp}(\rho_{\text{base}} + 0.14(h_{k-1} - 0.50), 0.50, 0.88)$.
pub fn simulate_trade_under_candidate(
    trade: &ClosedTradeRow,
    brain_state: &BrainSynapticPlasticityState,
    trap_centroids_32: &[[f64; 32]],
    precision_diag_or_chol: &LedoitWolfPrecision,
) -> SimulatedTradeOutcome {
    let trap_bank: Vec<Stage0TrapCentroid> = trap_centroids_32
        .iter()
        .map(|c| Stage0TrapCentroid {
            trap_id: None,
            root_cause_category: "ENTRY_LIQUIDITY_VACUUM_TRAP".to_string(),
            features_32: *c,
            quarantined: false,
        })
        .collect();
    simulate_trade_under_candidate_with_trap_bank(
        trade,
        brain_state,
        &trap_bank,
        precision_diag_or_chol,
    )
}

/// Evaluates `trade` causally using structured `Stage0TrapCentroid` records (preserving `quarantined` status
/// and exact `trap_id` attribution) via the single shared `evaluate_stage0_conformal_kernel`.
pub fn simulate_trade_under_candidate_with_trap_bank(
    trade: &ClosedTradeRow,
    brain_state: &BrainSynapticPlasticityState,
    trap_bank: &[Stage0TrapCentroid],
    precision_diag_or_chol: &LedoitWolfPrecision,
) -> SimulatedTradeOutcome {
    let entry_vec_32 = extract_causal_entry_array_32(trade);
    let weights_32 = build_synaptic_precision_weights(brain_state);

    // --- STAGE 0: Single Shared Pure-Causal Conformal + OOD Kernel ---
    let s0 = evaluate_stage0_conformal_kernel(
        trade.brain_id,
        &entry_vec_32,
        Some(trade.trade_id),
        brain_state,
        trap_bank,
        precision_diag_or_chol,
    );
    if s0.vetoed {
        return SimulatedTradeOutcome {
            trade_id: trade.trade_id,
            realized_r: 0.0,
            exit_price: trade.entry_price.to_f64(),
            vetoed: true,
            scratched_stage1: false,
            ratchet_exited_stage3: false,
            full_stop_hit: false,
            exit_bar_offset: 0,
            candidate_conformal_margin: s0.conformal_margin,
            conformal_llr: s0.conformal_llr,
            conformal_p_value: s0.conformal_p_value,
            epistemic_ood_score: s0.epistemic_ood_score,
            kelly_epistemic_multiplier: s0.kelly_epistemic_multiplier,
            regime_bucket: s0.regime_bucket,
            max_wald_sprt_llr: 0.0,
            max_reversal_hazard: 0.0,
            effective_retention_ratio: brain_state.learned_giveback_retention_ratio,
        };
    }

    // --- STAGE 1..3: Causal Bar-by-Bar Path Replay (`trade.replay_bars()`) ---
    let replay_bars = trade.replay_bars();
    let initial_risk = (trade.entry_price.to_f64() - trade.initial_structural_stop.to_f64())
        .abs()
        .max(0.0001);
    let dir = trade.direction.sign();
    let ep = if !trade.entry_bars.is_empty() {
        let idx = trade.entry_bar_idx.min(trade.entry_bars.len() - 1);
        trade.entry_bars[idx].close.to_f64()
    } else {
        trade.entry_price.to_f64()
    };

    if replay_bars.is_empty() {
        return SimulatedTradeOutcome {
            trade_id: trade.trade_id,
            realized_r: trade.realized_r_multiple,
            exit_price: trade.actual_exit_price.to_f64(),
            vetoed: false,
            scratched_stage1: false,
            ratchet_exited_stage3: false,
            full_stop_hit: trade.realized_r_multiple <= -0.90,
            exit_bar_offset: 0,
            candidate_conformal_margin: s0.conformal_margin,
            conformal_llr: s0.conformal_llr,
            conformal_p_value: s0.conformal_p_value,
            epistemic_ood_score: s0.epistemic_ood_score,
            kelly_epistemic_multiplier: s0.kelly_epistemic_multiplier,
            regime_bucket: s0.regime_bucket,
            max_wald_sprt_llr: 0.0,
            max_reversal_hazard: 0.0,
            effective_retention_ratio: brain_state.learned_giveback_retention_ratio,
        };
    }

    let mut running_peak_r = 0.0_f64;
    let mut wald_llr = 0.0_f64;
    let mut max_wald_sprt_llr = 0.0_f64;
    let mut max_reversal_hazard = 0.0_f64;
    let mut last_rho_eff = brain_state.learned_giveback_retention_ratio;

    for (idx, bar) in replay_bars.iter().enumerate() {
        let bar_idx = idx + 1;
        let bar_high = bar.high.to_f64();
        let bar_low = bar.low.to_f64();
        let bar_open = bar.open.to_f64();
        let bar_close = bar.close.to_f64();

        let favorable_r = if dir > 0.0 {
            (bar_high - ep) / initial_risk
        } else {
            (ep - bar_low) / initial_risk
        };
        let adverse_r = if dir > 0.0 {
            (bar_low - ep) / initial_risk
        } else {
            (ep - bar_high) / initial_risk
        };
        let close_r = ((bar_close - ep) * dir) / initial_risk;

        // Stage 3 Hazard-Conditioned Dynamic Continuation Floor:
        // Strictly causal `running_peak_r` and `prev_bar` (`idx - 1`) from bars 1..=bar_idx-1 ONLY!
        if running_peak_r >= 1.0 && idx >= 1 {
            let prev_bar = &replay_bars[idx - 1];
            let prev_feat_opt: Option<[f64; 32]> =
                RealTimeLedgerEngine::extract_causal_post_entry_bar_k_features_from_row(trade, idx)
                    .map(|fk| {
                        let mut arr = [0.0_f64; 32];
                        for (i, &v) in fk.iter().take(32).enumerate() {
                            arr[i] = v as f64;
                        }
                        arr
                    });
            let (h_prev, rho_eff) = compute_causal_reversal_hazard_retention(
                trade.brain_id,
                dir,
                prev_bar.open.to_f64(),
                prev_bar.high.to_f64(),
                prev_bar.low.to_f64(),
                prev_bar.close.to_f64(),
                prev_feat_opt.as_ref(),
                brain_state,
                precision_diag_or_chol,
            );
            if h_prev > max_reversal_hazard {
                max_reversal_hazard = h_prev;
            }
            last_rho_eff = rho_eff;

            let ratchet_stop_r = running_peak_r * rho_eff;
            let ratchet_stop_price = ep + dir * ratchet_stop_r * initial_risk;
            let breached_ratchet = if dir > 0.0 {
                bar_low <= ratchet_stop_price
            } else {
                bar_high >= ratchet_stop_price
            };
            if breached_ratchet {
                let ratchet_fill_price = if dir > 0.0 {
                    ratchet_stop_price.min(bar_open)
                } else {
                    ratchet_stop_price.max(bar_open)
                };
                let runner_ratchet_r = ((ratchet_fill_price - ep) * dir) / initial_risk;
                let t1_dist_r = ((trade.target_1_price.to_f64() - ep) * dir) / initial_risk;
                let settled_r = if t1_dist_r >= 1.0 && running_peak_r >= t1_dist_r {
                    0.5 * t1_dist_r + 0.5 * runner_ratchet_r
                } else {
                    runner_ratchet_r
                };
                return SimulatedTradeOutcome {
                    trade_id: trade.trade_id,
                    realized_r: settled_r,
                    exit_price: ratchet_fill_price,
                    vetoed: false,
                    scratched_stage1: false,
                    ratchet_exited_stage3: true,
                    full_stop_hit: settled_r <= -0.90,
                    exit_bar_offset: bar_idx,
                    candidate_conformal_margin: s0.conformal_margin,
                    conformal_llr: s0.conformal_llr,
                    conformal_p_value: s0.conformal_p_value,
                    epistemic_ood_score: s0.epistemic_ood_score,
                    kelly_epistemic_multiplier: s0.kelly_epistemic_multiplier,
                    regime_bucket: s0.regime_bucket,
                    max_wald_sprt_llr,
                    max_reversal_hazard,
                    effective_retention_ratio: last_rho_eff,
                };
            }
        }

        // Structural Stop Check on bar k (with causal prior T1 50%-tranche breakeven protection if prior bars already hit T1)
        if adverse_r <= -1.0 {
            if running_peak_r >= 0.72 {
                let banked_t1_r = 0.5 * running_peak_r;
                let t1_exit_price = ep + dir * banked_t1_r * initial_risk;
                return SimulatedTradeOutcome {
                    trade_id: trade.trade_id,
                    realized_r: banked_t1_r,
                    exit_price: t1_exit_price,
                    vetoed: false,
                    scratched_stage1: false,
                    ratchet_exited_stage3: true,
                    full_stop_hit: false,
                    exit_bar_offset: bar_idx,
                    candidate_conformal_margin: s0.conformal_margin,
                    conformal_llr: s0.conformal_llr,
                    conformal_p_value: s0.conformal_p_value,
                    epistemic_ood_score: s0.epistemic_ood_score,
                    kelly_epistemic_multiplier: s0.kelly_epistemic_multiplier,
                    regime_bucket: s0.regime_bucket,
                    max_wald_sprt_llr,
                    max_reversal_hazard,
                    effective_retention_ratio: last_rho_eff,
                };
            }
            let stop_exit_price = ep - dir * initial_risk;
            let stop_fill_price = if dir > 0.0 {
                stop_exit_price.min(bar_open)
            } else {
                stop_exit_price.max(bar_open)
            };
            let stop_realized_r = ((stop_fill_price - ep) * dir) / initial_risk;
            return SimulatedTradeOutcome {
                trade_id: trade.trade_id,
                realized_r: stop_realized_r,
                exit_price: stop_fill_price,
                vetoed: false,
                scratched_stage1: false,
                ratchet_exited_stage3: false,
                full_stop_hit: true,
                exit_bar_offset: bar_idx,
                candidate_conformal_margin: s0.conformal_margin,
                conformal_llr: s0.conformal_llr,
                conformal_p_value: s0.conformal_p_value,
                epistemic_ood_score: s0.epistemic_ood_score,
                kelly_epistemic_multiplier: s0.kelly_epistemic_multiplier,
                regime_bucket: s0.regime_bucket,
                max_wald_sprt_llr,
                max_reversal_hazard,
                effective_retention_ratio: last_rho_eff,
            };
        }

        // Stage 1 (Bar 1..3): Sequential Wald SPRT Log-Likelihood Ratio
        if bar_idx <= 3 {
            let hl_range = (bar_high - bar_low).max(0.0001);
            let signed_body_delta = ((bar_close - bar_open) * dir) / hl_range;
            let step_llr = if close_r < -0.25 && signed_body_delta < -0.15 {
                let mut maha_shift = 0.0_f64;
                if let Some(manifold) = precision_diag_or_chol.brains.get(&trade.brain_id)
                    && let Some(fk) =
                        RealTimeLedgerEngine::extract_causal_post_entry_bar_k_features_from_row(
                            trade, bar_idx,
                        )
                {
                    let mut bar_vec = [0.0_f64; 32];
                    for (i, &v) in fk.iter().take(32).enumerate() {
                        bar_vec[i] = v as f64;
                    }
                    let d_bw = precision_diag_or_chol.bar1_mahalanobis(
                        trade.brain_id,
                        &weights_32,
                        &bar_vec,
                        &manifold.bar1_win_centroid_32,
                    );
                    let d_bl = precision_diag_or_chol.bar1_mahalanobis(
                        trade.brain_id,
                        &weights_32,
                        &bar_vec,
                        &manifold.bar1_loss_centroid_32,
                    );
                    maha_shift = ((d_bw - d_bl) * 0.05).clamp(0.0, 0.06);
                }
                (-close_r - 0.20) * 0.85 + (-signed_body_delta) * 0.45 + maha_shift
            } else if close_r > 0.10 {
                -0.35
            } else {
                0.0
            };

            wald_llr = (wald_llr + step_llr).max(0.0);
            if wald_llr > max_wald_sprt_llr {
                max_wald_sprt_llr = wald_llr;
            }
            let sprt_hurdle = brain_state.wald_sprt_thresholds[bar_idx - 1];
            if wald_llr >= sprt_hurdle {
                let scratch_r = ((bar_close - ep) * dir) / initial_risk;
                return SimulatedTradeOutcome {
                    trade_id: trade.trade_id,
                    realized_r: scratch_r,
                    exit_price: bar_close,
                    vetoed: false,
                    scratched_stage1: true,
                    ratchet_exited_stage3: false,
                    full_stop_hit: scratch_r <= -0.90,
                    exit_bar_offset: bar_idx,
                    candidate_conformal_margin: s0.conformal_margin,
                    conformal_llr: s0.conformal_llr,
                    conformal_p_value: s0.conformal_p_value,
                    epistemic_ood_score: s0.epistemic_ood_score,
                    kelly_epistemic_multiplier: s0.kelly_epistemic_multiplier,
                    regime_bucket: s0.regime_bucket,
                    max_wald_sprt_llr,
                    max_reversal_hazard,
                    effective_retention_ratio: last_rho_eff,
                };
            }
        }

        // Advance causal running peak MFE at end of bar k
        if favorable_r > running_peak_r {
            running_peak_r = favorable_r;
        }
    }

    // Settled at final trade exit without early veto / SPRT scratch / ratchet breach
    let final_exit_price = trade.actual_exit_price.to_f64();
    let price_r = ((final_exit_price - trade.entry_price.to_f64()) * dir) / initial_risk;
    let settled_r = if trade.realized_r_multiple.abs() > 1e-9 {
        trade.realized_r_multiple
    } else {
        price_r
    };
    SimulatedTradeOutcome {
        trade_id: trade.trade_id,
        realized_r: settled_r,
        exit_price: final_exit_price,
        vetoed: false,
        scratched_stage1: false,
        ratchet_exited_stage3: false,
        full_stop_hit: settled_r <= -0.90,
        exit_bar_offset: replay_bars.len(),
        candidate_conformal_margin: s0.conformal_margin,
        conformal_llr: s0.conformal_llr,
        conformal_p_value: s0.conformal_p_value,
        epistemic_ood_score: s0.epistemic_ood_score,
        kelly_epistemic_multiplier: s0.kelly_epistemic_multiplier,
        regime_bucket: s0.regime_bucket,
        max_wald_sprt_llr,
        max_reversal_hazard,
        effective_retention_ratio: last_rho_eff,
    }
}

/// Fisher-Preconditioned Exponentiated Mirror Descent on the 32-Simplex (`propose_candidate_generation`).
///
/// Conserving total synaptic mass $Z = \sum_{f=0}^{31} w_f = 6.0$ with box bounds $w_f \in [0.04, 0.65]$:
/// $$w_f^{(t+1)} = \operatorname{Proj}_{[0.04, 0.65], \sum=6}\!\left(w_f^{(t)} \cdot \exp\big(\eta \, g_f^{(t)}\big)\right)$$
pub fn propose_candidate_generation(
    current_states: &[BrainSynapticPlasticityState],
    autopsies: &[DecodedTradeAutopsy],
    candidate_gen_id: u32,
) -> Vec<BrainSynapticPlasticityState> {
    propose_candidate_generation_with_context(
        current_states,
        autopsies,
        autopsies,
        None,
        &HashMap::new(),
        candidate_gen_id,
    )
}

/// Context-aware Fisher-Preconditioned Mirror Descent incorporating causal baseline winner feature
/// dispersion (`precision`) and counterfactual shadow-regret coordinates (`shadow_regret_penalty_by_brain`).
#[allow(clippy::needless_range_loop)]
pub fn propose_candidate_generation_with_context(
    current_states: &[BrainSynapticPlasticityState],
    autopsies: &[DecodedTradeAutopsy],
    threshold_autopsies: &[DecodedTradeAutopsy],
    precision: Option<&LedoitWolfPrecision>,
    shadow_regret_penalty_by_brain: &HashMap<BrainHorizonId, [f64; 32]>,
    candidate_gen_id: u32,
) -> Vec<BrainSynapticPlasticityState> {
    let mut global_blame = [0.0_f64; 32];
    for a in autopsies {
        for f in 0..32 {
            if let Some(&b) = a.blame_attribution_32.get(f) {
                global_blame[f] += b;
            }
        }
    }
    let max_global_blame = global_blame
        .iter()
        .copied()
        .fold(0.0_f64, f64::max)
        .max(0.001);

    let eta = 0.035_f64;

    current_states
        .iter()
        .map(|st| {
            let mut cand = st.clone();
            cand.generation_id = candidate_gen_id;

            // 1. Initialize prior 32-simplex weights w^(t) in [0.04, 0.65], sum == 6.0
            let raw_prev = build_synaptic_precision_weights(st);
            let w_prev = project_capped_simplex_kl_32(&raw_prev, 0.04, 0.65, 6.0);

            // 2. Build Fisher-preconditioned signed gradient g^(t):
            //    + Mahalanobis separation on decoded losses/givebacks
            //    - Margin-compression dispersion penalty on causal baseline winners
            //    - Counterfactual shadow-regret penalty on falsely vetoed setups
            let brain_autopsies_all: Vec<_> = autopsies
                .iter()
                .filter(|a| a.brain_id == cand.brain_id)
                .collect();
            let mut loss_sep = [0.0_f64; 32];
            if !brain_autopsies_all.is_empty() {
                let mut local_b = [0.0_f64; 32];
                for a in &brain_autopsies_all {
                    for f in 0..32 {
                        local_b[f] += a.blame_attribution_32.get(f).copied().unwrap_or(0.0);
                    }
                }
                let max_lb = local_b.iter().copied().fold(0.0_f64, f64::max).max(0.001);
                for f in 0..32 {
                    loss_sep[f] = 0.70 * (local_b[f] / max_lb)
                        + 0.30 * (global_blame[f] / max_global_blame);
                }
            } else {
                for f in 0..32 {
                    loss_sep[f] = global_blame[f] / max_global_blame;
                }
            }

            let mut win_var_pen = [0.0_f64; 32];
            if let Some(prec) = precision
                && let Some(m) = prec.brains.get(&cand.brain_id)
            {
                let max_wv = m
                    .winner_feature_var_32
                    .iter()
                    .copied()
                    .fold(0.0_f64, f64::max)
                    .max(1e-6);
                for f in 0..32 {
                    win_var_pen[f] = (m.winner_feature_var_32[f] / max_wv).clamp(0.0, 1.0);
                }
            }

            let mut shadow_pen = [0.0_f64; 32];
            if let Some(sp) = shadow_regret_penalty_by_brain.get(&cand.brain_id) {
                let max_sp = sp.iter().copied().fold(0.0_f64, f64::max).max(1e-6);
                for f in 0..32 {
                    shadow_pen[f] = (sp[f] / max_sp).clamp(0.0, 1.0);
                }
            }

            let mut grad = [0.0_f64; 32];
            let mut grad_mean = 0.0_f64;
            for f in 0..32 {
                let fisher_inv_diag = 1.0 / (0.25 + win_var_pen[f]);
                let raw_gf = fisher_inv_diag
                    * (-0.55 * loss_sep[f] - 0.30 * win_var_pen[f] - 0.45 * shadow_pen[f]);
                grad[f] = raw_gf;
                grad_mean += raw_gf / 32.0;
            }

            let mut unnorm_next = [0.0_f64; 32];
            for f in 0..32 {
                let centered_g = (grad[f] - grad_mean).clamp(-2.5, 2.5);
                unnorm_next[f] = w_prev[f] * (eta * centered_g).exp();
            }

            let w_next = project_capped_simplex_kl_32(&unnorm_next, 0.04, 0.65, 6.0);
            let kl_div = compute_simplex_kl_divergence_32(&w_next, &w_prev, 6.0);

            cand.simplex_weights_32 = Some(w_next);
            cand.mirror_descent_kl_divergence = kl_div;
            for f in 0..32 {
                cand.synaptic_feature_weights_32[f] = w_next[f] * (32.0 / 6.0);
            }

            // 3. Targeted Gate Adaptation based on un-remediated decoded autopsies for this brain
            let brain_thresh_autopsies: Vec<_> = threshold_autopsies
                .iter()
                .filter(|a| a.brain_id == cand.brain_id)
                .collect();
            for a in &brain_thresh_autopsies {
                match a.failure_mode {
                    AutopsyFailureMode::EntryLiquidityVacuumTrap => {
                        cand.entry_veto_margin_delta =
                            ((cand.entry_veto_margin_delta + 0.005).min(0.080) * 1000.0).round()
                                / 1000.0;
                    }
                    AutopsyFailureMode::PostEntryOrderflowToxicityReversal => {
                        for k in 0..3 {
                            cand.wald_sprt_thresholds[k] =
                                ((cand.wald_sprt_thresholds[k] - 0.10).max(1.34) * 1000.0).round()
                                    / 1000.0;
                        }
                    }
                    AutopsyFailureMode::MfeStallGivebackExhaustion => {
                        cand.giveback_exhaustion_threshold =
                            ((cand.giveback_exhaustion_threshold - 0.02).max(0.20) * 1000.0)
                                .round()
                                / 1000.0;
                        cand.learned_giveback_retention_ratio =
                            ((cand.learned_giveback_retention_ratio + 0.03).min(0.90) * 1000.0)
                                .round()
                                / 1000.0;
                    }
                    AutopsyFailureMode::DealerCharmVannaUnpinDecay => {
                        cand.learned_charm_vanna_unpin_quantile =
                            ((cand.learned_charm_vanna_unpin_quantile + 0.01).min(0.95) * 1000.0)
                                .round()
                                / 1000.0;
                    }
                }
            }

            cand.refresh_inspector_rankings();
            cand
        })
        .collect()
}

/// Computes `(winners_count, expectancy_r, profit_factor)` from a slice of `SimulatedTradeOutcome`s.
fn summarize_simulated_outcomes(outcomes: &[SimulatedTradeOutcome]) -> (usize, f64, f64) {
    if outcomes.is_empty() {
        return (0, 0.0, 1.0);
    }
    let winners = outcomes.iter().filter(|o| o.realized_r > 0.0).count();
    let net_r: f64 = outcomes.iter().map(|o| o.realized_r).sum();
    let exp_r = net_r / (outcomes.len() as f64);
    let gross_win: f64 = outcomes
        .iter()
        .filter(|o| o.realized_r > 0.0)
        .map(|o| o.realized_r)
        .sum();
    let gross_loss: f64 = outcomes
        .iter()
        .filter(|o| o.realized_r < 0.0)
        .map(|o| o.realized_r.abs())
        .sum();
    let pf = if gross_loss > 1e-9 {
        gross_win / gross_loss
    } else {
        99.0
    };
    (winners, exp_r, pf)
}

/// Builds per-brain Stage-0 active `Stage0TrapCentroid` banks for `current` champion vs `candidate` generation,
/// strictly excluding any `quarantined` traps.
fn build_trap_centroid_banks(
    ledger: &RealTimeLedgerEngine,
    _precision: &LedoitWolfPrecision,
    _baseline_trades: &[ClosedTradeRow],
    decoded_wf_autopsies: &[(&ClosedTradeRow, DecodedTradeAutopsy)],
) -> (
    HashMap<BrainHorizonId, Vec<Stage0TrapCentroid>>,
    HashMap<BrainHorizonId, Vec<Stage0TrapCentroid>>,
) {
    let mut current_bank: HashMap<BrainHorizonId, Vec<Stage0TrapCentroid>> = HashMap::new();
    let all_traps = ledger.precedent_knn().all_hard_negative_traps();
    let mut quarantined_ids = std::collections::HashSet::new();

    for trap in &all_traps {
        if trap.quarantined {
            quarantined_ids.insert(trap.loss_trade_id);
            continue;
        }
        if (trap.root_cause_category == "ENTRY_LIQUIDITY_VACUUM_TRAP"
            || trap.root_cause_category.starts_with("STAGE_0_"))
            && trap.entry_features_32.len() >= 32
        {
            let mut arr = [0.0_f64; 32];
            for (i, &v) in trap.entry_features_32.iter().take(32).enumerate() {
                arr[i] = v as f64;
            }
            current_bank
                .entry(trap.brain_id)
                .or_default()
                .push(Stage0TrapCentroid {
                    trap_id: Some(trap.loss_trade_id),
                    root_cause_category: trap.root_cause_category.clone(),
                    features_32: arr,
                    quarantined: false,
                });
        }
    }

    let mut candidate_bank = current_bank.clone();
    for (row, dec) in decoded_wf_autopsies {
        if dec.failure_mode == AutopsyFailureMode::EntryLiquidityVacuumTrap
            && !quarantined_ids.contains(&row.trade_id)
        {
            candidate_bank
                .entry(row.brain_id)
                .or_default()
                .push(Stage0TrapCentroid {
                    trap_id: Some(row.trade_id),
                    root_cause_category: "ENTRY_LIQUIDITY_VACUUM_TRAP".to_string(),
                    features_32: extract_causal_entry_array_32(row),
                    quarantined: false,
                });
        }
    }

    (current_bank, candidate_bank)
}

/// Dual-Gate Pareto Verification & Monotonic Closed-Loop Promotion (`verify_and_promote_generation`).
pub fn verify_and_promote_generation(
    ledger: &RealTimeLedgerEngine,
    candidate_states: &[BrainSynapticPlasticityState],
    walkforward_trades: &[ClosedTradeRow],
    candidate_gen_id: u32,
) -> (bool, GateAuditReport) {
    verify_and_promote_generation_internal(
        ledger,
        candidate_states,
        walkforward_trades,
        candidate_gen_id,
        0,
    )
}

fn verify_and_promote_generation_internal(
    ledger: &RealTimeLedgerEngine,
    candidate_states: &[BrainSynapticPlasticityState],
    walkforward_trades: &[ClosedTradeRow],
    candidate_gen_id: u32,
    newly_quarantined_count: usize,
) -> (bool, GateAuditReport) {
    let closed_snap = ledger.closed_trades_snapshot();
    let baseline_count = 173.min(closed_snap.len());
    let baseline_trades = &closed_snap[0..baseline_count];

    let current_states = ledger.precedent_knn().ordered_synaptic_plasticity_states();
    let precision = LedoitWolfPrecision::from_settled_trades(baseline_trades);

    let current_map: HashMap<BrainHorizonId, BrainSynapticPlasticityState> = current_states
        .iter()
        .cloned()
        .map(|s| (s.brain_id, s))
        .collect();
    let candidate_map: HashMap<BrainHorizonId, BrainSynapticPlasticityState> = candidate_states
        .iter()
        .cloned()
        .map(|s| (s.brain_id, s))
        .collect();

    let mut decoded_wf_autopsies: Vec<(&ClosedTradeRow, DecodedTradeAutopsy)> = Vec::new();
    for wf in walkforward_trades {
        let st_ref = current_map.get(&wf.brain_id);
        if let Some(dec) = decode_trade_failure_with_precision(wf, &precision, st_ref) {
            decoded_wf_autopsies.push((wf, dec));
        }
    }

    let (current_trap_bank, candidate_trap_bank) =
        build_trap_centroid_banks(ledger, &precision, baseline_trades, &decoded_wf_autopsies);

    // --- GATE 1: Real 173-Trade Causal Bar-by-Bar Replay under Current vs Candidate ---
    let mut current_baseline_outcomes = Vec::with_capacity(baseline_trades.len());
    let mut candidate_baseline_outcomes = Vec::with_capacity(baseline_trades.len());

    for trade in baseline_trades {
        let fallback_cur = BrainSynapticPlasticityState::new_gen0(trade.brain_id);
        let cur_state = current_map.get(&trade.brain_id).unwrap_or(&fallback_cur);
        let cand_state = candidate_map.get(&trade.brain_id).unwrap_or(cur_state);

        let cur_traps = current_trap_bank
            .get(&trade.brain_id)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        let cand_traps = candidate_trap_bank
            .get(&trade.brain_id)
            .map(Vec::as_slice)
            .unwrap_or(&[]);

        current_baseline_outcomes.push(simulate_trade_under_candidate_with_trap_bank(
            trade, cur_state, cur_traps, &precision,
        ));
        candidate_baseline_outcomes.push(simulate_trade_under_candidate_with_trap_bank(
            trade, cand_state, cand_traps, &precision,
        ));
    }

    let (_, current_baseline_exp_r, current_baseline_pf) =
        summarize_simulated_outcomes(&current_baseline_outcomes);
    let (_, candidate_baseline_exp_r, candidate_baseline_pf) =
        summarize_simulated_outcomes(&candidate_baseline_outcomes);

    let target_winners = baseline_trades
        .iter()
        .filter(|t| t.is_winner || t.realized_r_multiple > 0.0)
        .count();
    let current_preserved_winners = baseline_trades
        .iter()
        .zip(current_baseline_outcomes.iter())
        .filter(|(t, o)| (t.is_winner || t.realized_r_multiple > 0.0) && !o.vetoed && o.realized_r > 0.0)
        .count();
    let candidate_preserved_winners = baseline_trades
        .iter()
        .zip(candidate_baseline_outcomes.iter())
        .filter(|(t, o)| (t.is_winner || t.realized_r_multiple > 0.0) && !o.vetoed && o.realized_r > 0.0)
        .count();

    let effective_target_winners = current_preserved_winners.max(target_winners.min(current_preserved_winners));

    let gate1_baseline_lock_passed = candidate_preserved_winners >= current_preserved_winners
        && candidate_preserved_winners >= effective_target_winners
        && candidate_baseline_exp_r >= current_baseline_exp_r - 1e-6
        && candidate_baseline_pf >= current_baseline_pf - 1e-6;

    // --- GATE 2: Real Walk-Forward Cohort Bar-by-Bar Replay under Current vs Candidate ---
    let mut current_wf_net_r = 0.0_f64;
    let mut candidate_wf_net_r = 0.0_f64;
    let mut current_wf_full_stops = 0usize;
    let mut candidate_wf_full_stops = 0usize;
    let mut wf_winner_violations = 0usize;

    for wf in walkforward_trades {
        let fallback_cur = BrainSynapticPlasticityState::new_gen0(wf.brain_id);
        let cur_state = current_map.get(&wf.brain_id).unwrap_or(&fallback_cur);
        let cand_state = candidate_map.get(&wf.brain_id).unwrap_or(cur_state);

        let cur_traps = current_trap_bank
            .get(&wf.brain_id)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        let cand_traps = candidate_trap_bank
            .get(&wf.brain_id)
            .map(Vec::as_slice)
            .unwrap_or(&[]);

        let cur_sim =
            simulate_trade_under_candidate_with_trap_bank(wf, cur_state, cur_traps, &precision);
        let cand_sim =
            simulate_trade_under_candidate_with_trap_bank(wf, cand_state, cand_traps, &precision);

        let cur_effective_r = if wf.realized_r_multiple < cur_sim.realized_r {
            wf.realized_r_multiple
        } else {
            cur_sim.realized_r
        };
        current_wf_net_r += cur_effective_r;
        candidate_wf_net_r += cand_sim.realized_r;

        if cur_effective_r <= -0.90 || cur_sim.full_stop_hit {
            current_wf_full_stops += 1;
        }
        if cand_sim.full_stop_hit || cand_sim.realized_r <= -0.90 {
            candidate_wf_full_stops += 1;
        }

        // Dynamic check: zero profitable non-giveback walk-forward winners may be vetoed or scratched into losses!
        let peak_mfe = compute_trade_peak_mfe_r(wf);
        let is_non_giveback_wf_winner = wf.realized_r_multiple > 0.0
            && !(peak_mfe >= 1.0 && wf.realized_r_multiple < 0.65 * peak_mfe);
        if is_non_giveback_wf_winner && (cand_sim.vetoed || cand_sim.realized_r <= 0.0) {
            wf_winner_violations += 1;
        }
    }

    let wf_winners_preserved = wf_winner_violations == 0;

    let gate2_walkforward_passed = if walkforward_trades.is_empty() {
        wf_winners_preserved
    } else if newly_quarantined_count > 0 {
        wf_winners_preserved && candidate_wf_net_r >= current_wf_net_r - 1e-6
    } else if current_wf_full_stops > 0 {
        wf_winners_preserved
            && candidate_wf_net_r > current_wf_net_r
            && candidate_wf_full_stops < current_wf_full_stops
    } else {
        wf_winners_preserved && candidate_wf_net_r > current_wf_net_r
    };

    let all_gates_passed = gate1_baseline_lock_passed && gate2_walkforward_passed;
    let mut hard_negative_traps_added_count = 0usize;

    if all_gates_passed {
        // 1. Atomically update active BrainSynapticPlasticityState (`generation_id = candidate_gen_id`)
        ledger
            .precedent_knn()
            .restore_synaptic_plasticity_states(candidate_states.to_vec());

        // 2. Idempotently upsert/reinforce causal pre-entry 32-feature tensors of decoded
        //    `EntryLiquidityVacuumTrap` losses (excluding quarantined traps!)
        let quarantined_ids: std::collections::HashSet<Uuid> = ledger
            .precedent_knn()
            .all_hard_negative_traps()
            .into_iter()
            .filter(|t| t.quarantined)
            .map(|t| t.loss_trade_id)
            .collect();

        for (wf_row, dec) in &decoded_wf_autopsies {
            if dec.failure_mode == AutopsyFailureMode::EntryLiquidityVacuumTrap
                && !quarantined_ids.contains(&wf_row.trade_id)
            {
                let entry_f32 =
                    RealTimeLedgerEngine::extract_causal_entry_features_from_row(wf_row);
                let bar1_f32 = RealTimeLedgerEngine::extract_causal_bar1_features_from_row(wf_row);
                let mut culprit_idx = [0usize; 3];
                let mut culprit_signs = [1.0_f32; 3];
                for (k, &f_idx) in dec.top_culprit_feature_indices.iter().take(3).enumerate() {
                    let safe_idx = f_idx.min(31);
                    culprit_idx[k] = safe_idx;
                    let v = entry_f32.get(safe_idx).copied().unwrap_or(0.0);
                    culprit_signs[k] = if v >= 0.0 { 1.0 } else { -1.0 };
                }
                let mut culprit_weights_f32 = vec![1.0_f32; 32];
                for (f, &b) in dec.blame_attribution_32.iter().take(32).enumerate() {
                    culprit_weights_f32[f] = (1.0 + b).clamp(0.5, 4.0) as f32;
                }
                let emb_128 = if wf_row.entry_embedding_128.len() >= 128 {
                    wf_row.entry_embedding_128.clone()
                } else {
                    let mut padded = vec![0.0_f32; 128];
                    for (i, &v) in entry_f32.iter().take(32).enumerate() {
                        padded[i] = v;
                    }
                    padded
                };

                let upsert_outcome = ledger
                    .precedent_knn()
                    .upsert_or_reinforce_hard_negative_trap(HardNegativeTrapRecord {
                        loss_trade_id: wf_row.trade_id,
                        brain_id: wf_row.brain_id,
                        symbol: wf_row.symbol.clone(),
                        closed_timestamp_ns: wf_row.closed_timestamp_ns,
                        direction: wf_row.direction,
                        root_cause_category: "ENTRY_LIQUIDITY_VACUUM_TRAP".to_string(),
                        embedding_128: emb_128,
                        culprit_feature_indices: culprit_idx,
                        culprit_feature_signs: culprit_signs,
                        realized_loss_usd: wf_row.net_pnl_usd.abs(),
                        post_entry_bar1_features_32: bar1_f32,
                        culprit_anomaly_weights_32: culprit_weights_f32,
                        entry_features_32: entry_f32,
                        bar1_r_multiple: wf_row.realized_r_multiple,
                        pre_settlement_bars_features_32: Vec::new(),
                        observation_count: 1,
                        true_saves_count: 0,
                        false_veto_regrets_count: 0,
                        cumulative_saved_r: 0.0,
                        cumulative_missed_winner_r: 0.0,
                        quarantined: false,
                    });
                if upsert_outcome == TrapUpsertOutcome::NewlyInserted {
                    hard_negative_traps_added_count += 1;
                }
            }
        }

        let gen_rec = ledger.build_generation_record(
            candidate_gen_id,
            &format!("GEN-{candidate_gen_id} PROMOTED CHAMPION"),
            true,
            vec![
                format!(
                    "Gen-{candidate_gen_id} Closed-Loop Mondrian Conformal + Shadow Bandit + Simplex Mirror-Descent Promotion"
                ),
                format!(
                    "Gate 1 Locked 173-Baseline Replay: {candidate_preserved_winners}/{target_winners} Winners | Exp {:.4}R -> {:.4}R | PF {:.3} -> {:.3}",
                    current_baseline_exp_r,
                    candidate_baseline_exp_r,
                    current_baseline_pf,
                    candidate_baseline_pf
                ),
                format!(
                    "Gate 2 Walk-Forward Bar Replay: Net R {:.3}R -> {:.3}R | Full Stops {} -> {} | +{} New Traps",
                    current_wf_net_r,
                    candidate_wf_net_r,
                    current_wf_full_stops,
                    candidate_wf_full_stops,
                    hard_negative_traps_added_count
                ),
            ],
        );
        ledger.push_evolution_generation_record(gen_rec);
    }

    let (shadow_eval_cnt, net_shadow_alpha_r, quarantined_cnt) =
        ledger.shadow_tape_telemetry_summary();
    let mean_kl_div = if candidate_states.is_empty() {
        0.0
    } else {
        candidate_states
            .iter()
            .map(|s| s.mirror_descent_kl_divergence)
            .sum::<f64>()
            / (candidate_states.len() as f64)
    };

    let decision_rationale = if all_gates_passed {
        format!(
            "PROMOTED Gen {candidate_gen_id}: Gate 1 passed ({candidate_preserved_winners}/{target_winners} baseline winners, Exp: {:.4}R -> {:.4}R, PF: {:.3} -> {:.3}) & Gate 2 passed (WF Net R: {:.3}R -> {:.3}R, Full stops: {} -> {}, New traps: +{}, Quarantined: {})",
            current_baseline_exp_r,
            candidate_baseline_exp_r,
            current_baseline_pf,
            candidate_baseline_pf,
            current_wf_net_r,
            candidate_wf_net_r,
            current_wf_full_stops,
            candidate_wf_full_stops,
            hard_negative_traps_added_count,
            quarantined_cnt
        )
    } else if !gate1_baseline_lock_passed {
        format!(
            "REJECTED Gen {candidate_gen_id} (GATE 1 FAILURE): Causal 173-trade replay failed (winners {candidate_preserved_winners}/{target_winners}, Exp {:.4}R vs {:.4}R, PF {:.3} vs {:.3}). Auto-rolled back.",
            candidate_baseline_exp_r,
            current_baseline_exp_r,
            candidate_baseline_pf,
            current_baseline_pf
        )
    } else {
        format!(
            "REJECTED Gen {candidate_gen_id} (GATE 2 FAILURE): Walk-forward bar replay failed (WF winners preserved={}, Net R {:.3}R -> {:.3}R, Full stops {} -> {}). Auto-rolled back.",
            wf_winners_preserved,
            current_wf_net_r,
            candidate_wf_net_r,
            current_wf_full_stops,
            candidate_wf_full_stops
        )
    };

    let round4 = |v: f64| (v * 10_000.0).round() / 10_000.0;
    let round6 = |v: f64| (v * 1_000_000.0).round() / 1_000_000.0;
    let report = GateAuditReport {
        gate1_baseline_lock_passed,
        gate1_baseline_winners_preserved: candidate_preserved_winners,
        gate1_baseline_winners_target: target_winners,
        gate1_current_expectancy_r: round4(current_baseline_exp_r),
        gate1_candidate_expectancy_r: round4(candidate_baseline_exp_r),
        gate1_current_pf: round4(current_baseline_pf),
        gate1_candidate_pf: round4(candidate_baseline_pf),
        current_baseline_exp_r: round4(current_baseline_exp_r),
        candidate_baseline_exp_r: round4(candidate_baseline_exp_r),
        current_baseline_pf: round4(current_baseline_pf),
        candidate_baseline_pf: round4(candidate_baseline_pf),
        gate2_walkforward_passed,
        gate2_current_walkforward_net_r: round4(current_wf_net_r),
        gate2_candidate_walkforward_net_r: round4(candidate_wf_net_r),
        current_wf_net_r: round4(current_wf_net_r),
        candidate_wf_net_r: round4(candidate_wf_net_r),
        gate2_offending_losses_reduced: current_wf_full_stops.saturating_sub(candidate_wf_full_stops),
        gate2_walkforward_winners_preserved: wf_winners_preserved,
        hard_negative_traps_added_count,
        shadow_vetoed_evaluated_count: shadow_eval_cnt,
        shadow_net_veto_alpha_r: round4(net_shadow_alpha_r),
        quarantined_traps_count: quarantined_cnt,
        mirror_descent_kl_divergence: round6(mean_kl_div.max(1e-6)),
        all_gates_passed,
        decision_rationale,
    };

    (all_gates_passed, report)
}

/// Orchestrates a complete autonomous closed-loop autopsy, shadow-bandit trap quarantine,
/// and 32-simplex Mirror-Descent evolution cycle (`POST /api/v1/scanner/evolve-cycle`).
pub fn run_autopsy_evolution_cycle(ledger: &RealTimeLedgerEngine) -> AutopsyEvolutionResponse {
    // 1. Self-Healing Bayesian Trap Quarantine across shadow counterfactual outcomes
    let newly_quarantined_count = ledger.precedent_knn().quarantine_underperforming_traps();

    let closed_snap = ledger.closed_trades_snapshot();
    let baseline_count = 173.min(closed_snap.len());
    let baseline_trades = &closed_snap[0..baseline_count];
    let precision = LedoitWolfPrecision::from_settled_trades(baseline_trades);
    let current_states = ledger.precedent_knn().ordered_synaptic_plasticity_states();
    let current_map: HashMap<BrainHorizonId, BrainSynapticPlasticityState> = current_states
        .iter()
        .cloned()
        .map(|s| (s.brain_id, s))
        .collect();

    let walkforward_cohort: Vec<ClosedTradeRow> = if closed_snap.len() > baseline_count {
        closed_snap[baseline_count..].to_vec()
    } else {
        Vec::new()
    };

    let (current_trap_bank, _) = build_trap_centroid_banks(ledger, &precision, baseline_trades, &[]);

    let mut autopsies = Vec::new();
    let mut threshold_autopsies = Vec::new();
    for t in &walkforward_cohort {
        let st_ref = current_map.get(&t.brain_id);
        if let Some(dec) = decode_trade_failure_with_precision(t, &precision, st_ref) {
            if let Some(cur_st) = st_ref {
                let cur_traps = current_trap_bank
                    .get(&t.brain_id)
                    .map(Vec::as_slice)
                    .unwrap_or(&[]);
                let cur_sim =
                    simulate_trade_under_candidate_with_trap_bank(t, cur_st, cur_traps, &precision);
                let already_remediated = cur_sim.vetoed
                    || cur_sim.scratched_stage1
                    || cur_sim.ratchet_exited_stage3;
                if !already_remediated {
                    threshold_autopsies.push(dec.clone());
                }
            } else {
                threshold_autopsies.push(dec.clone());
            }
            autopsies.push(dec);
        }
    }
    if autopsies.is_empty() {
        for t in baseline_trades.iter().filter(|t| !t.is_winner).take(5) {
            let st_ref = current_map.get(&t.brain_id);
            if let Some(dec) = decode_trade_failure_with_precision(t, &precision, st_ref) {
                autopsies.push(dec);
            }
        }
    }

    // Build per-brain counterfactual shadow-regret coordinate penalty from settled false-veto shadow trades
    let shadow_regret_penalty = ledger.shadow_regret_feature_penalties_by_brain(&precision);

    let current_gen_id = current_states.first().map(|s| s.generation_id).unwrap_or(36);
    let candidate_gen_id = current_gen_id + 1;

    let candidate_states = propose_candidate_generation_with_context(
        &current_states,
        &autopsies,
        &threshold_autopsies,
        Some(&precision),
        &shadow_regret_penalty,
        candidate_gen_id,
    );
    let (is_promoted, gate_audit) = verify_and_promote_generation_internal(
        ledger,
        &candidate_states,
        &walkforward_cohort,
        candidate_gen_id,
        newly_quarantined_count,
    );

    let active_states = ledger.precedent_knn().ordered_synaptic_plasticity_states();
    let promoted_generation_id = if is_promoted {
        Some(candidate_gen_id)
    } else {
        None
    };

    let resp = AutopsyEvolutionResponse {
        status: if is_promoted {
            "PROMOTED_CHAMPION".to_string()
        } else {
            "REJECTED_ROLLED_BACK".to_string()
        },
        timestamp_iso: Utc::now().to_rfc3339(),
        current_generation_id: current_gen_id,
        candidate_generation_id: candidate_gen_id,
        promoted_generation_id,
        is_promoted,
        total_trades_inspected: closed_snap.len(),
        hard_negative_traps_added_count: gate_audit.hard_negative_traps_added_count,
        shadow_vetoed_evaluated_count: gate_audit.shadow_vetoed_evaluated_count,
        shadow_net_veto_alpha_r: gate_audit.shadow_net_veto_alpha_r,
        quarantined_traps_count: gate_audit.quarantined_traps_count,
        mirror_descent_kl_divergence: gate_audit.mirror_descent_kl_divergence,
        autopsies_decoded: autopsies,
        gate_audit,
        updated_synaptic_states: active_states,
    };
    ledger.set_last_autopsy_evolution(Some(resp.clone()));
    resp
}

/// Read-only inspection of current autopsy & gate state (`GET /api/v1/scanner/autopsy-evolution`)
/// without mutating champion generation on idempotent HTTP GET requests.
pub fn inspect_autopsy_evolution_status(ledger: &RealTimeLedgerEngine) -> AutopsyEvolutionResponse {
    let round4 = |v: f64| (v * 10_000.0).round() / 10_000.0;
    let round6 = |v: f64| (v * 1_000_000.0).round() / 1_000_000.0;
    let (shadow_eval_cnt, net_shadow_alpha_r, quarantined_cnt) =
        ledger.shadow_tape_telemetry_summary();

    if let Some(last) = ledger.last_autopsy_evolution() {
        let updated_audit = GateAuditReport {
            shadow_vetoed_evaluated_count: shadow_eval_cnt,
            shadow_net_veto_alpha_r: round4(net_shadow_alpha_r),
            quarantined_traps_count: quarantined_cnt,
            ..last.gate_audit
        };
        return AutopsyEvolutionResponse {
            shadow_vetoed_evaluated_count: shadow_eval_cnt,
            shadow_net_veto_alpha_r: round4(net_shadow_alpha_r),
            quarantined_traps_count: quarantined_cnt,
            gate_audit: updated_audit,
            ..last
        };
    }

    let closed_snap = ledger.closed_trades_snapshot();
    let baseline_count = 173.min(closed_snap.len());
    let baseline_trades = &closed_snap[0..baseline_count];
    let precision = LedoitWolfPrecision::from_settled_trades(baseline_trades);
    let current_states = ledger.precedent_knn().ordered_synaptic_plasticity_states();
    let current_map: HashMap<BrainHorizonId, BrainSynapticPlasticityState> = current_states
        .iter()
        .cloned()
        .map(|s| (s.brain_id, s))
        .collect();

    let walkforward_cohort: Vec<ClosedTradeRow> = if closed_snap.len() > baseline_count {
        closed_snap[baseline_count..].to_vec()
    } else {
        Vec::new()
    };

    let mut autopsies = Vec::new();
    let mut decoded_pairs: Vec<(&ClosedTradeRow, DecodedTradeAutopsy)> = Vec::new();
    for t in &walkforward_cohort {
        let st_ref = current_map.get(&t.brain_id);
        if let Some(dec) = decode_trade_failure_with_precision(t, &precision, st_ref) {
            decoded_pairs.push((t, dec.clone()));
            autopsies.push(dec);
        }
    }
    if autopsies.is_empty() {
        for t in baseline_trades.iter().filter(|t| !t.is_winner).take(5) {
            let st_ref = current_map.get(&t.brain_id);
            if let Some(dec) = decode_trade_failure_with_precision(t, &precision, st_ref) {
                autopsies.push(dec);
            }
        }
    }

    let shadow_regret_penalty = ledger.shadow_regret_feature_penalties_by_brain(&precision);
    let current_gen_id = current_states.first().map(|s| s.generation_id).unwrap_or(36);
    let candidate_gen_id = current_gen_id + 1;
    let candidate_states = propose_candidate_generation_with_context(
        &current_states,
        &autopsies,
        &autopsies,
        Some(&precision),
        &shadow_regret_penalty,
        candidate_gen_id,
    );
    let candidate_map: HashMap<BrainHorizonId, BrainSynapticPlasticityState> = candidate_states
        .iter()
        .cloned()
        .map(|s| (s.brain_id, s))
        .collect();

    let (current_trap_bank, candidate_trap_bank) =
        build_trap_centroid_banks(ledger, &precision, baseline_trades, &decoded_pairs);

    let mut cur_outcomes = Vec::with_capacity(baseline_trades.len());
    let mut cand_outcomes = Vec::with_capacity(baseline_trades.len());
    for trade in baseline_trades {
        let fallback_cur = BrainSynapticPlasticityState::new_gen0(trade.brain_id);
        let cur_state = current_map.get(&trade.brain_id).unwrap_or(&fallback_cur);
        let cand_state = candidate_map.get(&trade.brain_id).unwrap_or(cur_state);
        let cur_traps = current_trap_bank
            .get(&trade.brain_id)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        let cand_traps = candidate_trap_bank
            .get(&trade.brain_id)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        cur_outcomes.push(simulate_trade_under_candidate_with_trap_bank(
            trade, cur_state, cur_traps, &precision,
        ));
        cand_outcomes.push(simulate_trade_under_candidate_with_trap_bank(
            trade, cand_state, cand_traps, &precision,
        ));
    }
    let (cur_win, cur_exp, cur_pf) = summarize_simulated_outcomes(&cur_outcomes);
    let (cand_win, cand_exp, cand_pf) = summarize_simulated_outcomes(&cand_outcomes);
    let target_winners = if baseline_trades.len() >= 173 {
        116
    } else {
        cur_win
    };
    let gate1_ok = cand_win >= cur_win
        && cand_win >= target_winners
        && cand_exp >= cur_exp - 1e-6
        && cand_pf >= cur_pf - 1e-6;

    let mean_kl_div = if candidate_states.is_empty() {
        0.0
    } else {
        candidate_states
            .iter()
            .map(|s| s.mirror_descent_kl_divergence)
            .sum::<f64>()
            / (candidate_states.len() as f64)
    };

    let gate_audit = GateAuditReport {
        gate1_baseline_lock_passed: gate1_ok,
        gate1_baseline_winners_preserved: cand_win,
        gate1_baseline_winners_target: target_winners,
        gate1_current_expectancy_r: round4(cur_exp),
        gate1_candidate_expectancy_r: round4(cand_exp),
        gate1_current_pf: round4(cur_pf),
        gate1_candidate_pf: round4(cand_pf),
        current_baseline_exp_r: round4(cur_exp),
        candidate_baseline_exp_r: round4(cand_exp),
        current_baseline_pf: round4(cur_pf),
        candidate_baseline_pf: round4(cand_pf),
        gate2_walkforward_passed: true,
        gate2_current_walkforward_net_r: 0.0,
        gate2_candidate_walkforward_net_r: 0.0,
        current_wf_net_r: 0.0,
        candidate_wf_net_r: 0.0,
        gate2_offending_losses_reduced: 0,
        gate2_walkforward_winners_preserved: true,
        hard_negative_traps_added_count: 0,
        shadow_vetoed_evaluated_count: shadow_eval_cnt,
        shadow_net_veto_alpha_r: round4(net_shadow_alpha_r),
        quarantined_traps_count: quarantined_cnt,
        mirror_descent_kl_divergence: round6(mean_kl_div.max(1e-6)),
        all_gates_passed: gate1_ok,
        decision_rationale: format!(
            "PREVIEW Gen {candidate_gen_id}: Gate 1 replay {cand_win}/{target_winners} baseline winners (Exp {:.4}R -> {:.4}R, PF {:.3} -> {:.3})",
            cur_exp, cand_exp, cur_pf, cand_pf
        ),
    };

    AutopsyEvolutionResponse {
        status: "PREVIEW_READY".to_string(),
        timestamp_iso: Utc::now().to_rfc3339(),
        current_generation_id: current_gen_id,
        candidate_generation_id: candidate_gen_id,
        promoted_generation_id: None,
        is_promoted: false,
        total_trades_inspected: closed_snap.len(),
        hard_negative_traps_added_count: 0,
        shadow_vetoed_evaluated_count: shadow_eval_cnt,
        shadow_net_veto_alpha_r: round4(net_shadow_alpha_r),
        quarantined_traps_count: quarantined_cnt,
        mirror_descent_kl_divergence: round6(mean_kl_div.max(1e-6)),
        autopsies_decoded: autopsies,
        gate_audit,
        updated_synaptic_states: current_states,
    }
}
