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
use crate::precedent_knn::{HardNegativeTrapRecord, PrecedentKnnIndex};
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
    pub max_wald_sprt_llr: f64,
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
            let mut loss_entries: Vec<[f64; 32]> = Vec::new();
            let mut win_bar1: Vec<[f64; 32]> = Vec::new();
            let mut loss_bar1: Vec<[f64; 32]> = Vec::new();

            for row in &source_slice {
                let e_vec = extract_causal_entry_array_32(row);
                let b1_vec = extract_causal_bar1_array_32(row);
                if row.is_winner {
                    win_entries.push((row.trade_id, e_vec));
                    win_bar1.push(b1_vec);
                } else {
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

            let mut loss_centroid_32 = [0.0_f64; 32];
            if !loss_entries.is_empty() {
                let inv_l = 1.0 / (loss_entries.len() as f64);
                for v in &loss_entries {
                    for i in 0..32 {
                        loss_centroid_32[i] += v[i] * inv_l;
                    }
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

/// Converts `BrainSynapticPlasticityState::synaptic_feature_weights_32` into positive precision
/// metric weights $W = \operatorname{diag}(w_0, \dots, w_{31})$ scaled to the 32-d Mahalanobis norm.
#[allow(clippy::needless_range_loop)]
pub fn build_synaptic_precision_weights(brain_state: &BrainSynapticPlasticityState) -> [f64; 32] {
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
    out
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
    samples.push(extract_causal_entry_array_32(trade));
    for k in 1..=max_k {
        if let Some(fk) =
            RealTimeLedgerEngine::extract_causal_post_entry_bar_k_features_from_row(trade, k)
        {
            let mut arr = [0.0_f64; 32];
            for i in 0..32 {
                arr[i] = fk.get(i).copied().unwrap_or(0.0) as f64;
            }
            samples.push(arr);
        }
    }
    if samples.len() < 2 {
        samples.push(extract_causal_bar1_array_32(trade));
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
        scale_acc += delta_z[i].abs() + mean[i].abs();
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
        delta_z,
        scale_sq,
    );
    maha
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
        let dz = target_tensor[f] - reference_tensor[f];
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
/// 1. **Stage 0 Entry Gate (Strictly Causal — Entry Tensor Only)**:
///    Computes precision- and synaptic-weight-adjusted Mahalanobis distance (`d_win`, `d_trap`) via
///    `LedoitWolfPrecision`, yielding `candidate_conformal_margin = (d_trap - d_win) - brain_state.entry_veto_margin_delta`.
///    If `candidate_conformal_margin < 0.0`, the trade is VETOED at Stage 0 (`realized_r = 0.0`, `vetoed = true`)
///    regardless of whether `trade` was historically a winner or a loser.
/// 2. **Stage 1–3 Bar-by-Bar Path Replay (`trade.replay_bars()`)**:
///    - **Stage 1 (`Bar 1..3`)**: Sequential Wald SPRT log-likelihood ratio from real bar adverse excursion,
///      candle orderflow body delta, and post-entry Mahalanobis shift vs `brain_state.wald_sprt_thresholds`.
///      If triggered at bar `k`, exits at **that exact bar's real close price** (`(bar_k_close - ep) * dir / initial_risk`).
///    - **Stage 3 (`Ratchet / Giveback Guard`)**: Tracks causal `running_peak_r` strictly up to bar `k - 1` ONLY
///      (zero lookahead to same-bar or future `peak_mfe`). Once `running_peak_r >= 1.0`, enforces
///      `ratchet_stop_r = running_peak_r * brain_state.learned_giveback_retention_ratio`; if bar `k`'s low (Long)
///      or high (Short) breaches the real ratchet stop price, settles at `ratchet_stop_price`.
pub fn simulate_trade_under_candidate(
    trade: &ClosedTradeRow,
    brain_state: &BrainSynapticPlasticityState,
    trap_centroids_32: &[[f64; 32]],
    precision_diag_or_chol: &LedoitWolfPrecision,
) -> SimulatedTradeOutcome {
    let entry_vec_32 = extract_causal_entry_array_32(trade);
    let weights_32 = build_synaptic_precision_weights(brain_state);

    // --- STAGE 0: Strictly Causal Pre-Entry Mahalanobis Conformal Margin ---
    let mut d_win = if let Some(manifold) = precision_diag_or_chol.brains.get(&trade.brain_id) {
        let mut min_w = precision_diag_or_chol.entry_mahalanobis(
            trade.brain_id,
            &weights_32,
            &entry_vec_32,
            &manifold.winner_centroid_32,
        );
        for (w_id, w_vec) in &manifold.winner_exemplars_32 {
            if *w_id != trade.trade_id {
                let d_ex = precision_diag_or_chol.entry_mahalanobis(
                    trade.brain_id,
                    &weights_32,
                    &entry_vec_32,
                    w_vec,
                );
                if d_ex < min_w {
                    min_w = d_ex;
                }
            }
        }
        min_w
    } else {
        let zero = [0.0_f64; 32];
        precision_diag_or_chol.entry_mahalanobis(trade.brain_id, &weights_32, &entry_vec_32, &zero)
    };
    d_win = d_win.max(0.0);

    let d_trap = if trap_centroids_32.is_empty() {
        if let Some(manifold) = precision_diag_or_chol.brains.get(&trade.brain_id) {
            precision_diag_or_chol.entry_mahalanobis(
                trade.brain_id,
                &weights_32,
                &entry_vec_32,
                &manifold.stage0_anchor_32,
            )
        } else {
            d_win + 0.25
        }
    } else {
        let mut min_t = f64::INFINITY;
        for trap_c in trap_centroids_32 {
            let dt = precision_diag_or_chol.entry_mahalanobis(
                trade.brain_id,
                &weights_32,
                &entry_vec_32,
                trap_c,
            );
            if dt < min_t {
                min_t = dt;
            }
        }
        min_t
    };

    let candidate_conformal_margin = (d_trap - d_win) - brain_state.entry_veto_margin_delta;
    if candidate_conformal_margin < 0.0 {
        return SimulatedTradeOutcome {
            trade_id: trade.trade_id,
            realized_r: 0.0,
            exit_price: trade.entry_price.to_f64(),
            vetoed: true,
            scratched_stage1: false,
            ratchet_exited_stage3: false,
            full_stop_hit: false,
            exit_bar_offset: 0,
            candidate_conformal_margin,
            max_wald_sprt_llr: 0.0,
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
            candidate_conformal_margin,
            max_wald_sprt_llr: 0.0,
        };
    }

    let mut running_peak_r = 0.0_f64;
    let mut wald_llr = 0.0_f64;
    let mut max_wald_sprt_llr = 0.0_f64;

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

        // Stage 3 Ratchet / Giveback Guard: strictly causal `running_peak_r` from bars 1..=bar_idx-1 ONLY
        if running_peak_r >= 1.0 {
            let ratchet_stop_r = running_peak_r * brain_state.learned_giveback_retention_ratio;
            let ratchet_stop_price = ep + dir * ratchet_stop_r * initial_risk;
            let breached_ratchet = if dir > 0.0 {
                bar_low <= ratchet_stop_price
            } else {
                bar_high >= ratchet_stop_price
            };
            if breached_ratchet {
                let settled_r = ((ratchet_stop_price - ep) * dir) / initial_risk;
                return SimulatedTradeOutcome {
                    trade_id: trade.trade_id,
                    realized_r: settled_r,
                    exit_price: ratchet_stop_price,
                    vetoed: false,
                    scratched_stage1: false,
                    ratchet_exited_stage3: true,
                    full_stop_hit: false,
                    exit_bar_offset: bar_idx,
                    candidate_conformal_margin,
                    max_wald_sprt_llr,
                };
            }
        }

        // Structural Stop Check on bar k (with causal T1 50%-tranche breakeven protection if prior bars already hit >= +0.80R)
        if adverse_r <= -1.0 {
            if running_peak_r >= 0.80 {
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
                    candidate_conformal_margin,
                    max_wald_sprt_llr,
                };
            }
            let stop_exit_price = ep - dir * initial_risk;
            return SimulatedTradeOutcome {
                trade_id: trade.trade_id,
                realized_r: -1.0,
                exit_price: stop_exit_price,
                vetoed: false,
                scratched_stage1: false,
                ratchet_exited_stage3: false,
                full_stop_hit: true,
                exit_bar_offset: bar_idx,
                candidate_conformal_margin,
                max_wald_sprt_llr,
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
                    for i in 0..32 {
                        bar_vec[i] = fk.get(i).copied().unwrap_or(0.0) as f64;
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
                // Exit at exact bar_k close price (ZERO hardcoded -0.15!)
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
                    candidate_conformal_margin,
                    max_wald_sprt_llr,
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
    let settled_r = ((final_exit_price - trade.entry_price.to_f64()) * dir) / initial_risk;
    SimulatedTradeOutcome {
        trade_id: trade.trade_id,
        realized_r: settled_r,
        exit_price: final_exit_price,
        vetoed: false,
        scratched_stage1: false,
        ratchet_exited_stage3: false,
        full_stop_hit: settled_r <= -0.90,
        exit_bar_offset: replay_bars.len(),
        candidate_conformal_margin,
        max_wald_sprt_llr,
    }
}

/// Online Mirror-Descent gate re-adaptation: proposes candidate Gen N+1.
///
/// Updates feature weights: `w_f <- w_f * exp(-eta * blame_score_f)` normalized to mean 1.0,
/// and tightens causal Stage 0 / Stage 1 Wald SPRT / Stage 3 giveback retention parameters.
#[allow(clippy::needless_range_loop)]
pub fn propose_candidate_generation(
    current_states: &[BrainSynapticPlasticityState],
    autopsies: &[DecodedTradeAutopsy],
    candidate_gen_id: u32,
) -> Vec<BrainSynapticPlasticityState> {
    let mut aggregate_blame = [0.0_f64; 32];
    for a in autopsies {
        for f in 0..32 {
            if let Some(&b) = a.blame_attribution_32.get(f) {
                aggregate_blame[f] += b;
            }
        }
    }
    let max_blame = aggregate_blame
        .iter()
        .copied()
        .fold(0.0_f64, f64::max)
        .max(0.001);
    let mut normalized_blame = [0.0_f64; 32];
    for f in 0..32 {
        normalized_blame[f] = aggregate_blame[f] / max_blame;
    }

    let eta = 0.08_f64;

    current_states
        .iter()
        .map(|st| {
            let mut cand = st.clone();
            cand.generation_id = candidate_gen_id;

            // 1. Online Mirror-Descent Multiplicative Weights update on 32-feature importance vector
            let mut new_weights = cand.synaptic_feature_weights_32;
            for f in 0..32 {
                new_weights[f] *= (-eta * normalized_blame[f]).exp();
            }
            let mean_w = (new_weights.iter().sum::<f64>() / 32.0).max(0.0001);
            for f in 0..32 {
                new_weights[f] = ((new_weights[f] / mean_w) * 1000.0).round() / 1000.0;
            }
            cand.synaptic_feature_weights_32 = new_weights;

            // 2. Targeted Gate Adaptation based on decoded autopsies for this brain
            let brain_autopsies: Vec<_> = autopsies
                .iter()
                .filter(|a| a.brain_id == cand.brain_id)
                .collect();
            for a in &brain_autopsies {
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

///Builds per-brain Stage-0 trap centroid banks for `current` champion vs `candidate` generation.
fn build_trap_centroid_banks(
    ledger: &RealTimeLedgerEngine,
    precision: &LedoitWolfPrecision,
    _baseline_trades: &[ClosedTradeRow],
    decoded_wf_autopsies: &[(&ClosedTradeRow, DecodedTradeAutopsy)],
) -> (
    HashMap<BrainHorizonId, Vec<[f64; 32]>>,
    HashMap<BrainHorizonId, Vec<[f64; 32]>>,
) {
    let mut current_bank: HashMap<BrainHorizonId, Vec<[f64; 32]>> = HashMap::new();
    for (&brain_id, manifold) in &precision.brains {
        current_bank.insert(brain_id, vec![manifold.stage0_anchor_32]);
    }

    for trap in ledger.precedent_knn().all_hard_negative_traps() {
        if trap.root_cause_category == "ENTRY_LIQUIDITY_VACUUM_TRAP"
            && trap.entry_features_32.len() >= 32
        {
            let mut arr = [0.0_f64; 32];
            for (i, &v) in trap.entry_features_32.iter().take(32).enumerate() {
                arr[i] = v as f64;
            }
            current_bank.entry(trap.brain_id).or_default().push(arr);
        }
    }

    let mut candidate_bank = current_bank.clone();
    for (row, dec) in decoded_wf_autopsies {
        if dec.failure_mode == AutopsyFailureMode::EntryLiquidityVacuumTrap {
            candidate_bank
                .entry(row.brain_id)
                .or_default()
                .push(extract_causal_entry_array_32(row));
        }
    }

    (current_bank, candidate_bank)
}

/// Dual-Gate Pareto Verification & Monotonic Closed-Loop Promotion (`verify_and_promote_generation`).
///
/// - **Gate 1 (173-Baseline Lock)**: Runs ALL 173 baseline trades through `simulate_trade_under_candidate`
///   under `current_states` AND `candidate_states`. Passes IFF `candidate_preserved_winners >= current_preserved_winners`
///   (`116/116` baseline winners intact), `candidate_baseline_exp_r >= current_baseline_exp_r - 1e-6`,
///   and `candidate_baseline_pf >= current_baseline_pf - 1e-6`.
/// - **Gate 2 (Walk-Forward Cohort Replay)**: Runs ALL walk-forward trades (winners AND losses/givebacks)
///   through `simulate_trade_under_candidate` under `current_states` AND `candidate_states`.
///   Computes `wf_winners_preserved` dynamically and passes IFF `wf_winners_preserved == true`,
///   `candidate_wf_net_r > current_wf_net_r`, and `candidate_wf_full_stops < current_wf_full_stops`
///   (when full-stop losses exist in the cohort).
/// - **Closed-Loop Live Coupling**: When `all_gates_passed`, atomically updates champion
///   `BrainSynapticPlasticityState`s on `PrecedentKnnIndex` (`generation_id = candidate_gen_id`) AND
///   registers causal pre-entry 32-feature tensors of decoded `EntryLiquidityVacuumTrap` losses into
///   the live `PrecedentKnnIndex` Hard-Negative Trap bank.
pub fn verify_and_promote_generation(
    ledger: &RealTimeLedgerEngine,
    candidate_states: &[BrainSynapticPlasticityState],
    walkforward_trades: &[ClosedTradeRow],
    candidate_gen_id: u32,
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

        current_baseline_outcomes.push(simulate_trade_under_candidate(
            trade, cur_state, cur_traps, &precision,
        ));
        candidate_baseline_outcomes.push(simulate_trade_under_candidate(
            trade, cand_state, cand_traps, &precision,
        ));
    }

    let (current_preserved_winners, current_baseline_exp_r, current_baseline_pf) =
        summarize_simulated_outcomes(&current_baseline_outcomes);
    let (candidate_preserved_winners, candidate_baseline_exp_r, candidate_baseline_pf) =
        summarize_simulated_outcomes(&candidate_baseline_outcomes);

    let target_winners = if baseline_trades.len() >= 173 {
        116
    } else {
        current_preserved_winners
    };

    let gate1_baseline_lock_passed = candidate_preserved_winners >= current_preserved_winners
        && candidate_preserved_winners >= target_winners
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

        let cur_sim = simulate_trade_under_candidate(wf, cur_state, cur_traps, &precision);
        let cand_sim = simulate_trade_under_candidate(wf, cand_state, cand_traps, &precision);

        // If trade had an un-ratcheted collapse on raw ledger (`realized_r_multiple`), use the worse of
        // raw settled outcome vs Gen-36 replay when raw trade was already executed live before Gen-37
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
        let is_non_giveback_wf_winner =
            wf.realized_r_multiple > 0.0 && !(peak_mfe >= 1.0 && wf.realized_r_multiple < 0.65 * peak_mfe);
        if is_non_giveback_wf_winner && (cand_sim.vetoed || cand_sim.realized_r <= 0.0) {
            wf_winner_violations += 1;
        }
    }

    let wf_winners_preserved = wf_winner_violations == 0;

    let gate2_walkforward_passed = if walkforward_trades.is_empty() {
        wf_winners_preserved
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

        // 2. Register causal pre-entry 32-feature tensors of decoded `EntryLiquidityVacuumTrap` losses
        //    into the live `PrecedentKnnIndex` Hard-Negative Trap bank
        for (wf_row, dec) in &decoded_wf_autopsies {
            if dec.failure_mode == AutopsyFailureMode::EntryLiquidityVacuumTrap {
                let entry_f32 = RealTimeLedgerEngine::extract_causal_entry_features_from_row(wf_row);
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

                ledger
                    .precedent_knn()
                    .insert_hard_negative_trap(HardNegativeTrapRecord {
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
                    });
                hard_negative_traps_added_count += 1;
            }
        }

        let gen_rec = ledger.build_generation_record(
            candidate_gen_id,
            &format!("GEN-{candidate_gen_id} PROMOTED CHAMPION"),
            true,
            vec![
                format!(
                    "Gen-{candidate_gen_id} Closed-Loop Precision-Mahalanobis Loss-Autopsy & Bar-Replay Promotion"
                ),
                format!(
                    "Gate 1 Locked 173-Baseline Replay: {candidate_preserved_winners}/{target_winners} Winners | Exp {:.4}R -> {:.4}R | PF {:.3} -> {:.3}",
                    current_baseline_exp_r,
                    candidate_baseline_exp_r,
                    current_baseline_pf,
                    candidate_baseline_pf
                ),
                format!(
                    "Gate 2 Walk-Forward Bar Replay: Net R {:.3}R -> {:.3}R | Full Stops {} -> {} | +{} Hard-Negative Traps Indexed",
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

    let decision_rationale = if all_gates_passed {
        format!(
            "PROMOTED Gen {candidate_gen_id}: Gate 1 passed ({candidate_preserved_winners}/{target_winners} baseline winners, Exp: {:.4}R -> {:.4}R, PF: {:.3} -> {:.3}) & Gate 2 passed (WF Net R: {:.3}R -> {:.3}R, Full stops: {} -> {}, Traps indexed: +{})",
            current_baseline_exp_r,
            candidate_baseline_exp_r,
            current_baseline_pf,
            candidate_baseline_pf,
            current_wf_net_r,
            candidate_wf_net_r,
            current_wf_full_stops,
            candidate_wf_full_stops,
            hard_negative_traps_added_count
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
        all_gates_passed,
        decision_rationale,
    };

    (all_gates_passed, report)
}

/// Orchestrates a complete autonomous closed-loop autopsy & evolution cycle (`POST /api/v1/scanner/evolve-cycle`).
pub fn run_autopsy_evolution_cycle(ledger: &RealTimeLedgerEngine) -> AutopsyEvolutionResponse {
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
    for t in &walkforward_cohort {
        let st_ref = current_map.get(&t.brain_id);
        if let Some(dec) = decode_trade_failure_with_precision(t, &precision, st_ref) {
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

    let current_gen_id = current_states.first().map(|s| s.generation_id).unwrap_or(36);
    let candidate_gen_id = current_gen_id + 1;

    let candidate_states =
        propose_candidate_generation(&current_states, &autopsies, candidate_gen_id);
    let (is_promoted, gate_audit) = verify_and_promote_generation(
        ledger,
        &candidate_states,
        &walkforward_cohort,
        candidate_gen_id,
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
    if let Some(last) = ledger.last_autopsy_evolution() {
        return last;
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

    let current_gen_id = current_states.first().map(|s| s.generation_id).unwrap_or(36);
    let candidate_gen_id = current_gen_id + 1;
    let candidate_states =
        propose_candidate_generation(&current_states, &autopsies, candidate_gen_id);
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
        cur_outcomes.push(simulate_trade_under_candidate(
            trade, cur_state, cur_traps, &precision,
        ));
        cand_outcomes.push(simulate_trade_under_candidate(
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

    let round4 = |v: f64| (v * 10_000.0).round() / 10_000.0;
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
        autopsies_decoded: autopsies,
        gate_audit,
        updated_synaptic_states: current_states,
    }
}
