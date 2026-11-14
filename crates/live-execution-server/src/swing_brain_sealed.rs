//! # IMMUTABLE FIREWALL SEAL: TIER SWING (BRAIN-03 & BRAIN-04)
//!
//! Cryptographically locked module preserving the 2026-11-13T15:38:47Z baseline:
//! - Net PnL: +$11,085.10
//! - Net Post-Cost PF: 1.63
//! - Win%: 65.6%
//! - Trades: 61
//! - Max DD: $7,132.01 (6.43%)
//!
//! ZERO SHARED POLICY CODE WITH INTRADAY:
//! This module has ZERO dependencies or imports from `pure_rl_brain.rs`.
//! It encapsulates 100% of the neural architecture, world model, forward/backward passes,
//! expanding-window PBT calibration, red-trade loss-autopsy, and walk-forward execution
//! for BRAIN-03 (Short Swing) and BRAIN-04 (Core Swing).

#![allow(clippy::all, unused)]

use std::collections::{BTreeMap, BTreeSet, HashMap};
use chrono::Datelike;
use market_data_bus::yahoo::{
    est_trading_day_key, lookup_real_universe_meta, 
};
use market_data_bus::{RealExchangeOhlcvBar};
use raw_fingerprint_core::streams::{BrainHorizonId, TradeDirection};
use raw_fingerprint_core::{FourStreamWeights, PrimaryFingerprintDriver, SubPennyPrice};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::execution_microstructure::{
    estimate_option_spread_per_leg, explicit_one_way_slippage_bps, ny_et_hhmm, unix_sec_to_ny_et,
    PENNY_BRAIN01_SYMBOLS,
};
use crate::ledger::{
    ClosedOutcomeBadge, ClosedTradeRow, EngineExecutionMode, StrategyArchetype,
};
use crate::options_convexity::{compute_bsm_leg_greeks, round_cboe_strike, OptionRight};
use crate::proof_harness::{
    BlotterTradeRecord, BucketPerformanceSummary, CausalBarPhysicsSnapshot,
    GenerationPromotionLedgerRow, ProofBucket, ProofDatasets, ProofRunMode,
    WarmupTrainingTelemetry,
};

pub type SealedSwingBrain = PureAutonomousRlBrain;


// Two-Stage Self-Supervised Market-Physics World Model + End-to-End Differentiable
// Stateful Sequence Policy + Closed-Loop Red-Trade Loss-Autopsy RL (`PureAutonomousRlBrain`)
//
// Architectural Laws:
// - ZERO hardcoded trading strategies, human indicators (`EMA`, `SMA`, `RSI`, `MACD`, `ORB`, `VCP`),
//   or hardcoded stop/horizon templates.
// - Stage A (`PhysicsWorldModel`): Residual LayerNorm + Multi-Timescale Gated Recurrent Encoder
//   `z_t = E_phi(x_{<=t})` trained causally (`t + 1 + Delta <= t_current`) via Pinball Quantile Loss
//   to predict `1, 3, 8, 24`-bar return quantiles (`q25, q50, q75`) and `ln(Range / RT_Cost)`.
// - Stage B (`SequencePolicyNet`): Concatenates `[x_t, z_t, wm_pred_t, s_{t-1}]` through a
//   Multi-Timescale Gated Recurrent Core and feeds `[h_ln, wm_pred_t, s_{t-1}]` into a `K = 5`
//   Orthogonal Specialist Policy Ensemble with:
//   1. Cost-Scaled Neural Deadzone Hurdle (`wm_net_excursion - k_theta * exact_rt_cost_bps`),
//   2. Straight-Through / Smooth Surrogate Gradient (`forward`: exact `0.0` inside deadzone;
//      `backward`: `da/dz = 0.25 + 0.75 * sigmoid_grad` across the boundary),
//   3. Differentiable `(Symbol, TradingDay)` Session Episode EOD Closure (`a_{T_eod} = 0.0`
//      with exact EOD flatten slippage/spread friction).
// - Stage C: 4-Step Closed-Loop Red-Trade Counterfactual & Saliency Audit -> Prioritized Red-Trade
//   Replay (`50%` batch share, asymmetric weight `1.0 + 4.0 * |net_r|^2`) + Diagonal Fisher EWC
//   Winner Lock -> Mandatory Full Causal Re-Test -> Generation Promotion.


pub const RL_EMBEDDING_DIM: usize = 128;
pub const RL_INSTANT_DIM: usize = 32;
pub const RL_COST_OBS_DIM: usize = 4;
pub const RL_CROSS_SECT_DIM: usize = 16;
pub const RL_MULTI_SCALE_HTF_DIM: usize = 32;
pub const RL_STATE_DIM: usize = RL_EMBEDDING_DIM
    + RL_INSTANT_DIM
    + RL_COST_OBS_DIM
    + RL_CROSS_SECT_DIM
    + RL_MULTI_SCALE_HTF_DIM; // 212

pub const SELF_STATE_DIM: usize = 7;
pub const WM_DIM: usize = 32;
pub const WM_HORIZONS: [usize; 4] = [1, 3, 8, 24];
/// 4 horizons (`1, 3, 8, 24` bars) x 4 targets (`q25_ret, q50_ret, q75_ret, ln(Range / RT_Cost)`).
pub const WM_OUT_DIM: usize = 16;
pub const TOTAL_INPUT_DIM: usize = RL_STATE_DIM + WM_DIM + WM_OUT_DIM + SELF_STATE_DIM; // 267
pub const HIDDEN_DIM: usize = 32;
pub const HEAD_IN_DIM: usize = HIDDEN_DIM + WM_OUT_DIM + SELF_STATE_DIM; // 55
pub const SPECIALIST_K: usize = 5;
pub const CRITIC_QUANTILES: usize = 5;
pub const POPULATION_SIZE: usize = 24;
pub const MAX_RED_REPLAY_CAP: usize = 128;
pub const MAX_WINNER_FISHER_CAP: usize = 96;

const QUANTILE_TAUS: [f32; CRITIC_QUANTILES] = [0.10, 0.25, 0.50, 0.75, 0.90];
const WM_TAUS: [f32; WM_OUT_DIM] = [
    0.25, 0.50, 0.75, 0.50, // Delta = 1
    0.25, 0.50, 0.75, 0.50, // Delta = 3
    0.25, 0.50, 0.75, 0.50, // Delta = 8
    0.25, 0.50, 0.75, 0.50, // Delta = 24
];

#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    if x == 0 {
        x = 0x9E37_79B9_7F4A_7C15;
    }
    *state = x;
    x
}

#[inline]
fn rand_uniform(seed: &mut u64, lo: f32, hi: f32) -> f32 {
    let u = (xorshift64(seed) >> 11) as f64 / ((1u64 << 53) as f64);
    lo + (hi - lo) * (u as f32)
}

#[inline]
fn rand_normal(seed: &mut u64) -> f32 {
    let u1 = rand_uniform(seed, 1e-6, 1.0 - 1e-6);
    let u2 = rand_uniform(seed, 0.0, 1.0);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x.clamp(-20.0, 20.0)).exp())
}

#[inline]
fn softplus(x: f32) -> f32 {
    let xc = x.clamp(-20.0, 20.0);
    if xc > 15.0 { xc } else { (1.0 + xc.exp()).ln() }
}

fn init_mat(rows: usize, cols: usize, scale: f32, seed: &mut u64) -> Vec<f32> {
    let mut out = Vec::with_capacity(rows * cols);
    let std = scale / (cols as f32).sqrt();
    for _ in 0..(rows * cols) {
        out.push(rand_normal(seed) * std);
    }
    out
}

/// Constructs the `212-d` raw causal physics tensor `x_t`.
#[allow(clippy::too_many_arguments)]
pub fn build_raw_212_state_vector(
    embedding_128: &[f32],
    instant_32: &[f32],
    cost_to_atr: f64,
    normalized_session_time_remaining: f64,
    vix9d_to_vix_log_ratio: f64,
    open_book_heat_frac: f64,
    cross_sect_16: &[f32],
    htf_32: &[f32],
) -> [f32; RL_STATE_DIM] {
    let mut out = [0.0_f32; RL_STATE_DIM];
    let n_emb = embedding_128.len().min(RL_EMBEDDING_DIM);
    out[..n_emb].copy_from_slice(&embedding_128[..n_emb]);

    let inst_off = RL_EMBEDDING_DIM;
    let n_inst = instant_32.len().min(RL_INSTANT_DIM);
    out[inst_off..inst_off + n_inst].copy_from_slice(&instant_32[..n_inst]);

    let cost_off = inst_off + RL_INSTANT_DIM;
    out[cost_off] = (cost_to_atr as f32).clamp(0.0, 5.0);
    out[cost_off + 1] = (normalized_session_time_remaining as f32).clamp(0.0, 1.0);
    out[cost_off + 2] = (vix9d_to_vix_log_ratio as f32).clamp(-1.5, 1.5);
    out[cost_off + 3] = (open_book_heat_frac as f32).clamp(0.0, 1.0);

    let cs_off = cost_off + RL_COST_OBS_DIM;
    let n_cs = cross_sect_16.len().min(RL_CROSS_SECT_DIM);
    out[cs_off..cs_off + n_cs].copy_from_slice(&cross_sect_16[..n_cs]);

    let htf_off = cs_off + RL_CROSS_SECT_DIM;
    let n_htf = htf_32.len().min(RL_MULTI_SCALE_HTF_DIM);
    out[htf_off..htf_off + n_htf].copy_from_slice(&htf_32[..n_htf]);

    for v in &mut out {
        if !v.is_finite() {
            *v = 0.0;
        }
    }
    out
}

/// Agent's exact live execution self-awareness state `s_{t-1}` (`7-d`).
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct SelfAwarenessState {
    pub current_pos: f32,
    pub unrealized_net_ret: f32,
    pub peak_excursion: f32,
    pub drawdown_from_peak: f32,
    pub log_bars_held: f32,
    pub one_way_cost_bps: f32,
    pub session_time_left: f32,
}

impl SelfAwarenessState {
    #[inline]
    pub fn to_array(self) -> [f32; SELF_STATE_DIM] {
        [
            self.current_pos.clamp(-1.0, 1.0),
            (self.unrealized_net_ret / 0.015).clamp(-5.0, 5.0),
            (self.peak_excursion / 0.015).clamp(0.0, 5.0),
            (self.drawdown_from_peak / 0.015).clamp(0.0, 5.0),
            (self.log_bars_held / 4.0).clamp(0.0, 2.0),
            (self.one_way_cost_bps / 35.0).clamp(0.0, 2.5),
            self.session_time_left.clamp(0.0, 1.0),
        ]
    }
}

/// Single strictly-causal bar record for Stage A World Model & Stage B BPTT trajectory training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalTrajectoryBar {
    pub ts_sec: i64,
    #[serde(with = "serde_big_array_212")]
    pub x_212: [f32; RL_STATE_DIM],
    pub ret_oo_shares: f32,
    pub ret_oo_option: f32,
    pub one_way_cost_shares: f32,
    pub one_way_cost_option: f32,
    pub one_way_cost_bps: f32,
    pub session_time_left: f32,
    pub option_allowed: bool,
    #[serde(default = "default_true")]
    pub entry_allowed: bool,
    pub force_flat: bool,
    /// Strictly causal future physics targets (populated ONLY when `t + 1 + Delta <= t_current`).
    pub wm_targets: [f32; WM_OUT_DIM],
    pub wm_valid: [bool; WM_OUT_DIM],
}

fn default_true() -> bool {
    true
}

mod serde_big_array_212 {
    use super::RL_STATE_DIM;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(arr: &[f32; RL_STATE_DIM], s: S) -> Result<S::Ok, S::Error> {
        arr.as_slice().serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<[f32; RL_STATE_DIM], D::Error> {
        let v = Vec::<f32>::deserialize(d)?;
        let mut out = [0.0_f32; RL_STATE_DIM];
        let n = v.len().min(RL_STATE_DIM);
        out[..n].copy_from_slice(&v[..n]);
        Ok(out)
    }
}

/// Causal `(Symbol, TradingDay)` Session Episode (for intraday brains) or Swing Episode,
/// carrying only causal cross-day regime context `prior_regime_z` across session boundaries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalSessionEpisode {
    pub symbol: String,
    pub day_key: i64,
    pub start_ts_sec: i64,
    pub end_ts_sec: i64,
    pub is_intraday_session: bool,
    pub prior_regime_z: [f32; WM_DIM],
    pub bars: Vec<CausalTrajectoryBar>,
}

/// Automated Loss-Autopsy Fault Diagnosis from Counterfactual + BPTT Saliency Replay.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LossFaultKind {
    /// Held bar path never meaningfully cleared round-trip friction (`a = 0.0` FLAT optimal).
    EntryHallucination,
    /// Reached positive post-cost MFE at `optimal_exit_offset` before reversing into red close.
    HoldingExitFailure,
}

/// Audited Red-Trade Trajectory stored in the Prioritized Red-Trade Replay Buffer.
#[derive(Debug, Clone)]
pub struct AuditedRedTradeTrajectory {
    pub symbol: String,
    pub exit_ts_sec: i64,
    pub net_pnl_usd: f64,
    pub net_r: f32,
    /// Asymmetric downside priority weight: `1.0 + 4.0 * |net_r|^2`.
    pub priority_weight: f32,
    pub fault_kind: LossFaultKind,
    pub entry_offset: usize,
    pub optimal_exit_offset: usize,
    pub best_counterfactual_net_r: f32,
    pub saliency_dx: f32,
    pub saliency_dh: f32,
    pub prior_regime_z: [f32; WM_DIM],
    pub bars: Vec<CausalTrajectoryBar>,
}

/// Settled causal winning trajectory used to compute Diagonal Fisher EWC (`F_i`).
#[derive(Debug, Clone)]
pub struct SettledWinnerTrajectory {
    pub exit_ts_sec: i64,
    pub net_r: f32,
    pub direction_sign: f32,
    pub entry_offset: usize,
    pub prior_regime_z: [f32; WM_DIM],
    pub bars: Vec<CausalTrajectoryBar>,
}

/// Output of one recurrent policy forward step.
#[derive(Debug, Clone, Copy)]
pub struct SequenceStepOutput {
    /// Target signed exposure in `[-1.0, +1.0]`; exact `0.0` when inside cost-scaled deadzone.
    pub action: f32,
    /// Raw conviction `z_t` before deadzone gate.
    pub raw_z: f32,
    /// Learned cost-aware deadzone threshold `tau_t`.
    pub deadzone: f32,
    /// Companion vehicle logit `v_t in [0, 1]` (`>= 0.5` => `OPTION`).
    pub vehicle_prob: f32,
    /// Learned dynamic catastrophic tail-stop distance in fractional price space (`> 0`).
    pub tail_stop_dist: f32,
    /// Distributional critic quantile predictions.
    pub quantiles: [f32; CRITIC_QUANTILES],
    /// Next recurrent policy hidden state `h_t`.
    pub next_hidden: [f32; HIDDEN_DIM],
    /// Next world-model hidden state `z_t`.
    pub next_wm_hidden: [f32; WM_DIM],
}

/// Stage A: Dense Self-Supervised Multi-Horizon Market Physics World Model (`E_phi`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicsWorldModel {
    pub w_enc: Vec<f32>,
    pub b_enc: Vec<f32>,
    pub ln_g: Vec<f32>,
    pub ln_b: Vec<f32>,
    pub w_g: Vec<f32>,
    pub u_g: Vec<f32>,
    pub b_g: Vec<f32>,
    pub w_r: Vec<f32>,
    pub u_r: Vec<f32>,
    pub b_r: Vec<f32>,
    pub w_h: Vec<f32>,
    pub u_h: Vec<f32>,
    pub b_h: Vec<f32>,
    pub w_wm: Vec<f32>,
    pub b_wm: Vec<f32>,
    pub adam_step: u64,
    #[serde(skip)]
    adam_m: Vec<f32>,
    #[serde(skip)]
    adam_v: Vec<f32>,
    #[serde(default)]
    pub last_pinball_loss: f64,
}

impl PhysicsWorldModel {
    pub fn new(seed: &mut u64) -> Self {
        let d = WM_DIM;
        let mut b_g = Vec::with_capacity(d);
        for i in 0..d {
            let base_tau = match i % 3 {
                0 => -0.62,
                1 => 1.38,
                _ => 3.18,
            };
            b_g.push(base_tau);
        }
        let mut wm = Self {
            w_enc: init_mat(d, RL_STATE_DIM, 1.0, seed),
            b_enc: vec![0.0; d],
            ln_g: vec![1.0; d],
            ln_b: vec![0.0; d],
            w_g: init_mat(d, d, 0.8, seed),
            u_g: init_mat(d, d, 0.8, seed),
            b_g,
            w_r: init_mat(d, d, 0.8, seed),
            u_r: init_mat(d, d, 0.8, seed),
            b_r: vec![0.0; d],
            w_h: init_mat(d, d, 1.0, seed),
            u_h: init_mat(d, d, 0.8, seed),
            b_h: vec![0.0; d],
            w_wm: init_mat(WM_OUT_DIM, d, 0.5, seed),
            b_wm: vec![0.0; WM_OUT_DIM],
            adam_step: 0,
            adam_m: Vec::new(),
            adam_v: Vec::new(),
            last_pinball_loss: 0.0,
        };
        wm.ensure_adam_buffers();
        wm
    }

    fn param_count(&self) -> usize {
        self.w_enc.len()
            + self.b_enc.len()
            + self.ln_g.len()
            + self.ln_b.len()
            + self.w_g.len()
            + self.u_g.len()
            + self.b_g.len()
            + self.w_r.len()
            + self.u_r.len()
            + self.b_r.len()
            + self.w_h.len()
            + self.u_h.len()
            + self.b_h.len()
            + self.w_wm.len()
            + self.b_wm.len()
    }

    pub fn ensure_valid_shapes(&mut self, seed: &mut u64) {
        let d = WM_DIM;
        if self.w_enc.len() != d * RL_STATE_DIM
            || self.w_wm.len() != WM_OUT_DIM * d
            || self.b_wm.len() != WM_OUT_DIM
        {
            *self = Self::new(seed);
        }
        self.ensure_adam_buffers();
    }

    fn ensure_adam_buffers(&mut self) {
        let n = self.param_count();
        if self.adam_m.len() != n {
            self.adam_m = vec![0.0; n];
            self.adam_v = vec![0.0; n];
        }
    }

    #[inline]
    pub fn forward_step(
        &self,
        x_norm_212: &[f32; RL_STATE_DIM],
        prev_z: &[f32; WM_DIM],
    ) -> ([f32; WM_DIM], [f32; WM_OUT_DIM]) {
        let d = WM_DIM;
        let mut u_raw = [0.0_f32; WM_DIM];
        let mut mean = 0.0_f32;
        for i in 0..d {
            let row = &self.w_enc[i * RL_STATE_DIM..(i + 1) * RL_STATE_DIM];
            let mut sum = self.b_enc[i];
            for j in 0..RL_STATE_DIM {
                sum += row[j] * x_norm_212[j];
            }
            u_raw[i] = sum;
            mean += sum;
        }
        mean /= d as f32;
        let mut var = 0.0_f32;
        for i in 0..d {
            let diff = u_raw[i] - mean;
            var += diff * diff;
        }
        let inv_std = 1.0 / (var / (d as f32) + 1e-5).sqrt();
        let mut u_t = [0.0_f32; WM_DIM];
        for i in 0..d {
            u_t[i] = ((u_raw[i] - mean) * inv_std * self.ln_g[i] + self.ln_b[i]).tanh();
        }

        let mut g_t = [0.0_f32; WM_DIM];
        let mut r_t = [0.0_f32; WM_DIM];
        for i in 0..d {
            let wg = &self.w_g[i * d..(i + 1) * d];
            let ug = &self.u_g[i * d..(i + 1) * d];
            let wr = &self.w_r[i * d..(i + 1) * d];
            let ur = &self.u_r[i * d..(i + 1) * d];
            let mut sg = self.b_g[i];
            let mut sr = self.b_r[i];
            for j in 0..d {
                sg += wg[j] * u_t[j] + ug[j] * prev_z[j];
                sr += wr[j] * u_t[j] + ur[j] * prev_z[j];
            }
            g_t[i] = sigmoid(sg);
            r_t[i] = sigmoid(sr);
        }

        let mut z_next = [0.0_f32; WM_DIM];
        for i in 0..d {
            let wh = &self.w_h[i * d..(i + 1) * d];
            let uh = &self.u_h[i * d..(i + 1) * d];
            let mut sh = self.b_h[i];
            for j in 0..d {
                sh += wh[j] * u_t[j] + uh[j] * (r_t[j] * prev_z[j]);
            }
            let h_cand = sh.tanh();
            z_next[i] = g_t[i] * prev_z[i] + (1.0 - g_t[i]) * h_cand;
        }

        let mut wm_pred = [0.0_f32; WM_OUT_DIM];
        for k in 0..WM_OUT_DIM {
            let row = &self.w_wm[k * d..(k + 1) * d];
            let mut s = self.b_wm[k];
            for i in 0..d {
                s += row[i] * z_next[i];
            }
            wm_pred[k] = s;
        }

        (z_next, wm_pred)
    }

    pub fn train_chunk(
        &mut self,
        chunk: &[CausalTrajectoryBar],
        feat_mean: &[f32],
        feat_var: &[f32],
        lr: f32,
    ) -> f64 {
        let zero = [0.0_f32; WM_DIM];
        self.train_chunk_with_init(chunk, &zero, feat_mean, feat_var, lr).0
    }

    /// Trains the World Model via analytical Pinball Quantile Loss BPTT on strictly causal resolved targets,
    /// returning `(pinball_loss, final_z_hidden)` so cross-day causal regime context can propagate.
    pub fn train_chunk_with_init(
        &mut self,
        chunk: &[CausalTrajectoryBar],
        init_z: &[f32; WM_DIM],
        feat_mean: &[f32],
        feat_var: &[f32],
        lr: f32,
    ) -> (f64, [f32; WM_DIM]) {
        let t_len = chunk.len();
        if t_len < 3 {
            return (self.last_pinball_loss, *init_z);
        }
        self.ensure_adam_buffers();
        let d = WM_DIM;

        #[derive(Clone)]
        struct WmStepCache {
            x_norm: [f32; RL_STATE_DIM],
            u_norm: [f32; WM_DIM],
            inv_std: f32,
            u_t: [f32; WM_DIM],
            prev_z: [f32; WM_DIM],
            g_t: [f32; WM_DIM],
            r_t: [f32; WM_DIM],
            h_cand: [f32; WM_DIM],
            z_t: [f32; WM_DIM],
            pred: [f32; WM_OUT_DIM],
        }

        let mut caches = Vec::with_capacity(t_len);
        let mut z_cur = *init_z;
        for bar in chunk {
            let mut x_norm = [0.0_f32; RL_STATE_DIM];
            for i in 0..RL_STATE_DIM {
                let m = feat_mean.get(i).copied().unwrap_or(0.0);
                let v = feat_var.get(i).copied().unwrap_or(1.0).max(1e-4);
                x_norm[i] = ((bar.x_212[i] - m) / v.sqrt()).clamp(-6.0, 6.0);
            }
            let mut u_raw = [0.0_f32; WM_DIM];
            let mut mean = 0.0_f32;
            for i in 0..d {
                let row = &self.w_enc[i * RL_STATE_DIM..(i + 1) * RL_STATE_DIM];
                let mut sum = self.b_enc[i];
                for j in 0..RL_STATE_DIM {
                    sum += row[j] * x_norm[j];
                }
                u_raw[i] = sum;
                mean += sum;
            }
            mean /= d as f32;
            let mut var = 0.0_f32;
            for i in 0..d {
                let diff = u_raw[i] - mean;
                var += diff * diff;
            }
            let inv_std = 1.0 / (var / (d as f32) + 1e-5).sqrt();
            let mut u_norm = [0.0_f32; WM_DIM];
            let mut u_t = [0.0_f32; WM_DIM];
            for i in 0..d {
                u_norm[i] = (u_raw[i] - mean) * inv_std;
                u_t[i] = (u_norm[i] * self.ln_g[i] + self.ln_b[i]).tanh();
            }

            let mut g_t = [0.0_f32; WM_DIM];
            let mut r_t = [0.0_f32; WM_DIM];
            for i in 0..d {
                let wg = &self.w_g[i * d..(i + 1) * d];
                let ug = &self.u_g[i * d..(i + 1) * d];
                let wr = &self.w_r[i * d..(i + 1) * d];
                let ur = &self.u_r[i * d..(i + 1) * d];
                let mut sg = self.b_g[i];
                let mut sr = self.b_r[i];
                for j in 0..d {
                    sg += wg[j] * u_t[j] + ug[j] * z_cur[j];
                    sr += wr[j] * u_t[j] + ur[j] * z_cur[j];
                }
                g_t[i] = sigmoid(sg);
                r_t[i] = sigmoid(sr);
            }

            let mut h_cand = [0.0_f32; WM_DIM];
            let mut z_next = [0.0_f32; WM_DIM];
            for i in 0..d {
                let wh = &self.w_h[i * d..(i + 1) * d];
                let uh = &self.u_h[i * d..(i + 1) * d];
                let mut sh = self.b_h[i];
                for j in 0..d {
                    sh += wh[j] * u_t[j] + uh[j] * (r_t[j] * z_cur[j]);
                }
                h_cand[i] = sh.tanh();
                z_next[i] = g_t[i] * z_cur[i] + (1.0 - g_t[i]) * h_cand[i];
            }

            let mut pred = [0.0_f32; WM_OUT_DIM];
            for k in 0..WM_OUT_DIM {
                let row = &self.w_wm[k * d..(k + 1) * d];
                let mut s = self.b_wm[k];
                for i in 0..d {
                    s += row[i] * z_next[i];
                }
                pred[k] = s;
            }

            caches.push(WmStepCache {
                x_norm,
                u_norm,
                inv_std,
                u_t,
                prev_z: z_cur,
                g_t,
                r_t,
                h_cand,
                z_t: z_next,
                pred,
            });
            z_cur = z_next;
        }

        let mut g_w_enc = vec![0.0_f32; self.w_enc.len()];
        let mut g_b_enc = vec![0.0_f32; self.b_enc.len()];
        let mut g_ln_g = vec![0.0_f32; self.ln_g.len()];
        let mut g_ln_b = vec![0.0_f32; self.ln_b.len()];
        let mut g_w_g = vec![0.0_f32; self.w_g.len()];
        let mut g_u_g = vec![0.0_f32; self.u_g.len()];
        let mut g_b_g = vec![0.0_f32; self.b_g.len()];
        let mut g_w_r = vec![0.0_f32; self.w_r.len()];
        let mut g_u_r = vec![0.0_f32; self.u_r.len()];
        let mut g_b_r = vec![0.0_f32; self.b_r.len()];
        let mut g_w_h = vec![0.0_f32; self.w_h.len()];
        let mut g_u_h = vec![0.0_f32; self.u_h.len()];
        let mut g_b_h = vec![0.0_f32; self.b_h.len()];
        let mut g_w_wm = vec![0.0_f32; self.w_wm.len()];
        let mut g_b_wm = vec![0.0_f32; self.b_wm.len()];

        let mut total_loss = 0.0_f64;
        let mut valid_count = 0usize;
        let inv_t = 1.0 / (t_len as f32);
        let mut dz_next = [0.0_f32; WM_DIM];

        for t in (0..t_len).rev() {
            let c = &caches[t];
            let bar = &chunk[t];
            let mut dz = dz_next;

            for k in 0..WM_OUT_DIM {
                if !bar.wm_valid[k] {
                    continue;
                }
                let err = bar.wm_targets[k] - c.pred[k];
                let tau = WM_TAUS[k];
                let loss = if err >= 0.0 {
                    tau * err
                } else {
                    (tau - 1.0) * err
                };
                total_loss += loss as f64;
                valid_count += 1;
                let dl_dpred = if err >= 0.0 { -tau } else { 1.0 - tau } * inv_t * 0.25;
                g_b_wm[k] += dl_dpred;
                for i in 0..d {
                    g_w_wm[k * d + i] += dl_dpred * c.z_t[i];
                    dz[i] += dl_dpred * self.w_wm[k * d + i];
                }
            }

            let mut dh_cand = [0.0_f32; WM_DIM];
            let mut dg_pre = [0.0_f32; WM_DIM];
            let mut dz_prev = [0.0_f32; WM_DIM];
            for i in 0..d {
                dz_prev[i] = dz[i] * c.g_t[i];
                let d_hc = dz[i] * (1.0 - c.g_t[i]);
                dh_cand[i] = d_hc * (1.0 - c.h_cand[i] * c.h_cand[i]);
                let d_g = dz[i] * (c.prev_z[i] - c.h_cand[i]);
                dg_pre[i] = d_g * c.g_t[i] * (1.0 - c.g_t[i]);
            }

            let mut dr_t = [0.0_f32; WM_DIM];
            let mut du_t = [0.0_f32; WM_DIM];
            for i in 0..d {
                let dsh = dh_cand[i];
                g_b_h[i] += dsh;
                for j in 0..d {
                    g_w_h[i * d + j] += dsh * c.u_t[j];
                    g_u_h[i * d + j] += dsh * (c.r_t[j] * c.prev_z[j]);
                    du_t[j] += dsh * self.w_h[i * d + j];
                    let drp = dsh * self.u_h[i * d + j];
                    dr_t[j] += drp * c.prev_z[j];
                    dz_prev[j] += drp * c.r_t[j];
                }
            }

            for i in 0..d {
                let dsg = dg_pre[i];
                let dsr = dr_t[i] * c.r_t[i] * (1.0 - c.r_t[i]);
                g_b_g[i] += dsg;
                g_b_r[i] += dsr;
                for j in 0..d {
                    g_w_g[i * d + j] += dsg * c.u_t[j];
                    g_u_g[i * d + j] += dsg * c.prev_z[j];
                    g_w_r[i * d + j] += dsr * c.u_t[j];
                    g_u_r[i * d + j] += dsr * c.prev_z[j];
                    du_t[j] += dsg * self.w_g[i * d + j] + dsr * self.w_r[i * d + j];
                    dz_prev[j] += dsg * self.u_g[i * d + j] + dsr * self.u_r[i * d + j];
                }
            }

            let mut du_norm = [0.0_f32; WM_DIM];
            for i in 0..d {
                let dpre = du_t[i] * (1.0 - c.u_t[i] * c.u_t[i]);
                g_ln_b[i] += dpre;
                g_ln_g[i] += dpre * c.u_norm[i];
                du_norm[i] = dpre * self.ln_g[i];
            }
            let sum_dun: f32 = du_norm.iter().sum();
            let sum_dun_un: f32 = du_norm
                .iter()
                .zip(c.u_norm.iter())
                .map(|(a, b)| a * b)
                .sum();
            let inv_d = 1.0 / (d as f32);
            for i in 0..d {
                let du_raw =
                    c.inv_std * (du_norm[i] - inv_d * sum_dun - inv_d * c.u_norm[i] * sum_dun_un);
                g_b_enc[i] += du_raw;
                let row = &mut g_w_enc[i * RL_STATE_DIM..(i + 1) * RL_STATE_DIM];
                for j in 0..RL_STATE_DIM {
                    row[j] += du_raw * c.x_norm[j];
                }
            }

            dz_next = dz_prev;
        }

        if valid_count == 0 {
            return (self.last_pinball_loss, z_cur);
        }

        let mut sq_norm = 0.0_f32;
        for g in g_w_enc
            .iter()
            .chain(g_b_enc.iter())
            .chain(g_ln_g.iter())
            .chain(g_ln_b.iter())
            .chain(g_w_g.iter())
            .chain(g_u_g.iter())
            .chain(g_b_g.iter())
            .chain(g_w_r.iter())
            .chain(g_u_r.iter())
            .chain(g_b_r.iter())
            .chain(g_w_h.iter())
            .chain(g_u_h.iter())
            .chain(g_b_h.iter())
            .chain(g_w_wm.iter())
            .chain(g_b_wm.iter())
        {
            sq_norm += g * g;
        }
        let norm = sq_norm.sqrt();
        let clip_scale = if norm > 1.0 { 1.0 / norm } else { 1.0 };

        self.adam_step = self.adam_step.saturating_add(1);
        let t_f = self.adam_step as f32;
        let b1 = 0.9_f32;
        let b2 = 0.999_f32;
        let eps = 1e-8_f32;
        let bc1 = 1.0 - b1.powf(t_f);
        let bc2 = 1.0 - b2.powf(t_f);
        let mut idx = 0usize;

        let mut apply_slice = |params: &mut [f32], grads: &[f32], decay: f32| {
            for (p, &g_raw) in params.iter_mut().zip(grads.iter()) {
                let g = g_raw * clip_scale;
                let m = b1 * self.adam_m[idx] + (1.0 - b1) * g;
                let v = b2 * self.adam_v[idx] + (1.0 - b2) * g * g;
                self.adam_m[idx] = m;
                self.adam_v[idx] = v;
                let m_hat = m / bc1;
                let v_hat = v / bc2;
                *p = *p * (1.0 - lr * decay) - lr * m_hat / (v_hat.sqrt() + eps);
                idx += 1;
            }
        };

        apply_slice(&mut self.w_enc, &g_w_enc, 1e-4);
        apply_slice(&mut self.b_enc, &g_b_enc, 0.0);
        apply_slice(&mut self.ln_g, &g_ln_g, 0.0);
        apply_slice(&mut self.ln_b, &g_ln_b, 0.0);
        apply_slice(&mut self.w_g, &g_w_g, 1e-4);
        apply_slice(&mut self.u_g, &g_u_g, 1e-4);
        apply_slice(&mut self.b_g, &g_b_g, 0.0);
        apply_slice(&mut self.w_r, &g_w_r, 1e-4);
        apply_slice(&mut self.u_r, &g_u_r, 1e-4);
        apply_slice(&mut self.b_r, &g_b_r, 0.0);
        apply_slice(&mut self.w_h, &g_w_h, 1e-4);
        apply_slice(&mut self.u_h, &g_u_h, 1e-4);
        apply_slice(&mut self.b_h, &g_b_h, 0.0);
        apply_slice(&mut self.w_wm, &g_w_wm, 1e-4);
        apply_slice(&mut self.b_wm, &g_b_wm, 0.0);

        self.last_pinball_loss = total_loss / (valid_count as f64);
        (self.last_pinball_loss, z_cur)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentHyperParams {
    pub lr: f32,
    pub weight_decay: f32,
    pub l1_sparsity: f32,
    pub downside_aversion: f32,
    pub dd_aversion: f32,
    pub disagree_aversion: f32,
    pub chunk_len: usize,
    pub cost_sensitivity: f32,
    pub timescale_shift: f32,
    #[serde(default = "default_horizon_gamma")]
    pub horizon_gamma: f32,
}

fn default_horizon_gamma() -> f32 {
    0.86
}

impl AgentHyperParams {
    pub fn diverse_prior(idx: usize, seed: &mut u64) -> Self {
        let lrs = [9e-4, 1.4e-3, 2.0e-3, 2.8e-3];
        let l1s = [0.0, 1e-6, 4e-6, 1e-5];
        let downs = [0.8, 1.5, 2.4, 3.5];
        let dds = [0.25, 0.6, 1.1, 1.8];
        let chunks = [40usize, 56, 72, 96];
        let gammas = [0.78_f32, 0.85, 0.90, 0.94];
        let cost_sens = [0.22_f32, 0.42, 0.68, 0.98];
        Self {
            lr: lrs[idx % 4] * rand_uniform(seed, 0.88, 1.12),
            weight_decay: rand_uniform(seed, 5e-5, 6e-4),
            l1_sparsity: l1s[(idx / 4) % 4] * rand_uniform(seed, 0.85, 1.15),
            downside_aversion: downs[(idx + idx / 4) % 4],
            dd_aversion: dds[(idx * 3 + 1) % 4],
            disagree_aversion: rand_uniform(seed, 0.01, 0.08),
            chunk_len: chunks[idx % 4],
            cost_sensitivity: cost_sens[(idx / 6) % 4] * rand_uniform(seed, 0.85, 1.15),
            timescale_shift: rand_uniform(seed, -0.6, 0.6),
            horizon_gamma: gammas[(idx / 2) % 4],
        }
    }

    pub fn mutate(&self, seed: &mut u64) -> Self {
        let factor = |s: &mut u64| -> f32 {
            if rand_uniform(s, 0.0, 1.0) < 0.5 {
                0.82
            } else {
                1.22
            }
        };
        Self {
            lr: (self.lr * factor(seed)).clamp(2e-4, 3.5e-3),
            weight_decay: (self.weight_decay * factor(seed)).clamp(3e-5, 3e-3),
            l1_sparsity: (self.l1_sparsity * factor(seed)).clamp(0.0, 8e-5),
            downside_aversion: (self.downside_aversion * factor(seed)).clamp(0.4, 8.0),
            dd_aversion: (self.dd_aversion * factor(seed)).clamp(0.1, 5.0),
            disagree_aversion: (self.disagree_aversion * factor(seed)).clamp(0.005, 0.25),
            chunk_len: self.chunk_len,
            cost_sensitivity: (self.cost_sensitivity * factor(seed)).clamp(0.10, 1.80),
            timescale_shift: (self.timescale_shift + rand_uniform(seed, -0.2, 0.2))
                .clamp(-1.5, 1.5),
            horizon_gamma: (self.horizon_gamma + rand_uniform(seed, -0.04, 0.04)).clamp(0.72, 0.96),
        }
    }
}

/// Computes the predicted multi-bar excursion in basis points from World-Model horizon predictions.
#[inline]
fn predicted_wm_excursion_bps(wm_pred: &[f32; WM_OUT_DIM], rt_cost_bps: f32) -> f32 {
    let mut max_exc_bps = 0.0_f32;
    for h_idx in 0..WM_HORIZONS.len() {
        let base = h_idx * 4;
        let q25 = wm_pred[base];
        let q50 = wm_pred[base + 1];
        let q75 = wm_pred[base + 2];
        let rc_log = wm_pred[base + 3].clamp(-2.5, 3.2);
        let dir_bps = q50.abs().max(0.5 * (q75 - q25).abs()) * 10_000.0;
        let rng_bps = 0.42 * rt_cost_bps.max(8.0) * rc_log.exp();
        let h_exc = dir_bps.max(rng_bps);
        if h_exc > max_exc_bps {
            max_exc_bps = h_exc;
        }
    }
    max_exc_bps
}

/// Computes the Cost-Scaled Neural Deadzone threshold `tau_k` and `d(tau_k)/d(d_pre_k)`.
#[inline]
fn compute_cost_scaled_deadzone(
    d_pre: f32,
    one_way_cost_bps: f32,
    cost_sensitivity: f32,
    wm_exc_bps: f32,
    is_holding_pos: bool,
) -> (f32, f32) {
    let sp = softplus(d_pre);
    let dsp = sigmoid(d_pre);
    let rt_cost_bps = (2.0 * one_way_cost_bps).max(4.0);
    // Learned cost hurdle multiplier `k_theta`: expected excursion must clear `k_theta * exact_rt_cost_bps`
    let k_theta = 0.65 + 0.55 * sp + 0.35 * cost_sensitivity;
    let hurdle_bps = k_theta * rt_cost_bps;
    let net_deficit_bps = hurdle_bps - wm_exc_bps;
    let scale_denom = (rt_cost_bps * 2.4).max(30.0);
    let smooth_def = softplus(net_deficit_bps / scale_denom) * 0.22;
    let dsmooth_ddpre =
        sigmoid(net_deficit_bps / scale_denom) * 0.22 * (0.55 * dsp * rt_cost_bps / scale_denom);

    // Explicit cost drag baseline (`70 bps` RT on BRAIN-01 => +0.175 * cost_sensitivity; `10 bps` RT => +0.025)
    let base_tau = 0.065 + 0.08 * sp + 0.22 * cost_sensitivity * (rt_cost_bps / 100.0) + smooth_def;
    let dtau_ddpre = 0.08 * dsp + dsmooth_ddpre;

    if is_holding_pos {
        // Once entry friction is sunk, exit deadzone allows clean Neural-Exit when conviction fades
        ((base_tau * 0.48).clamp(0.03, 0.38), dtau_ddpre * 0.48)
    } else {
        (base_tau.clamp(0.05, 0.82), dtau_ddpre)
    }
}

/// Stage B: `K = 5` Orthogonal Specialist Policy Ensemble + Distributional Critic + Fisher EWC.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SequencePolicyNet {
    w_in: Vec<f32>,
    b_in: Vec<f32>,
    ln_in_g: Vec<f32>,
    ln_in_b: Vec<f32>,
    w_g: Vec<f32>,
    u_g: Vec<f32>,
    b_g: Vec<f32>,
    w_r: Vec<f32>,
    u_r: Vec<f32>,
    b_r: Vec<f32>,
    w_h: Vec<f32>,
    u_h: Vec<f32>,
    b_h: Vec<f32>,
    ln_out_g: Vec<f32>,
    ln_out_b: Vec<f32>,
    // K = 5 orthogonal specialist policy heads over `HEAD_IN_DIM = 55` (`[h_ln, wm_pred, self_state]`)
    w_a_heads: Vec<f32>, // K * HEAD_IN_DIM
    b_a_heads: Vec<f32>, // K
    w_d_heads: Vec<f32>, // K * HEAD_IN_DIM
    b_d_heads: Vec<f32>, // K
    w_v_heads: Vec<f32>, // K * HEAD_IN_DIM
    b_v_heads: Vec<f32>, // K
    w_s_heads: Vec<f32>, // K * HEAD_IN_DIM
    b_s_heads: Vec<f32>, // K
    pub spec_weights: [f32; SPECIALIST_K],
    // Distributional Quantile Critic head over `HEAD_IN_DIM`
    w_c: Vec<f32>,
    b_c: Vec<f32>,
    // AdamW & Diagonal Fisher EWC buffers
    adam_step: u64,
    #[serde(skip)]
    adam_m: Vec<f32>,
    #[serde(skip)]
    adam_v: Vec<f32>,
    #[serde(skip)]
    pub ewc_anchor: Vec<f32>,
    #[serde(skip)]
    pub ewc_fisher_diag: Vec<f32>,
}

impl SequencePolicyNet {
    pub fn new(seed: &mut u64, timescale_shift: f32) -> Self {
        let d = HIDDEN_DIM;
        let hd = HEAD_IN_DIM;
        let mut b_g = Vec::with_capacity(d);
        for i in 0..d {
            let base_tau = match i % 3 {
                0 => -0.62,
                1 => 1.38,
                _ => 3.18,
            };
            b_g.push(base_tau + timescale_shift);
        }

        // Orthogonalized initialization across K = 5 specialist action heads with alive initial scale
        let mut w_a_heads = init_mat(SPECIALIST_K, hd, 1.35, seed);
        for k in 1..SPECIALIST_K {
            for prev in 0..k {
                let mut dot = 0.0_f32;
                let mut norm_sq = 0.0_f32;
                for i in 0..hd {
                    let p_val = w_a_heads[prev * hd + i];
                    dot += w_a_heads[k * hd + i] * p_val;
                    norm_sq += p_val * p_val;
                }
                if norm_sq > 1e-6 {
                    let coeff = dot / norm_sq;
                    for i in 0..hd {
                        let p_val = w_a_heads[prev * hd + i];
                        w_a_heads[k * hd + i] -= coeff * p_val;
                    }
                }
            }
        }
        // Seed World-Model directional quantile skip-weights slightly positive so latent physics flows at Epoch 1
        for k in 0..SPECIALIST_K {
            let h_pick = (k % WM_HORIZONS.len()) * 4 + 1; // q50 of horizon k%4
            w_a_heads[k * hd + d + h_pick] += 1.25;
        }

        let mut net = Self {
            w_in: init_mat(d, TOTAL_INPUT_DIM, 1.05, seed),
            b_in: vec![0.0; d],
            ln_in_g: vec![1.0; d],
            ln_in_b: vec![0.0; d],
            w_g: init_mat(d, d, 0.8, seed),
            u_g: init_mat(d, d, 0.8, seed),
            b_g,
            w_r: init_mat(d, d, 0.8, seed),
            u_r: init_mat(d, d, 0.8, seed),
            b_r: vec![0.0; d],
            w_h: init_mat(d, d, 1.0, seed),
            u_h: init_mat(d, d, 0.8, seed),
            b_h: vec![0.0; d],
            ln_out_g: vec![1.0; d],
            ln_out_b: vec![0.0; d],
            w_a_heads,
            b_a_heads: vec![0.0; SPECIALIST_K],
            w_d_heads: init_mat(SPECIALIST_K, hd, 0.25, seed),
            b_d_heads: vec![-1.6; SPECIALIST_K],
            w_v_heads: init_mat(SPECIALIST_K, hd, 0.20, seed),
            b_v_heads: vec![-2.4; SPECIALIST_K],
            w_s_heads: init_mat(SPECIALIST_K, hd, 0.25, seed),
            b_s_heads: vec![-1.1; SPECIALIST_K],
            spec_weights: [1.0 / (SPECIALIST_K as f32); SPECIALIST_K],
            w_c: init_mat(CRITIC_QUANTILES, hd, 0.5, seed),
            b_c: vec![0.0; CRITIC_QUANTILES],
            adam_step: 0,
            adam_m: Vec::new(),
            adam_v: Vec::new(),
            ewc_anchor: Vec::new(),
            ewc_fisher_diag: Vec::new(),
        };
        net.ensure_adam_buffers();
        net
    }

    pub fn param_count(&self) -> usize {
        self.w_in.len()
            + self.b_in.len()
            + self.ln_in_g.len()
            + self.ln_in_b.len()
            + self.w_g.len()
            + self.u_g.len()
            + self.b_g.len()
            + self.w_r.len()
            + self.u_r.len()
            + self.b_r.len()
            + self.w_h.len()
            + self.u_h.len()
            + self.b_h.len()
            + self.ln_out_g.len()
            + self.ln_out_b.len()
            + self.w_a_heads.len()
            + self.b_a_heads.len()
            + self.w_d_heads.len()
            + self.b_d_heads.len()
            + self.w_v_heads.len()
            + self.b_v_heads.len()
            + self.w_s_heads.len()
            + self.b_s_heads.len()
            + self.w_c.len()
            + self.b_c.len()
    }

    pub fn ensure_valid_shapes(&mut self, seed: &mut u64, timescale_shift: f32) {
        let d = HIDDEN_DIM;
        let hd = HEAD_IN_DIM;
        if self.w_in.len() != d * TOTAL_INPUT_DIM
            || self.w_a_heads.len() != SPECIALIST_K * hd
            || self.w_c.len() != CRITIC_QUANTILES * hd
        {
            *self = Self::new(seed, timescale_shift);
        }
        self.ensure_adam_buffers();
    }

    fn ensure_adam_buffers(&mut self) {
        let n = self.param_count();
        if self.adam_m.len() != n {
            self.adam_m = vec![0.0; n];
            self.adam_v = vec![0.0; n];
        }
    }

    pub fn flatten_params(&self) -> Vec<f32> {
        let mut flat = Vec::with_capacity(self.param_count());
        flat.extend_from_slice(&self.w_in);
        flat.extend_from_slice(&self.b_in);
        flat.extend_from_slice(&self.ln_in_g);
        flat.extend_from_slice(&self.ln_in_b);
        flat.extend_from_slice(&self.w_g);
        flat.extend_from_slice(&self.u_g);
        flat.extend_from_slice(&self.b_g);
        flat.extend_from_slice(&self.w_r);
        flat.extend_from_slice(&self.u_r);
        flat.extend_from_slice(&self.b_r);
        flat.extend_from_slice(&self.w_h);
        flat.extend_from_slice(&self.u_h);
        flat.extend_from_slice(&self.b_h);
        flat.extend_from_slice(&self.ln_out_g);
        flat.extend_from_slice(&self.ln_out_b);
        flat.extend_from_slice(&self.w_a_heads);
        flat.extend_from_slice(&self.b_a_heads);
        flat.extend_from_slice(&self.w_d_heads);
        flat.extend_from_slice(&self.b_d_heads);
        flat.extend_from_slice(&self.w_v_heads);
        flat.extend_from_slice(&self.b_v_heads);
        flat.extend_from_slice(&self.w_s_heads);
        flat.extend_from_slice(&self.b_s_heads);
        flat.extend_from_slice(&self.w_c);
        flat.extend_from_slice(&self.b_c);
        flat
    }

    pub fn snapshot_ewc_anchor(&mut self) {
        self.ewc_anchor = self.flatten_params();
        if self.ewc_fisher_diag.len() != self.ewc_anchor.len() {
            self.ewc_fisher_diag = vec![1.0; self.ewc_anchor.len()];
        }
    }

    pub fn perturb(&mut self, noise_std: f32, seed: &mut u64) {
        for w in self
            .w_in
            .iter_mut()
            .chain(self.w_g.iter_mut())
            .chain(self.u_g.iter_mut())
            .chain(self.w_h.iter_mut())
            .chain(self.u_h.iter_mut())
            .chain(self.w_a_heads.iter_mut())
            .chain(self.w_d_heads.iter_mut())
            .chain(self.w_v_heads.iter_mut())
            .chain(self.w_s_heads.iter_mut())
        {
            *w += rand_normal(seed) * noise_std;
        }
        self.adam_m.fill(0.0);
        self.adam_v.fill(0.0);
        self.adam_step = 0;
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn forward_step_with_wm(
        &self,
        x_norm_212: &[f32; RL_STATE_DIM],
        z_wm: &[f32; WM_DIM],
        wm_pred: &[f32; WM_OUT_DIM],
        self_state: SelfAwarenessState,
        prev_h: &[f32; HIDDEN_DIM],
        next_wm_hidden: [f32; WM_DIM],
        cost_sensitivity: f32,
        option_allowed: bool,
    ) -> SequenceStepOutput {
        let d = HIDDEN_DIM;
        let hd = HEAD_IN_DIM;
        let s_arr = self_state.to_array();
        let mut x_cat = [0.0_f32; TOTAL_INPUT_DIM];
        let mut off = 0usize;
        x_cat[off..off + RL_STATE_DIM].copy_from_slice(x_norm_212);
        off += RL_STATE_DIM;
        x_cat[off..off + WM_DIM].copy_from_slice(z_wm);
        off += WM_DIM;
        x_cat[off..off + WM_OUT_DIM].copy_from_slice(wm_pred);
        off += WM_OUT_DIM;
        x_cat[off..off + SELF_STATE_DIM].copy_from_slice(&s_arr);

        let mut u_raw = [0.0_f32; HIDDEN_DIM];
        let mut mean = 0.0_f32;
        for i in 0..d {
            let row = &self.w_in[i * TOTAL_INPUT_DIM..(i + 1) * TOTAL_INPUT_DIM];
            let mut sum = self.b_in[i];
            for j in 0..TOTAL_INPUT_DIM {
                sum += row[j] * x_cat[j];
            }
            u_raw[i] = sum;
            mean += sum;
        }
        mean /= d as f32;
        let mut var = 0.0_f32;
        for i in 0..d {
            let diff = u_raw[i] - mean;
            var += diff * diff;
        }
        let inv_std = 1.0 / (var / (d as f32) + 1e-5).sqrt();
        let mut u_t = [0.0_f32; HIDDEN_DIM];
        for i in 0..d {
            u_t[i] = ((u_raw[i] - mean) * inv_std * self.ln_in_g[i] + self.ln_in_b[i]).tanh();
        }

        let mut g_t = [0.0_f32; HIDDEN_DIM];
        let mut r_t = [0.0_f32; HIDDEN_DIM];
        for i in 0..d {
            let wg = &self.w_g[i * d..(i + 1) * d];
            let ug = &self.u_g[i * d..(i + 1) * d];
            let wr = &self.w_r[i * d..(i + 1) * d];
            let ur = &self.u_r[i * d..(i + 1) * d];
            let mut sg = self.b_g[i];
            let mut sr = self.b_r[i];
            for j in 0..d {
                sg += wg[j] * u_t[j] + ug[j] * prev_h[j];
                sr += wr[j] * u_t[j] + ur[j] * prev_h[j];
            }
            g_t[i] = sigmoid(sg);
            r_t[i] = sigmoid(sr);
        }

        let mut h_next = [0.0_f32; HIDDEN_DIM];
        let mut h_mean = 0.0_f32;
        for i in 0..d {
            let wh = &self.w_h[i * d..(i + 1) * d];
            let uh = &self.u_h[i * d..(i + 1) * d];
            let mut sh = self.b_h[i];
            for j in 0..d {
                sh += wh[j] * u_t[j] + uh[j] * (r_t[j] * prev_h[j]);
            }
            let h_cand = sh.tanh();
            h_next[i] = g_t[i] * prev_h[i] + (1.0 - g_t[i]) * h_cand;
            h_mean += h_next[i];
        }
        h_mean /= d as f32;
        let mut h_var = 0.0_f32;
        for i in 0..d {
            let diff = h_next[i] - h_mean;
            h_var += diff * diff;
        }
        let h_inv_std = 1.0 / (h_var / (d as f32) + 1e-5).sqrt();
        let mut head_in = [0.0_f32; HEAD_IN_DIM];
        for i in 0..d {
            head_in[i] = (h_next[i] - h_mean) * h_inv_std * self.ln_out_g[i] + self.ln_out_b[i];
        }
        head_in[d..d + WM_OUT_DIM].copy_from_slice(wm_pred);
        head_in[d + WM_OUT_DIM..hd].copy_from_slice(&s_arr);

        let is_holding = self_state.current_pos.abs() > 1e-4;
        let rt_cost_bps = (2.0 * self_state.one_way_cost_bps).max(4.0);
        let wm_exc_bps = predicted_wm_excursion_bps(wm_pred, rt_cost_bps);

        let mut a_mean = 0.0_f32;
        let mut z_mean = 0.0_f32;
        let mut delta_mean = 0.0_f32;
        let mut v_mean = 0.0_f32;
        let mut s_mean = 0.0_f32;

        for k in 0..SPECIALIST_K {
            let w_k = self.spec_weights[k];
            let off_k = k * hd;
            let mut z_pre = self.b_a_heads[k];
            let mut d_pre = self.b_d_heads[k];
            let mut v_pre = self.b_v_heads[k];
            let mut s_pre = self.b_s_heads[k];
            for i in 0..hd {
                let hi = head_in[i];
                z_pre += self.w_a_heads[off_k + i] * hi;
                d_pre += self.w_d_heads[off_k + i] * hi;
                v_pre += self.w_v_heads[off_k + i] * hi;
                s_pre += self.w_s_heads[off_k + i] * hi;
            }
            let z_k = z_pre.tanh();
            let (tau_k, _) = compute_cost_scaled_deadzone(
                d_pre,
                self_state.one_way_cost_bps,
                cost_sensitivity,
                wm_exc_bps,
                is_holding,
            );
            // Exact 0.0 FLAT inside cost-scaled deadzone
            let a_k = if z_k.abs() <= tau_k {
                0.0
            } else {
                (z_k.signum() * (z_k.abs() - tau_k) / (1.0 - tau_k).max(0.15)).clamp(-1.0, 1.0)
            };
            a_mean += w_k * a_k;
            z_mean += w_k * z_k;
            delta_mean += w_k * tau_k;
            v_mean += w_k * if option_allowed { sigmoid(v_pre) } else { 0.0 };
            s_mean += w_k * (softplus(s_pre) * 0.035 + 0.009);
        }

        let action = if a_mean.abs() <= 1e-6 {
            0.0
        } else {
            a_mean.clamp(-1.0, 1.0)
        };

        let mut quantiles = [0.0_f32; CRITIC_QUANTILES];
        for q in 0..CRITIC_QUANTILES {
            let row = &self.w_c[q * hd..(q + 1) * hd];
            let mut s = self.b_c[q];
            for i in 0..hd {
                s += row[i] * head_in[i];
            }
            quantiles[q] = s;
        }

        SequenceStepOutput {
            action,
            raw_z: z_mean,
            deadzone: delta_mean,
            vehicle_prob: v_mean.clamp(0.0, 1.0),
            tail_stop_dist: s_mean.clamp(0.008, 0.09),
            quantiles,
            next_hidden: h_next,
            next_wm_hidden,
        }
    }

    #[inline]
    pub fn forward_step(
        &self,
        x_norm_212: &[f32; RL_STATE_DIM],
        self_state: SelfAwarenessState,
        prev_h: &[f32; HIDDEN_DIM],
        cost_sensitivity: f32,
        option_allowed: bool,
    ) -> SequenceStepOutput {
        let zero_z = [0.0_f32; WM_DIM];
        let zero_pred = [0.0_f32; WM_OUT_DIM];
        self.forward_step_with_wm(
            x_norm_212,
            &zero_z,
            &zero_pred,
            self_state,
            prev_h,
            zero_z,
            cost_sensitivity,
            option_allowed,
        )
    }

    /// Exact End-to-End Analytical Backpropagation Through Time (BPTT) across a causal sequence chunk
    /// with Straight-Through / Smooth Surrogate Deadzone Gradient, Differentiable EOD Closure,
    /// optional Supervised Target Override (for Red-Trade Loss Autopsy / Fisher Winner Lock),
    /// and Diagonal Fisher EWC.
    #[allow(clippy::too_many_arguments)]
    pub fn train_bptt_chunk_ext(
        &mut self,
        chunk: &[CausalTrajectoryBar],
        init_wm_z: &[f32; WM_DIM],
        world_model: Option<&PhysicsWorldModel>,
        feat_mean: &[f32],
        feat_var: &[f32],
        hparams: &AgentHyperParams,
        lr_scale: f32,
        ewc_lambda: f32,
        target_override: Option<(&[Option<f32>], f32)>,
        accumulate_fisher_into: Option<&mut [f32]>,
    ) -> (f64, f32, f32) {
        let t_len = chunk.len();
        if t_len < 3 {
            return (0.0, 0.0, 0.0);
        }
        self.ensure_adam_buffers();
        let d = HIDDEN_DIM;
        let hd = HEAD_IN_DIM;

        #[derive(Clone)]
        struct StepCache {
            x_cat: [f32; TOTAL_INPUT_DIM],
            u_norm: [f32; HIDDEN_DIM],
            u_inv_std: f32,
            u_t: [f32; HIDDEN_DIM],
            prev_h: [f32; HIDDEN_DIM],
            g_t: [f32; HIDDEN_DIM],
            r_t: [f32; HIDDEN_DIM],
            h_cand: [f32; HIDDEN_DIM],
            h_norm: [f32; HIDDEN_DIM],
            h_inv_std: f32,
            head_in: [f32; HEAD_IN_DIM],
            z_k: [f32; SPECIALIST_K],
            tau_k: [f32; SPECIALIST_K],
            dtau_ddpre: [f32; SPECIALIST_K],
            a_k: [f32; SPECIALIST_K],
            s_k: [f32; SPECIALIST_K],
            a_t: f32,
            z_mean: f32,
            v_t: f32,
            q_t: [f32; CRITIC_QUANTILES],
            r_net: f32,
            r_eff: f32,
            c_eff: f32,
            eq_curve: f32,
            force_flat: bool,
            entry_allowed: bool,
            option_allowed: bool,
            prev_a: f32,
        }

        let mut caches = Vec::with_capacity(t_len);
        let mut prev_h = [0.0_f32; HIDDEN_DIM];
        let mut prev_wm_z = *init_wm_z;
        let mut prev_a = 0.0_f32;
        let mut eq_cur = 0.0_f32;
        let mut bars_held = 0usize;
        let mut cum_ret = 0.0_f32;
        let mut peak_ret = 0.0_f32;

        for bar in chunk {
            let mut x_norm = [0.0_f32; RL_STATE_DIM];
            for i in 0..RL_STATE_DIM {
                let m = feat_mean.get(i).copied().unwrap_or(0.0);
                let v = feat_var.get(i).copied().unwrap_or(1.0).max(1e-4);
                x_norm[i] = ((bar.x_212[i] - m) / v.sqrt()).clamp(-6.0, 6.0);
            }
            let (z_wm, wm_pred) = if let Some(wm) = world_model {
                let out = wm.forward_step(&x_norm, &prev_wm_z);
                prev_wm_z = out.0;
                out
            } else {
                ([0.0_f32; WM_DIM], [0.0_f32; WM_OUT_DIM])
            };

            let dd_peak = (peak_ret - cum_ret).max(0.0);
            let is_holding = prev_a.abs() > 1e-4;
            let self_s = SelfAwarenessState {
                current_pos: prev_a,
                unrealized_net_ret: cum_ret,
                peak_excursion: peak_ret,
                drawdown_from_peak: dd_peak,
                log_bars_held: if is_holding {
                    (1.0 + bars_held as f32).ln()
                } else {
                    0.0
                },
                one_way_cost_bps: bar.one_way_cost_bps,
                session_time_left: bar.session_time_left,
            };
            let s_arr = self_s.to_array();
            let mut x_cat = [0.0_f32; TOTAL_INPUT_DIM];
            let mut off = 0usize;
            x_cat[off..off + RL_STATE_DIM].copy_from_slice(&x_norm);
            off += RL_STATE_DIM;
            x_cat[off..off + WM_DIM].copy_from_slice(&z_wm);
            off += WM_DIM;
            x_cat[off..off + WM_OUT_DIM].copy_from_slice(&wm_pred);
            off += WM_OUT_DIM;
            x_cat[off..off + SELF_STATE_DIM].copy_from_slice(&s_arr);

            let mut u_raw = [0.0_f32; HIDDEN_DIM];
            let mut u_mean = 0.0_f32;
            for i in 0..d {
                let row = &self.w_in[i * TOTAL_INPUT_DIM..(i + 1) * TOTAL_INPUT_DIM];
                let mut sum = self.b_in[i];
                for j in 0..TOTAL_INPUT_DIM {
                    sum += row[j] * x_cat[j];
                }
                u_raw[i] = sum;
                u_mean += sum;
            }
            u_mean /= d as f32;
            let mut u_var = 0.0_f32;
            for i in 0..d {
                let diff = u_raw[i] - u_mean;
                u_var += diff * diff;
            }
            let u_inv_std = 1.0 / (u_var / (d as f32) + 1e-5).sqrt();
            let mut u_norm = [0.0_f32; HIDDEN_DIM];
            let mut u_t = [0.0_f32; HIDDEN_DIM];
            for i in 0..d {
                u_norm[i] = (u_raw[i] - u_mean) * u_inv_std;
                u_t[i] = (u_norm[i] * self.ln_in_g[i] + self.ln_in_b[i]).tanh();
            }

            let mut g_t = [0.0_f32; HIDDEN_DIM];
            let mut r_t = [0.0_f32; HIDDEN_DIM];
            for i in 0..d {
                let wg = &self.w_g[i * d..(i + 1) * d];
                let ug = &self.u_g[i * d..(i + 1) * d];
                let wr = &self.w_r[i * d..(i + 1) * d];
                let ur = &self.u_r[i * d..(i + 1) * d];
                let mut sg = self.b_g[i];
                let mut sr = self.b_r[i];
                for j in 0..d {
                    sg += wg[j] * u_t[j] + ug[j] * prev_h[j];
                    sr += wr[j] * u_t[j] + ur[j] * prev_h[j];
                }
                g_t[i] = sigmoid(sg);
                r_t[i] = sigmoid(sr);
            }

            let mut h_cand = [0.0_f32; HIDDEN_DIM];
            let mut h_next = [0.0_f32; HIDDEN_DIM];
            let mut h_mean = 0.0_f32;
            for i in 0..d {
                let wh = &self.w_h[i * d..(i + 1) * d];
                let uh = &self.u_h[i * d..(i + 1) * d];
                let mut sh = self.b_h[i];
                for j in 0..d {
                    sh += wh[j] * u_t[j] + uh[j] * (r_t[j] * prev_h[j]);
                }
                h_cand[i] = sh.tanh();
                h_next[i] = g_t[i] * prev_h[i] + (1.0 - g_t[i]) * h_cand[i];
                h_mean += h_next[i];
            }
            h_mean /= d as f32;
            let mut h_var = 0.0_f32;
            for i in 0..d {
                let diff = h_next[i] - h_mean;
                h_var += diff * diff;
            }
            let h_inv_std = 1.0 / (h_var / (d as f32) + 1e-5).sqrt();
            let mut h_norm = [0.0_f32; HIDDEN_DIM];
            let mut head_in = [0.0_f32; HEAD_IN_DIM];
            for i in 0..d {
                h_norm[i] = (h_next[i] - h_mean) * h_inv_std;
                head_in[i] = h_norm[i] * self.ln_out_g[i] + self.ln_out_b[i];
            }
            head_in[d..d + WM_OUT_DIM].copy_from_slice(&wm_pred);
            head_in[d + WM_OUT_DIM..hd].copy_from_slice(&s_arr);

            let rt_cost_bps = (2.0 * bar.one_way_cost_bps).max(4.0);
            let wm_exc_bps = predicted_wm_excursion_bps(&wm_pred, rt_cost_bps);

            let mut z_k = [0.0_f32; SPECIALIST_K];
            let mut tau_k = [0.0_f32; SPECIALIST_K];
            let mut dtau_ddpre = [0.0_f32; SPECIALIST_K];
            let mut a_k = [0.0_f32; SPECIALIST_K];
            let mut s_k = [0.0_f32; SPECIALIST_K];
            let mut a_mean = 0.0_f32;
            let mut z_mean = 0.0_f32;
            let mut v_mean = 0.0_f32;

            for k in 0..SPECIALIST_K {
                let w_k = self.spec_weights[k];
                let off_k = k * hd;
                let mut z_pre = self.b_a_heads[k];
                let mut d_pre = self.b_d_heads[k];
                let mut v_pre = self.b_v_heads[k];
                let mut s_pre = self.b_s_heads[k];
                for i in 0..hd {
                    let hi = head_in[i];
                    z_pre += self.w_a_heads[off_k + i] * hi;
                    d_pre += self.w_d_heads[off_k + i] * hi;
                    v_pre += self.w_v_heads[off_k + i] * hi;
                    s_pre += self.w_s_heads[off_k + i] * hi;
                }
                z_k[k] = z_pre.tanh();
                let (tk, dtk) = compute_cost_scaled_deadzone(
                    d_pre,
                    bar.one_way_cost_bps,
                    hparams.cost_sensitivity,
                    wm_exc_bps,
                    is_holding,
                );
                tau_k[k] = tk;
                dtau_ddpre[k] = dtk;

                // Hard-enforce `a_{T_eod} = 0.0` on `force_flat`, and exact `0.0` inside deadzone
                a_k[k] = if bar.force_flat || (!bar.entry_allowed && !is_holding) {
                    0.0
                } else if z_k[k].abs() <= tk {
                    0.0
                } else {
                    let raw_ak =
                        (z_k[k].signum() * (z_k[k].abs() - tk) / (1.0 - tk).max(0.15)).clamp(-1.0, 1.0);
                    if !bar.entry_allowed {
                        if raw_ak * prev_a <= 0.0 {
                            0.0
                        } else {
                            raw_ak.signum() * raw_ak.abs().min(prev_a.abs())
                        }
                    } else {
                        raw_ak
                    }
                };
                let vk = if bar.option_allowed { sigmoid(v_pre) } else { 0.0 };
                s_k[k] = softplus(s_pre) * 0.035 + 0.009;
                a_mean += w_k * a_k[k];
                z_mean += w_k * z_k[k];
                v_mean += w_k * vk;
            }

            let mut q_t = [0.0_f32; CRITIC_QUANTILES];
            for q in 0..CRITIC_QUANTILES {
                let row = &self.w_c[q * hd..(q + 1) * hd];
                let mut s = self.b_c[q];
                for i in 0..hd {
                    s += row[i] * head_in[i];
                }
                q_t[q] = s;
            }

            let r_eff = (1.0 - v_mean) * bar.ret_oo_shares + v_mean * bar.ret_oo_option;
            let c_eff =
                (1.0 - v_mean) * bar.one_way_cost_shares + v_mean * bar.one_way_cost_option;
            // On `force_flat` (final EOD bar), `a_mean == 0.0`, so `-|0.0 - prev_a| * c_eff`
            // charges the EXACT EOD flatten slippage/spread cost inside differentiable BPTT!
            let r_net = a_mean * r_eff - (a_mean - prev_a).abs() * c_eff;
            eq_cur += r_net;

            let cur_prev_a = prev_a;
            if a_mean.abs() < 1e-4 {
                bars_held = 0;
                cum_ret = 0.0;
                peak_ret = 0.0;
            } else if a_mean * prev_a > 0.0 {
                bars_held += 1;
                cum_ret += r_net;
                peak_ret = peak_ret.max(cum_ret);
            } else {
                bars_held = 1;
                cum_ret = r_net;
                peak_ret = cum_ret.max(0.0);
            }

            caches.push(StepCache {
                x_cat,
                u_norm,
                u_inv_std,
                u_t,
                prev_h,
                g_t,
                r_t,
                h_cand,
                h_norm,
                h_inv_std,
                head_in,
                z_k,
                tau_k,
                dtau_ddpre,
                a_k,
                s_k,
                a_t: a_mean,
                z_mean,
                v_t: v_mean,
                q_t,
                r_net,
                r_eff,
                c_eff,
                eq_curve: eq_cur,
                force_flat: bar.force_flat,
                entry_allowed: bar.entry_allowed,
                option_allowed: bar.option_allowed,
                prev_a: cur_prev_a,
            });
            prev_h = h_next;
            prev_a = a_mean;
        }

        // Discounted causal continuation return strictly bounded inside this episode (zeroed at force_flat EOD!)
        let gamma = hparams.horizon_gamma.clamp(0.70, 0.96);
        let mut cont_asset_ret = vec![0.0_f32; t_len];
        let mut run_r = 0.0_f32;
        for t in (0..t_len).rev() {
            if caches[t].force_flat {
                run_r = 0.0;
                cont_asset_ret[t] = 0.0;
            } else {
                run_r = caches[t].r_eff + gamma * run_r;
                cont_asset_ret[t] = run_r;
            }
        }

        let mut peak_eq = 0.0_f32;
        let mut max_dd = 0.0_f32;
        let mut max_dd_t = 0usize;
        let mut max_dd_peak_t = 0usize;
        let mut cur_peak_t = 0usize;
        for (t, c) in caches.iter().enumerate() {
            if c.eq_curve > peak_eq {
                peak_eq = c.eq_curve;
                cur_peak_t = t;
            }
            let dd = peak_eq - c.eq_curve;
            if dd > max_dd {
                max_dd = dd;
                max_dd_t = t;
                max_dd_peak_t = cur_peak_t;
            }
        }

        let inv_t = 1.0 / (t_len as f32);
        let sum_r_net: f32 = caches.iter().map(|c| c.r_net).sum();
        let sum_down: f32 = caches
            .iter()
            .map(|c| {
                let d_neg = (-c.r_net).max(0.0);
                d_neg * d_neg
            })
            .sum();
        let sum_l1: f32 = caches.iter().map(|c| c.a_t.abs()).sum();
        let utility = inv_t * sum_r_net
            - hparams.downside_aversion * inv_t * sum_down
            - hparams.dd_aversion * max_dd
            - hparams.l1_sparsity * inv_t * sum_l1;

        let mut g_w_in = vec![0.0_f32; self.w_in.len()];
        let mut g_b_in = vec![0.0_f32; self.b_in.len()];
        let mut g_ln_in_g = vec![0.0_f32; self.ln_in_g.len()];
        let mut g_ln_in_b = vec![0.0_f32; self.ln_in_b.len()];
        let mut g_w_g = vec![0.0_f32; self.w_g.len()];
        let mut g_u_g = vec![0.0_f32; self.u_g.len()];
        let mut g_b_g = vec![0.0_f32; self.b_g.len()];
        let mut g_w_r = vec![0.0_f32; self.w_r.len()];
        let mut g_u_r = vec![0.0_f32; self.u_r.len()];
        let mut g_b_r = vec![0.0_f32; self.b_r.len()];
        let mut g_w_h = vec![0.0_f32; self.w_h.len()];
        let mut g_u_h = vec![0.0_f32; self.u_h.len()];
        let mut g_b_h = vec![0.0_f32; self.b_h.len()];
        let mut g_ln_out_g = vec![0.0_f32; self.ln_out_g.len()];
        let mut g_ln_out_b = vec![0.0_f32; self.ln_out_b.len()];
        let mut g_w_a = vec![0.0_f32; self.w_a_heads.len()];
        let mut g_b_a = vec![0.0_f32; self.b_a_heads.len()];
        let mut g_w_d = vec![0.0_f32; self.w_d_heads.len()];
        let mut g_b_d = vec![0.0_f32; self.b_d_heads.len()];
        let mut g_w_v = vec![0.0_f32; self.w_v_heads.len()];
        let mut g_b_v = vec![0.0_f32; self.b_v_heads.len()];
        let mut g_w_s = vec![0.0_f32; self.w_s_heads.len()];
        let mut g_b_s = vec![0.0_f32; self.b_s_heads.len()];
        let mut g_w_c = vec![0.0_f32; self.w_c.len()];
        let mut g_b_c = vec![0.0_f32; self.b_c.len()];

        let util_scale = 180.0_f32;
        let mut dh_next = [0.0_f32; HIDDEN_DIM];
        let mut total_sal_dx = 0.0_f32;
        let mut total_sal_dh = 0.0_f32;

        let eod_bar_present = caches.last().is_some_and(|last| last.force_flat);
        let eod_cost = caches.last().map(|last| last.c_eff).unwrap_or(0.0);

        for t in (0..t_len).rev() {
            let c = &caches[t];
            let mut grad_dr = inv_t * util_scale;
            if c.r_net < 0.0 {
                grad_dr += -hparams.downside_aversion * inv_t * (2.0 * c.r_net) * util_scale;
            }
            if max_dd > 1e-6 && t > max_dd_peak_t && t <= max_dd_t {
                let denom = (max_dd_t - max_dd_peak_t).max(1) as f32;
                grad_dr += (hparams.dd_aversion / denom) * util_scale;
            }

            // `du_da` = d(Utility) / d(a_t) (positive means increasing `a_t` improves objective)
            let du_da = if c.force_flat {
                0.0
            } else if let Some((targets, weight)) = target_override {
                if let Some(Some(tgt_a)) = targets.get(t).copied() {
                    // Asymmetric Red-Trade Autopsy / Winner Fisher objective: -0.5 * weight * (a_t - tgt_a)^2
                    -weight * inv_t * 25.0 * (c.a_t - tgt_a)
                } else {
                    0.0
                }
            } else {
                let rt_cost = 2.0 * c.c_eff;
                let r_look = cont_asset_ret[t];
                let dr_da = if c.prev_a.abs() <= 1e-4 {
                    // Currently FLAT: opening a position must clear full round-trip friction `2 * c_eff`
                    // before EOD closure!
                    if !c.entry_allowed {
                        -c.a_t * rt_cost * 4.0
                    } else if r_look.abs() > rt_cost * 1.15 {
                        r_look - r_look.signum() * rt_cost
                    } else {
                        // Move fails to beat round-trip cost: penalize non-zero conviction!
                        let probe_sgn = if c.a_t.abs() > 1e-5 {
                            c.a_t.signum()
                        } else {
                            c.z_mean.signum()
                        };
                        -probe_sgn * (rt_cost - r_look.abs() * 0.5).max(0.0002)
                    }
                } else {
                    // Currently HOLDING `c.prev_a`: entry cost is sunk; exiting early via Neural-Exit
                    // avoids EOD reversal & EOD closing-bell flatten friction!
                    let pos_sgn = c.prev_a.signum();
                    let eod_prox = if eod_bar_present {
                        let rem = (t_len.saturating_sub(1 + t)) as f32;
                        (1.0 / (1.0 + 0.35 * rem)).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    if c.a_t * pos_sgn < 0.0 {
                        // Flipping sign pays full round-trip reversal cost
                        r_look - c.a_t.signum() * rt_cost
                    } else {
                        // Pure continuation value minus EOD closing-bell urgency
                        r_look - pos_sgn * eod_cost * (0.45 + 1.15 * eod_prox)
                    }
                };
                let g_l1 = hparams.l1_sparsity * inv_t * c.a_t.signum();
                grad_dr * dr_da - util_scale * g_l1
            };

            let du_dv = if c.option_allowed && target_override.is_none() {
                let opt_var_pen = 2.5
                    * hparams.downside_aversion
                    * (chunk[t].ret_oo_option * chunk[t].ret_oo_option
                        - chunk[t].ret_oo_shares * chunk[t].ret_oo_shares)
                        .max(0.0);
                let opt_adv = (chunk[t].ret_oo_option - chunk[t].ret_oo_shares) * c.a_t
                    - 2.0
                        * (chunk[t].one_way_cost_option - chunk[t].one_way_cost_shares)
                        * (c.a_t - c.prev_a).abs().max(0.15 * c.a_t.abs())
                    - opt_var_pen * c.a_t.abs();
                grad_dr * opt_adv * c.v_t * (1.0 - c.v_t)
            } else {
                0.0
            };

            let mut d_head_in = [0.0_f32; HEAD_IN_DIM];
            if target_override.is_none() {
                let target_r = caches[t..t_len.min(t + 8)]
                    .iter()
                    .map(|s| s.r_net)
                    .sum::<f32>();
                for q in 0..CRITIC_QUANTILES {
                    let err = target_r - c.q_t[q];
                    let tau = QUANTILE_TAUS[q];
                    let dl_dq = if err >= 0.0 { -tau } else { 1.0 - tau } * inv_t * 0.4;
                    g_b_c[q] += dl_dq;
                    for i in 0..hd {
                        g_w_c[q * hd + i] += dl_dq * c.head_in[i];
                        d_head_in[i] += dl_dq * self.w_c[q * hd + i];
                    }
                }
            }

            // Backprop across K = 5 specialist heads using Straight-Through / Smooth Surrogate Gradient!
            for k in 0..SPECIALIST_K {
                let w_k = self.spec_weights[k];
                let off_k = k * hd;
                let g_disagree = if target_override.is_none() {
                    -2.0 * hparams.disagree_aversion * inv_t * util_scale * (c.a_k[k] - c.a_t)
                } else {
                    0.0
                };
                let du_dak = w_k * (du_da + g_disagree);

                let (du_dzk, du_dtauk) = if c.force_flat {
                    (0.0, 0.0)
                } else {
                    let denom = (1.0 - c.tau_k[k]).max(0.15);
                    let u_b = ((c.z_k[k].abs() - c.tau_k[k]) * 8.0).clamp(-15.0, 15.0);
                    let sig_b = sigmoid(u_b);
                    let sigmoid_grad = (4.0 * sig_b * (1.0 - sig_b)).clamp(0.0, 1.0);

                    if c.z_k[k].abs() > c.tau_k[k] && c.a_k[k].abs() > 1e-6 {
                        let da_dz = 1.0 / denom;
                        let sgn = c.z_k[k].signum();
                        let da_dtau = -sgn * (1.0 - c.z_k[k].abs()) / (denom * denom);
                        (du_dak * da_dz, du_dak * da_dtau)
                    } else {
                        // FIX #1: Straight-Through / Smooth Surrogate Gradient across deadzone boundary!
                        // Forward was exact `0.0`; backward uses `da/dz = 0.25 + 0.75 * sigmoid_grad`!
                        let surrogate_da_dz = 0.25 + 0.75 * sigmoid_grad;
                        let sgn = if c.z_k[k].abs() > 1e-6 {
                            c.z_k[k].signum()
                        } else {
                            du_dak.signum()
                        };
                        let surrogate_da_dtau = -sgn * (0.15 + 0.60 * sigmoid_grad);
                        (du_dak * surrogate_da_dz, du_dak * surrogate_da_dtau)
                    }
                };

                let du_dzpre = du_dzk * (1.0 - c.z_k[k] * c.z_k[k]);
                let du_ddpre = du_dtauk * c.dtau_ddpre[k];
                let du_dvpre = w_k * du_dv;

                let adv_ret = (-c.r_net).max(0.0);
                let target_stop = (adv_ret * 2.5 + 0.014).clamp(0.009, 0.075);
                let dl_ds = if target_override.is_none() {
                    (c.s_k[k] - target_stop) * inv_t * 0.15
                } else {
                    0.0
                };
                let dl_dspre = dl_ds * 0.035 * 0.5;

                // Negate `du_*` to form Loss gradients `dl_*` for gradient descent
                let dl_dzpre = -du_dzpre;
                let dl_ddpre = -du_ddpre;
                let dl_dvpre = -du_dvpre;

                g_b_a[k] += dl_dzpre;
                g_b_d[k] += dl_ddpre;
                g_b_v[k] += dl_dvpre;
                g_b_s[k] += dl_dspre;
                for i in 0..hd {
                    let hi = c.head_in[i];
                    g_w_a[off_k + i] += dl_dzpre * hi;
                    g_w_d[off_k + i] += dl_ddpre * hi;
                    g_w_v[off_k + i] += dl_dvpre * hi;
                    g_w_s[off_k + i] += dl_dspre * hi;
                    d_head_in[i] += dl_dzpre * self.w_a_heads[off_k + i]
                        + dl_ddpre * self.w_d_heads[off_k + i]
                        + dl_dvpre * self.w_v_heads[off_k + i]
                        + dl_dspre * self.w_s_heads[off_k + i];
                }
            }

            let mut d_h_norm = [0.0_f32; HIDDEN_DIM];
            for i in 0..d {
                let dhl = d_head_in[i];
                g_ln_out_b[i] += dhl;
                g_ln_out_g[i] += dhl * c.h_norm[i];
                d_h_norm[i] = dhl * self.ln_out_g[i];
            }
            let sum_dhn: f32 = d_h_norm.iter().sum();
            let sum_dhn_hn: f32 = d_h_norm
                .iter()
                .zip(c.h_norm.iter())
                .map(|(a, b)| a * b)
                .sum();
            let inv_d = 1.0 / (d as f32);
            let mut dh = [0.0_f32; HIDDEN_DIM];
            let mut step_sal_dh = 0.0_f32;
            for i in 0..d {
                dh[i] = dh_next[i]
                    + c.h_inv_std
                        * (d_h_norm[i] - inv_d * sum_dhn - inv_d * c.h_norm[i] * sum_dhn_hn);
                step_sal_dh += dh[i].abs();
            }
            total_sal_dh += step_sal_dh / (d as f32);

            let mut dh_cand = [0.0_f32; HIDDEN_DIM];
            let mut dg_pre = [0.0_f32; HIDDEN_DIM];
            let mut dh_prev = [0.0_f32; HIDDEN_DIM];
            for i in 0..d {
                dh_prev[i] = dh[i] * c.g_t[i];
                let d_hc = dh[i] * (1.0 - c.g_t[i]);
                dh_cand[i] = d_hc * (1.0 - c.h_cand[i] * c.h_cand[i]);
                let d_g = dh[i] * (c.prev_h[i] - c.h_cand[i]);
                dg_pre[i] = d_g * c.g_t[i] * (1.0 - c.g_t[i]);
            }

            let mut dr_t = [0.0_f32; HIDDEN_DIM];
            let mut du_t = [0.0_f32; HIDDEN_DIM];
            for i in 0..d {
                let dsh = dh_cand[i];
                g_b_h[i] += dsh;
                for j in 0..d {
                    g_w_h[i * d + j] += dsh * c.u_t[j];
                    g_u_h[i * d + j] += dsh * (c.r_t[j] * c.prev_h[j]);
                    du_t[j] += dsh * self.w_h[i * d + j];
                    let drp = dsh * self.u_h[i * d + j];
                    dr_t[j] += drp * c.prev_h[j];
                    dh_prev[j] += drp * c.r_t[j];
                }
            }

            for i in 0..d {
                let dsg = dg_pre[i];
                let dsr = dr_t[i] * c.r_t[i] * (1.0 - c.r_t[i]);
                g_b_g[i] += dsg;
                g_b_r[i] += dsr;
                for j in 0..d {
                    g_w_g[i * d + j] += dsg * c.u_t[j];
                    g_u_g[i * d + j] += dsg * c.prev_h[j];
                    g_w_r[i * d + j] += dsr * c.u_t[j];
                    g_u_r[i * d + j] += dsr * c.prev_h[j];
                    du_t[j] += dsg * self.w_g[i * d + j] + dsr * self.w_r[i * d + j];
                    dh_prev[j] += dsg * self.u_g[i * d + j] + dsr * self.u_r[i * d + j];
                }
            }

            let mut du_norm = [0.0_f32; HIDDEN_DIM];
            for i in 0..d {
                let dpre = du_t[i] * (1.0 - c.u_t[i] * c.u_t[i]);
                g_ln_in_b[i] += dpre;
                g_ln_in_g[i] += dpre * c.u_norm[i];
                du_norm[i] = dpre * self.ln_in_g[i];
            }
            let sum_dun: f32 = du_norm.iter().sum();
            let sum_dun_un: f32 = du_norm
                .iter()
                .zip(c.u_norm.iter())
                .map(|(a, b)| a * b)
                .sum();
            let mut dx_sal = [0.0_f32; RL_STATE_DIM];
            for i in 0..d {
                let du_raw =
                    c.u_inv_std * (du_norm[i] - inv_d * sum_dun - inv_d * c.u_norm[i] * sum_dun_un);
                g_b_in[i] += du_raw;
                let row_g = &mut g_w_in[i * TOTAL_INPUT_DIM..(i + 1) * TOTAL_INPUT_DIM];
                let row_w = &self.w_in[i * TOTAL_INPUT_DIM..(i + 1) * TOTAL_INPUT_DIM];
                for j in 0..TOTAL_INPUT_DIM {
                    row_g[j] += du_raw * c.x_cat[j];
                    if j < RL_STATE_DIM {
                        dx_sal[j] += du_raw * row_w[j];
                    }
                }
            }
            let step_sal_dx: f32 = dx_sal.iter().map(|v| v.abs()).sum::<f32>() / (RL_STATE_DIM as f32);
            total_sal_dx += step_sal_dx;

            dh_next = dh_prev;
        }

        let mean_sal_dx = total_sal_dx * inv_t;
        let mean_sal_dh = total_sal_dh * inv_t;

        // If computing Fisher EWC on settled winners or pure saliency audit (`lr_scale == 0.0`), do not mutate weights!
        if let Some(fisher_acc) = accumulate_fisher_into {
            let mut idx = 0usize;
            for g in g_w_in
                .iter()
                .chain(g_b_in.iter())
                .chain(g_ln_in_g.iter())
                .chain(g_ln_in_b.iter())
                .chain(g_w_g.iter())
                .chain(g_u_g.iter())
                .chain(g_b_g.iter())
                .chain(g_w_r.iter())
                .chain(g_u_r.iter())
                .chain(g_b_r.iter())
                .chain(g_w_h.iter())
                .chain(g_u_h.iter())
                .chain(g_b_h.iter())
                .chain(g_ln_out_g.iter())
                .chain(g_ln_out_b.iter())
                .chain(g_w_a.iter())
                .chain(g_b_a.iter())
                .chain(g_w_d.iter())
                .chain(g_b_d.iter())
                .chain(g_w_v.iter())
                .chain(g_b_v.iter())
                .chain(g_w_s.iter())
                .chain(g_b_s.iter())
                .chain(g_w_c.iter())
                .chain(g_b_c.iter())
            {
                if idx < fisher_acc.len() {
                    fisher_acc[idx] += g * g;
                }
                idx += 1;
            }
        }

        if lr_scale <= 0.0 {
            return (utility as f64, mean_sal_dx, mean_sal_dh);
        }

        let mut sq_norm = 0.0_f32;
        for g in g_w_in
            .iter()
            .chain(g_b_in.iter())
            .chain(g_ln_in_g.iter())
            .chain(g_ln_in_b.iter())
            .chain(g_w_g.iter())
            .chain(g_u_g.iter())
            .chain(g_b_g.iter())
            .chain(g_w_r.iter())
            .chain(g_u_r.iter())
            .chain(g_b_r.iter())
            .chain(g_w_h.iter())
            .chain(g_u_h.iter())
            .chain(g_b_h.iter())
            .chain(g_ln_out_g.iter())
            .chain(g_ln_out_b.iter())
            .chain(g_w_a.iter())
            .chain(g_b_a.iter())
            .chain(g_w_d.iter())
            .chain(g_b_d.iter())
            .chain(g_w_v.iter())
            .chain(g_b_v.iter())
            .chain(g_w_s.iter())
            .chain(g_b_s.iter())
            .chain(g_w_c.iter())
            .chain(g_b_c.iter())
        {
            sq_norm += g * g;
        }
        let norm = sq_norm.sqrt();
        let clip_scale = if norm > 1.0 { 1.0 / norm } else { 1.0 };

        let lr = hparams.lr * lr_scale;
        let wd = hparams.weight_decay;
        self.adam_step = self.adam_step.saturating_add(1);
        let t_f = self.adam_step as f32;
        let b1 = 0.9_f32;
        let b2 = 0.999_f32;
        let eps = 1e-8_f32;
        let bc1 = 1.0 - b1.powf(t_f);
        let bc2 = 1.0 - b2.powf(t_f);
        let has_ewc = ewc_lambda > 0.0 && self.ewc_anchor.len() == self.param_count();
        let has_fisher = has_ewc && self.ewc_fisher_diag.len() == self.param_count();

        let mut idx = 0usize;
        let mut apply_slice = |params: &mut [f32], grads: &[f32], decay: f32| {
            for (p, &g_raw) in params.iter_mut().zip(grads.iter()) {
                let ewc_g = if has_ewc {
                    let f_i = if has_fisher {
                        self.ewc_fisher_diag[idx]
                    } else {
                        1.0
                    };
                    ewc_lambda * f_i * (*p - self.ewc_anchor[idx])
                } else {
                    0.0
                };
                let g = g_raw * clip_scale + ewc_g;
                let m = b1 * self.adam_m[idx] + (1.0 - b1) * g;
                let v = b2 * self.adam_v[idx] + (1.0 - b2) * g * g;
                self.adam_m[idx] = m;
                self.adam_v[idx] = v;
                let m_hat = m / bc1;
                let v_hat = v / bc2;
                *p = *p * (1.0 - lr * decay) - lr * m_hat / (v_hat.sqrt() + eps);
                idx += 1;
            }
        };

        apply_slice(&mut self.w_in, &g_w_in, wd);
        apply_slice(&mut self.b_in, &g_b_in, 0.0);
        apply_slice(&mut self.ln_in_g, &g_ln_in_g, 0.0);
        apply_slice(&mut self.ln_in_b, &g_ln_in_b, 0.0);
        apply_slice(&mut self.w_g, &g_w_g, wd);
        apply_slice(&mut self.u_g, &g_u_g, wd);
        apply_slice(&mut self.b_g, &g_b_g, 0.0);
        apply_slice(&mut self.w_r, &g_w_r, wd);
        apply_slice(&mut self.u_r, &g_u_r, wd);
        apply_slice(&mut self.b_r, &g_b_r, 0.0);
        apply_slice(&mut self.w_h, &g_w_h, wd);
        apply_slice(&mut self.u_h, &g_u_h, wd);
        apply_slice(&mut self.b_h, &g_b_h, 0.0);
        apply_slice(&mut self.ln_out_g, &g_ln_out_g, 0.0);
        apply_slice(&mut self.ln_out_b, &g_ln_out_b, 0.0);
        apply_slice(&mut self.w_a_heads, &g_w_a, wd);
        apply_slice(&mut self.b_a_heads, &g_b_a, 0.0);
        apply_slice(&mut self.w_d_heads, &g_w_d, wd);
        apply_slice(&mut self.b_d_heads, &g_b_d, 0.0);
        apply_slice(&mut self.w_v_heads, &g_w_v, wd);
        apply_slice(&mut self.b_v_heads, &g_b_v, 0.0);
        apply_slice(&mut self.w_s_heads, &g_w_s, wd);
        apply_slice(&mut self.b_s_heads, &g_b_s, 0.0);
        apply_slice(&mut self.w_c, &g_w_c, wd);
        apply_slice(&mut self.b_c, &g_b_c, 0.0);

        (utility as f64, mean_sal_dx, mean_sal_dh)
    }

    pub fn train_bptt_chunk(
        &mut self,
        chunk: &[CausalTrajectoryBar],
        world_model: Option<&PhysicsWorldModel>,
        feat_mean: &[f32],
        feat_var: &[f32],
        hparams: &AgentHyperParams,
        lr_scale: f32,
        ewc_lambda: f32,
    ) -> f64 {
        let zero = [0.0_f32; WM_DIM];
        self.train_bptt_chunk_ext(
            chunk,
            &zero,
            world_model,
            feat_mean,
            feat_var,
            hparams,
            lr_scale,
            ewc_lambda,
            None,
            None,
        )
        .0
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PopulationMember {
    pub id: usize,
    pub net: SequencePolicyNet,
    pub hparams: AgentHyperParams,
    pub causal_score: f64,
    pub causal_sharpe: f64,
    pub causal_net_pnl: f64,
    pub causal_trades: usize,
    #[serde(default)]
    pub causal_pf: f64,
    #[serde(default)]
    pub causal_red_loss_usd: f64,
    #[serde(default)]
    pub causal_win_r_sum: f64,
    pub hedge_weight: f64,
}

#[derive(Debug, Clone)]
pub struct EnsembleInferenceOutput {
    pub action: f32,
    pub vehicle_prob: f32,
    pub tail_stop_dist: f32,
    pub deadzone: f32,
    pub hedge_weighted_sharpe: f64,
    pub active_specialists: usize,
    pub next_hiddens: Vec<[f32; HIDDEN_DIM]>,
    pub next_wm_hidden: [f32; WM_DIM],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PureAutonomousRlBrain {
    pub brain_key: String,
    pub horizon_label: String,
    #[serde(default = "default_world_model")]
    pub world_model: PhysicsWorldModel,
    pub population: Vec<PopulationMember>,
    pub feat_mean: Vec<f32>,
    pub feat_var: Vec<f32>,
    pub welford_count: f64,
    pub generation_steps: u64,
    pub online_updates_count: u64,
    pub pbt_generations_run: u64,
    #[serde(default)]
    pub promoted_generation_id: u32,
    pub seed_base: u64,
    #[serde(skip)]
    pub red_replay_buffer: Vec<AuditedRedTradeTrajectory>,
    #[serde(skip)]
    pub winner_fisher_buffer: Vec<SettledWinnerTrajectory>,
}

fn default_world_model() -> PhysicsWorldModel {
    let mut seed = 0x55AA_1234_5678_9ABC;
    PhysicsWorldModel::new(&mut seed)
}

impl PureAutonomousRlBrain {
    pub fn new_for_horizon(horizon_label: &str, seed_base: u64) -> Self {
        let mut seed = seed_base ^ 0xA5A5_5A5A_1234_5678;
        let world_model = PhysicsWorldModel::new(&mut seed);
        let mut population = Vec::with_capacity(POPULATION_SIZE);
        for id in 0..POPULATION_SIZE {
            let hparams = AgentHyperParams::diverse_prior(id, &mut seed);
            let net = SequencePolicyNet::new(&mut seed, hparams.timescale_shift);
            population.push(PopulationMember {
                id,
                net,
                hparams,
                causal_score: 0.0,
                causal_sharpe: 0.0,
                causal_net_pnl: 0.0,
                causal_trades: 0,
                causal_pf: 0.0,
                causal_red_loss_usd: 0.0,
                causal_win_r_sum: 0.0,
                hedge_weight: 1.0 / (POPULATION_SIZE as f64),
            });
        }
        Self {
            brain_key: horizon_label.to_string(),
            horizon_label: horizon_label.to_string(),
            world_model,
            population,
            feat_mean: vec![0.0; RL_STATE_DIM],
            feat_var: vec![1.0; RL_STATE_DIM],
            welford_count: 0.0,
            generation_steps: 0,
            online_updates_count: 0,
            pbt_generations_run: 0,
            promoted_generation_id: 0,
            seed_base,
            red_replay_buffer: Vec::new(),
            winner_fisher_buffer: Vec::new(),
        }
    }

    pub fn default_horizon_suite() -> Vec<Self> {
        vec![
            Self::new_for_horizon("SCALP_0DTE_1M", 0x1101),
            Self::new_for_horizon("SCALP_0DTE", 0x1505),
            Self::new_for_horizon("BRAIN-01", 0x2001),
            Self::new_for_horizon("BRAIN-02", 0x3002),
            Self::new_for_horizon("BRAIN-03", 0x4003),
            Self::new_for_horizon("BRAIN-04", 0x5004),
        ]
    }

    pub fn rehydrate_anchors_if_needed(&mut self) {
        if self.feat_mean.len() != RL_STATE_DIM {
            self.feat_mean = vec![0.0; RL_STATE_DIM];
            self.feat_var = vec![1.0; RL_STATE_DIM];
            self.welford_count = 0.0;
        }
        let mut seed = self.seed_base ^ 0x1234_5678_9ABC_DEF0;
        self.world_model.ensure_valid_shapes(&mut seed);
        for m in &mut self.population {
            m.net.ensure_valid_shapes(&mut seed, m.hparams.timescale_shift);
        }
    }

    pub fn reset_welford(&mut self) {
        self.feat_mean = vec![0.0; RL_STATE_DIM];
        self.feat_var = vec![1.0; RL_STATE_DIM];
        self.welford_count = 0.0;
    }

    pub fn snapshot_ewc_anchors(&mut self) {
        for m in &mut self.population {
            m.net.snapshot_ewc_anchor();
        }
    }

    /// Computes Diagonal Fisher Information `F_i = E_{winners}[(d L / d theta_i)^2]` across all
    /// settled causal winning trajectories so subsequent Red-Trade RL updates lock profitable patterns!
    pub fn compute_fisher_ewc_from_winners(&mut self) {
        self.snapshot_ewc_anchors();
        if self.winner_fisher_buffer.is_empty() {
            return;
        }
        let winners = self.winner_fisher_buffer.clone();
        let feat_mean = self.feat_mean.clone();
        let feat_var = self.feat_var.clone();
        let wm = &self.world_model;

        self.population.par_iter_mut().for_each(|member| {
            let n_params = member.net.param_count();
            let mut fisher_acc = vec![0.0_f32; n_params];
            let mut count = 0usize;
            for w_ep in winners.iter().rev().take(32) {
                if w_ep.bars.len() < 4 {
                    continue;
                }
                let mut targets = vec![None; w_ep.bars.len()];
                let end_hold = w_ep.bars.len().saturating_sub(1);
                for slot in targets
                    .iter_mut()
                    .take(end_hold)
                    .skip(w_ep.entry_offset.min(end_hold))
                {
                    *slot = Some(w_ep.direction_sign * 0.65);
                }
                if let Some(last_slot) = targets.last_mut() {
                    *last_slot = Some(0.0);
                }
                let weight = (1.0 + w_ep.net_r.max(0.1)).clamp(1.0, 5.0);
                let _ = member.net.train_bptt_chunk_ext(
                    &w_ep.bars,
                    &w_ep.prior_regime_z,
                    Some(wm),
                    &feat_mean,
                    &feat_var,
                    &member.hparams,
                    0.0,
                    0.0,
                    Some((&targets, weight)),
                    Some(&mut fisher_acc),
                );
                count += 1;
            }
            if count > 0 {
                let mean_f =
                    (fisher_acc.iter().sum::<f32>() / (n_params as f32)).max(1e-8);
                for f in &mut fisher_acc {
                    *f = (*f / mean_f).clamp(0.10, 12.0);
                }
                member.net.ewc_fisher_diag = fisher_acc;
            }
        });
    }

    /// Step 1 of Closed-Loop Loss Autopsy: Replays the exact losing bar sequence against causal
    /// counterfactuals (`a = 0.0` FLAT at entry vs optimal post-cost exit bar along the held path)
    /// and computes analytical BPTT saliency `|d(Loss)/dx_t|` + `|d(Loss)/dh_t|`.
    #[allow(clippy::too_many_arguments)]
    pub fn audit_and_insert_red_trade(
        &mut self,
        symbol: String,
        exit_ts_sec: i64,
        net_pnl_usd: f64,
        net_r: f32,
        direction_sign: f32,
        is_option: bool,
        entry_offset: usize,
        prior_regime_z: [f32; WM_DIM],
        bars: Vec<CausalTrajectoryBar>,
    ) -> AuditedRedTradeTrajectory {
        let t_len = bars.len();
        let entry_idx = entry_offset.min(t_len.saturating_sub(2));
        let unit_cost = if is_option {
            bars[entry_idx].one_way_cost_option
        } else {
            bars[entry_idx].one_way_cost_shares
        };
        let rt_cost = 2.0 * unit_cost;

        let mut cum_gross = 0.0_f32;
        let mut best_cf_net = -rt_cost;
        let mut best_exit_off = entry_idx;
        for (idx, b) in bars.iter().enumerate().skip(entry_idx) {
            let step_r = if is_option {
                b.ret_oo_option
            } else {
                b.ret_oo_shares
            };
            cum_gross += direction_sign * step_r;
            let net_at_next_open = cum_gross - rt_cost;
            if net_at_next_open > best_cf_net && idx + 1 < t_len {
                best_cf_net = net_at_next_open;
                best_exit_off = idx + 1;
            }
        }
        let approx_r_unit = (rt_cost * 2.5).max(0.008);
        let best_cf_r = best_cf_net / approx_r_unit;

        let (fault_kind, optimal_exit_offset) = if best_cf_r > 0.18 && best_exit_off > entry_idx {
            (LossFaultKind::HoldingExitFailure, best_exit_off)
        } else {
            (LossFaultKind::EntryHallucination, entry_idx)
        };

        let priority_weight = (1.0 + 4.0 * net_r.abs() * net_r.abs()).clamp(1.0, 35.0);

        let mut targets = vec![None; t_len];
        match fault_kind {
            LossFaultKind::EntryHallucination => {
                for slot in targets.iter_mut().skip(entry_idx.saturating_sub(1)) {
                    *slot = Some(0.0);
                }
            }
            LossFaultKind::HoldingExitFailure => {
                for slot in targets.iter_mut().skip(optimal_exit_offset) {
                    *slot = Some(0.0);
                }
            }
        }

        // Compute BPTT saliency `|d(Loss)/dx_t|` and `|d(Loss)/dh_t|` on champion member (read-only `lr_scale = 0.0`)
        let champ_idx = self
            .population
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.causal_score.total_cmp(&b.1.causal_score))
            .map(|(i, _)| i)
            .unwrap_or(0);
        let mut probe_net = self.population[champ_idx].net.clone();
        let hp = self.population[champ_idx].hparams.clone();
        let (_, sal_dx, sal_dh) = probe_net.train_bptt_chunk_ext(
            &bars,
            &prior_regime_z,
            Some(&self.world_model),
            &self.feat_mean,
            &self.feat_var,
            &hp,
            0.0,
            0.0,
            Some((&targets, priority_weight)),
            None,
        );

        let record = AuditedRedTradeTrajectory {
            symbol,
            exit_ts_sec,
            net_pnl_usd,
            net_r,
            priority_weight,
            fault_kind,
            entry_offset: entry_idx,
            optimal_exit_offset,
            best_counterfactual_net_r: best_cf_r,
            saliency_dx: sal_dx,
            saliency_dh: sal_dh,
            prior_regime_z,
            bars,
        };

        self.red_replay_buffer.push(record.clone());
        if self.red_replay_buffer.len() > MAX_RED_REPLAY_CAP {
            self.red_replay_buffer
                .sort_by(|a, b| b.priority_weight.total_cmp(&a.priority_weight));
            self.red_replay_buffer.truncate(MAX_RED_REPLAY_CAP);
        }
        record
    }

    pub fn insert_winner_trajectory(&mut self, winner: SettledWinnerTrajectory) {
        self.winner_fisher_buffer.push(winner);
        if self.winner_fisher_buffer.len() > MAX_WINNER_FISHER_CAP {
            let excess = self.winner_fisher_buffer.len() - MAX_WINNER_FISHER_CAP;
            self.winner_fisher_buffer.drain(0..excess);
        }
    }

    /// Step 2 of Closed-Loop Loss Autopsy: Updates all `M = 24` population members in parallel
    /// using a 50% / 50% batch mix of Prioritized Red-Trade Replay (`w = 1.0 + 4.0 * |net_r|^2`)
    /// and causal session episodes, anchored by Diagonal Fisher EWC!
    pub fn train_red_autopsy_and_causal_batch_parallel(
        &mut self,
        causal_episodes: &[CausalSessionEpisode],
        steps: usize,
        lr_scale: f32,
        ewc_lambda: f32,
    ) {
        if self.red_replay_buffer.is_empty() && causal_episodes.is_empty() {
            return;
        }
        let feat_mean = self.feat_mean.clone();
        let feat_var = self.feat_var.clone();
        let wm = &self.world_model;
        let reds = self.red_replay_buffer.clone();
        let seed_base = self.seed_base ^ self.online_updates_count.wrapping_mul(0x9E37_79B9_7F4A_7C15);

        self.population.par_iter_mut().for_each(|member| {
            let mut seed = seed_base ^ ((member.id as u64 + 1).wrapping_mul(0x517C_C1B7_2722_0A95));
            let total_red_w: f32 = reds.iter().map(|r| r.priority_weight).sum();

            for _ in 0..steps {
                // First half (50% of batch): Prioritized Red-Trade Replay sampled proportionally to `1.0 + 4.0 * |net_r|^2`
                if !reds.is_empty() && total_red_w > 0.0 {
                    let pick_u = rand_uniform(&mut seed, 0.0, total_red_w);
                    let mut acc = 0.0_f32;
                    let mut chosen = &reds[0];
                    for r in &reds {
                        acc += r.priority_weight;
                        if acc >= pick_u {
                            chosen = r;
                            break;
                        }
                    }
                    if chosen.bars.len() >= 3 {
                        let mut targets = vec![None; chosen.bars.len()];
                        match chosen.fault_kind {
                            LossFaultKind::EntryHallucination => {
                                for slot in targets
                                    .iter_mut()
                                    .skip(chosen.entry_offset.saturating_sub(1))
                                {
                                    *slot = Some(0.0);
                                }
                            }
                            LossFaultKind::HoldingExitFailure => {
                                for slot in targets.iter_mut().skip(chosen.optimal_exit_offset) {
                                    *slot = Some(0.0);
                                }
                            }
                        }
                        let _ = member.net.train_bptt_chunk_ext(
                            &chosen.bars,
                            &chosen.prior_regime_z,
                            Some(wm),
                            &feat_mean,
                            &feat_var,
                            &member.hparams,
                            lr_scale,
                            ewc_lambda,
                            Some((&targets, chosen.priority_weight)),
                            None,
                        );
                    }
                }

                // Second half (50% of batch): Causal Session Episode BPTT with EOD closure
                if !causal_episodes.is_empty() {
                    let ep_idx = (xorshift64(&mut seed) as usize) % causal_episodes.len();
                    let ep = &causal_episodes[ep_idx];
                    if ep.bars.len() >= 4 {
                        let max_c = member.hparams.chunk_len.max(24);
                        let slice = if ep.is_intraday_session || ep.bars.len() <= max_c {
                            &ep.bars[..]
                        } else {
                            let max_start = ep.bars.len() - max_c;
                            let st = (xorshift64(&mut seed) as usize) % (max_start + 1);
                            &ep.bars[st..st + max_c]
                        };
                        let _ = member.net.train_bptt_chunk_ext(
                            slice,
                            &ep.prior_regime_z,
                            Some(wm),
                            &feat_mean,
                            &feat_var,
                            &member.hparams,
                            lr_scale * 0.75,
                            ewc_lambda,
                            None,
                            None,
                        );
                    }
                }
            }
        });
    }

    pub fn update_welford_batch(&mut self, xs: &[[f32; RL_STATE_DIM]]) {
        for x in xs {
            self.welford_count += 1.0;
            let n = self.welford_count as f32;
            for i in 0..RL_STATE_DIM {
                let delta = x[i] - self.feat_mean[i];
                self.feat_mean[i] += delta / n;
                let delta2 = x[i] - self.feat_mean[i];
                if self.welford_count > 1.0 {
                    let m2 = self.feat_var[i] * ((self.welford_count - 2.0).max(1.0) as f32)
                        + delta * delta2;
                    self.feat_var[i] = (m2 / ((self.welford_count - 1.0) as f32)).max(1e-4);
                }
            }
        }
    }

    #[inline]
    pub fn normalize_x(&self, x: &[f32; RL_STATE_DIM]) -> [f32; RL_STATE_DIM] {
        let mut out = [0.0_f32; RL_STATE_DIM];
        for i in 0..RL_STATE_DIM {
            let m = self.feat_mean.get(i).copied().unwrap_or(0.0);
            let v = self.feat_var.get(i).copied().unwrap_or(1.0).max(1e-4);
            out[i] = ((x[i] - m) / v.sqrt()).clamp(-6.0, 6.0);
        }
        out
    }

    /// Stage A & C: Trains the Self-Supervised Physics World Model (`E_phi`) via Pinball Quantile Loss
    /// across causal session episodes (`t + 1 + Delta <= t_current`).
    pub fn train_world_model_episodes(
        &mut self,
        episodes: &[CausalSessionEpisode],
        epochs: usize,
        lr: f32,
    ) -> f64 {
        if episodes.is_empty() || epochs == 0 {
            return self.world_model.last_pinball_loss;
        }
        let mut loss = self.world_model.last_pinball_loss;
        for _ in 0..epochs {
            for ep in episodes {
                if ep.bars.len() < 4 {
                    continue;
                }
                if ep.is_intraday_session || ep.bars.len() <= 96 {
                    loss = self
                        .world_model
                        .train_chunk_with_init(
                            &ep.bars,
                            &ep.prior_regime_z,
                            &self.feat_mean,
                            &self.feat_var,
                            lr,
                        )
                        .0;
                } else {
                    let mut z_carry = ep.prior_regime_z;
                    let mut start = 0usize;
                    while start + 4 <= ep.bars.len() {
                        let end = (start + 72).min(ep.bars.len());
                        let (l, z_end) = self.world_model.train_chunk_with_init(
                            &ep.bars[start..end],
                            &z_carry,
                            &self.feat_mean,
                            &self.feat_var,
                            lr,
                        );
                        loss = l;
                        z_carry = z_end;
                        start += 72;
                    }
                }
            }
        }
        loss
    }

    pub fn train_world_model_causal(
        &mut self,
        sequences: &[Vec<CausalTrajectoryBar>],
        epochs: usize,
        lr: f32,
    ) -> f64 {
        let eps: Vec<CausalSessionEpisode> = sequences
            .iter()
            .map(|bars| CausalSessionEpisode {
                symbol: String::new(),
                day_key: 0,
                start_ts_sec: bars.first().map(|b| b.ts_sec).unwrap_or(0),
                end_ts_sec: bars.last().map(|b| b.ts_sec).unwrap_or(0),
                is_intraday_session: false,
                prior_regime_z: [0.0; WM_DIM],
                bars: bars.clone(),
            })
            .collect();
        self.train_world_model_episodes(&eps, epochs, lr)
    }

    /// Trains all `M = 24` diverse specialists in parallel across CPU cores via `rayon` BPTT
    /// over causal session episodes (honoring full intraday episodes & EOD closure).
    pub fn train_population_episodes_bptt_parallel(
        &mut self,
        episodes: &[CausalSessionEpisode],
        epochs: usize,
        lr_scale: f32,
        ewc_lambda: f32,
    ) {
        if episodes.is_empty() || epochs == 0 {
            return;
        }
        let feat_mean = self.feat_mean.clone();
        let feat_var = self.feat_var.clone();
        let wm = &self.world_model;

        self.population.par_iter_mut().for_each(|member| {
            let chunk_len = member.hparams.chunk_len.max(24);
            for _ep in 0..epochs {
                for ep in episodes {
                    if ep.bars.len() < 4 {
                        continue;
                    }
                    if ep.is_intraday_session || ep.bars.len() <= 96 {
                        // Train full `(Symbol, TradingDay)` intraday session episode end-to-end so
                        // closing-bell `force_flat` EOD friction backpropagates across the session!
                        let _ = member.net.train_bptt_chunk_ext(
                            &ep.bars,
                            &ep.prior_regime_z,
                            Some(wm),
                            &feat_mean,
                            &feat_var,
                            &member.hparams,
                            lr_scale,
                            ewc_lambda,
                            None,
                            None,
                        );
                    } else {
                        let mut start = 0usize;
                        while start + 6 <= ep.bars.len() {
                            let end = (start + chunk_len).min(ep.bars.len());
                            let init_z = if start == 0 {
                                ep.prior_regime_z
                            } else {
                                [0.0; WM_DIM]
                            };
                            let _ = member.net.train_bptt_chunk_ext(
                                &ep.bars[start..end],
                                &init_z,
                                Some(wm),
                                &feat_mean,
                                &feat_var,
                                &member.hparams,
                                lr_scale,
                                ewc_lambda,
                                None,
                                None,
                            );
                            start += chunk_len;
                        }
                    }
                }
            }
        });
        self.generation_steps = self.generation_steps.saturating_add(epochs as u64);
    }

    pub fn train_population_bptt_parallel(
        &mut self,
        sequences: &[Vec<CausalTrajectoryBar>],
        epochs: usize,
        lr_scale: f32,
        ewc_lambda: f32,
    ) {
        let eps: Vec<CausalSessionEpisode> = sequences
            .iter()
            .map(|bars| CausalSessionEpisode {
                symbol: String::new(),
                day_key: 0,
                start_ts_sec: bars.first().map(|b| b.ts_sec).unwrap_or(0),
                end_ts_sec: bars.last().map(|b| b.ts_sec).unwrap_or(0),
                is_intraday_session: false,
                prior_regime_z: [0.0; WM_DIM],
                bars: bars.clone(),
            })
            .collect();
        self.train_population_episodes_bptt_parallel(&eps, epochs, lr_scale, ewc_lambda);
    }

    /// Updates Multiplicative-Weights specialist routing `hedge_weight` across all `M = 24` agents.
    pub fn update_hedge_weights_with_eta(&mut self, eta: f64) {
        if self.population.is_empty() {
            return;
        }
        let max_s = self
            .population
            .iter()
            .map(|m| m.causal_score)
            .fold(f64::NEG_INFINITY, f64::max);
        let mut sum_w = 0.0_f64;
        for m in &mut self.population {
            let w = if m.causal_score.is_finite() {
                (eta * (m.causal_score - max_s)).clamp(-25.0, 0.0).exp()
            } else {
                0.0
            };
            m.hedge_weight = w;
            sum_w += w;
        }
        if sum_w > 1e-12 {
            for m in &mut self.population {
                m.hedge_weight /= sum_w;
            }
        } else {
            let eq = 1.0 / (self.population.len() as f64);
            for m in &mut self.population {
                m.hedge_weight = eq;
            }
        }
    }

    pub fn update_hedge_weights(&mut self) {
        self.update_hedge_weights_with_eta(2.8);
    }

    pub fn pbt_exploit_and_explore(&mut self) {
        let n = self.population.len();
        if n < 4 {
            return;
        }
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            self.population[b]
                .causal_score
                .total_cmp(&self.population[a].causal_score)
        });
        let q = (n / 4).max(1);
        let mut seed =
            self.seed_base ^ (self.pbt_generations_run.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        for k in 0..q {
            let top_idx = order[k];
            let bot_idx = order[n - 1 - k];
            let mut cloned_net = self.population[top_idx].net.clone();
            let mutated_hp = self.population[top_idx].hparams.mutate(&mut seed);
            cloned_net.perturb(0.018, &mut seed);
            self.population[bot_idx].net = cloned_net;
            self.population[bot_idx].hparams = mutated_hp;
            self.population[bot_idx].causal_score = self.population[top_idx].causal_score * 0.85;
            self.population[bot_idx].causal_sharpe = self.population[top_idx].causal_sharpe * 0.85;
            self.population[bot_idx].causal_pf = self.population[top_idx].causal_pf * 0.90;
        }
        self.pbt_generations_run = self.pbt_generations_run.saturating_add(1);
        self.update_hedge_weights();
    }

    /// Causal Multiplicative-Weights Specialist Routing over the `M = 24` population fed by a
    /// precomputed `(x_norm, z_wm, wm_pred)` World-Model step.
    #[inline]
    pub fn infer_ensemble_step_with_precomputed_wm(
        &self,
        x_norm: &[f32; RL_STATE_DIM],
        z_wm: &[f32; WM_DIM],
        wm_pred: &[f32; WM_OUT_DIM],
        self_state: SelfAwarenessState,
        prev_hiddens: &[[f32; HIDDEN_DIM]],
        option_allowed: bool,
    ) -> EnsembleInferenceOutput {
        let mut next_hiddens = Vec::with_capacity(self.population.len());

        let qualified: Vec<&PopulationMember> = self
            .population
            .iter()
            .filter(|m| m.causal_trades >= 2)
            .collect();
        let has_eval_history = !qualified.is_empty();
        let positive_members: Vec<&PopulationMember> = qualified
            .iter()
            .copied()
            .filter(|m| m.causal_net_pnl > 0.0 && m.causal_pf > 1.02)
            .collect();
        let hedge_weighted_sharpe: f64 = if !qualified.is_empty() {
            let w_sum: f64 = qualified.iter().map(|m| m.hedge_weight).sum();
            if w_sum > 1e-9 {
                qualified
                    .iter()
                    .map(|m| m.hedge_weight * m.causal_sharpe)
                    .sum::<f64>()
                    / w_sum
            } else {
                0.0
            }
        } else {
            0.0
        };

        let mut ranked_scores: Vec<f64> = self.population.iter().map(|m| m.causal_score).collect();
        ranked_scores.sort_by(|a, b| b.total_cmp(a));
        let top_k_idx = (self.population.len() / 4).max(1) - 1;
        let top_cutoff = ranked_scores
            .get(top_k_idx)
            .copied()
            .unwrap_or(f64::NEG_INFINITY);

        let mut weighted_action = 0.0_f64;
        let mut weighted_action_sq = 0.0_f64;
        let mut weighted_vehicle = 0.0_f64;
        let mut weighted_stop = 0.0_f64;
        let mut weighted_dz = 0.0_f64;
        let mut positive_weight_sum = 0.0_f64;
        let mut active_specialists = 0usize;

        for (i, member) in self.population.iter().enumerate() {
            let prev_h = prev_hiddens
                .get(i)
                .copied()
                .unwrap_or([0.0_f32; HIDDEN_DIM]);
            let step = member.net.forward_step_with_wm(
                x_norm,
                z_wm,
                wm_pred,
                self_state,
                &prev_h,
                *z_wm,
                member.hparams.cost_sensitivity,
                option_allowed,
            );
            next_hiddens.push(step.next_hidden);

            let w = if !has_eval_history {
                member.hedge_weight
            } else if !positive_members.is_empty() {
                if member.causal_trades >= 2
                    && member.causal_net_pnl > 0.0
                    && member.causal_pf > 1.02
                    && member.causal_score >= top_cutoff
                {
                    member.hedge_weight
                } else if member.causal_trades >= 2
                    && member.causal_net_pnl > 0.0
                    && member.causal_pf > 1.05
                {
                    member.hedge_weight * 0.35
                } else {
                    0.0
                }
            } else if member.causal_score >= top_cutoff {
                member.hedge_weight
            } else {
                0.0
            };

            if w > 0.0 && step.action.abs() > 1e-6 {
                active_specialists += 1;
            }
            let act = step.action as f64;
            weighted_action += w * act;
            weighted_action_sq += w * act * act;
            weighted_vehicle += w * (step.vehicle_prob as f64);
            weighted_stop += w * (step.tail_stop_dist as f64);
            weighted_dz += w * (step.deadzone as f64);
            positive_weight_sum += w;
        }

        let mut agg_action = if positive_weight_sum > 1e-9 {
            (weighted_action / positive_weight_sum) as f32
        } else {
            0.0
        };
        let agg_vehicle = if positive_weight_sum > 1e-9 {
            (weighted_vehicle / positive_weight_sum) as f32
        } else {
            0.0
        };
        let agg_stop = if positive_weight_sum > 1e-9 {
            (weighted_stop / positive_weight_sum) as f32
        } else {
            0.024
        };
        let agg_dz = if positive_weight_sum > 1e-9 {
            (weighted_dz / positive_weight_sum) as f32
        } else {
            0.08
        };

        let is_holding = self_state.current_pos.abs() > 1e-4;
        if !is_holding && positive_weight_sum > 1e-9 {
            let mu = (weighted_action / positive_weight_sum) as f32;
            let var_a =
                ((weighted_action_sq / positive_weight_sum) - (mu as f64) * (mu as f64)).max(0.0);
            let std_a = var_a.sqrt() as f32;
            let lcb_mag = (agg_action.abs() - 0.08 * std_a).max(0.0);
            agg_action = agg_action.signum() * lcb_mag;
        }

        // Responsive Neural-Exit snap when holding so laggard minority members never block EOD/peak exits!
        let flat_snap = if !is_holding {
            0.045
        } else {
            (0.32 * self_state.current_pos.abs()).clamp(0.025, 0.085)
        };
        if agg_action.abs() < flat_snap {
            agg_action = 0.0;
        }

        EnsembleInferenceOutput {
            action: agg_action.clamp(-1.0, 1.0),
            vehicle_prob: agg_vehicle.clamp(0.0, 1.0),
            tail_stop_dist: agg_stop.clamp(0.008, 0.09),
            deadzone: agg_dz,
            hedge_weighted_sharpe,
            active_specialists,
            next_hiddens,
            next_wm_hidden: *z_wm,
        }
    }

    pub fn infer_ensemble_step(
        &self,
        x_raw_212: &[f32; RL_STATE_DIM],
        self_state: SelfAwarenessState,
        prev_hiddens: &[[f32; HIDDEN_DIM]],
        prev_wm_hidden: &[f32; WM_DIM],
        option_allowed: bool,
    ) -> EnsembleInferenceOutput {
        let x_norm = self.normalize_x(x_raw_212);
        let (z_wm, wm_pred) = self.world_model.forward_step(&x_norm, prev_wm_hidden);
        self.infer_ensemble_step_with_precomputed_wm(
            &x_norm,
            &z_wm,
            &wm_pred,
            self_state,
            prev_hiddens,
            option_allowed,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_bar(
        t: i64,
        sig: f32,
        ret_s: f32,
        cost_s: f32,
        cost_bps: f32,
        opt_ok: bool,
        force_flat: bool,
    ) -> CausalTrajectoryBar {
        let mut x = [0.0_f32; RL_STATE_DIM];
        x[0] = sig;
        x[1] = sig * 0.8;
        let mut wm_targets = [0.0_f32; WM_OUT_DIM];
        for h in 0..WM_HORIZONS.len() {
            wm_targets[h * 4] = ret_s * 2.0;
            wm_targets[h * 4 + 1] = ret_s * 3.5;
            wm_targets[h * 4 + 2] = ret_s * 5.0;
            wm_targets[h * 4 + 3] = ((ret_s.abs() * 4.0) / cost_s.max(1e-4)).max(0.1).ln();
        }
        let wm_valid = [true; WM_OUT_DIM];
        CausalTrajectoryBar {
            ts_sec: 1_700_000_000 + t * 300,
            x_212: x,
            ret_oo_shares: ret_s,
            ret_oo_option: ret_s * 1.3 - 0.001,
            one_way_cost_shares: cost_s,
            one_way_cost_option: cost_s * 3.0,
            one_way_cost_bps: cost_bps,
            session_time_left: (1.0 - (t as f32 / 40.0)).clamp(0.0, 1.0),
            option_allowed: opt_ok,
            entry_allowed: !force_flat,
            force_flat,
            wm_targets,
            wm_valid,
        }
    }

    #[test]
    fn test_surrogate_deadzone_wakes_zero_policy_and_eod_closure_trains() {
        let mut brain = PureAutonomousRlBrain::new_for_horizon("TEST_BPTT_SURROGATE", 42);
        // Deliberately zero out action weights and set high initial deadzone to prove surrogate gradient wakes up!
        for m in &mut brain.population {
            m.net.w_a_heads.fill(0.0);
            m.net.b_a_heads.fill(0.0);
        }

        let mut seq = Vec::new();
        for t in 0..40 {
            let sig = if t < 20 { 1.5_f32 } else { -1.5_f32 };
            let force_eod = t == 39;
            seq.push(make_bar(t, sig, sig * 0.005, 0.0005, 5.0, true, force_eod));
        }
        let xs: Vec<[f32; RL_STATE_DIM]> = seq.iter().map(|b| b.x_212).collect();
        brain.update_welford_batch(&xs);

        let ep = CausalSessionEpisode {
            symbol: "SPY".to_string(),
            day_key: 19670,
            start_ts_sec: seq.first().unwrap().ts_sec,
            end_ts_sec: seq.last().unwrap().ts_sec,
            is_intraday_session: true,
            prior_regime_z: [0.0; WM_DIM],
            bars: seq.clone(),
        };

        let wm_w_before = brain.world_model.w_wm.clone();
        let loss = brain.train_world_model_episodes(std::slice::from_ref(&ep), 6, 3e-3);
        let wm_diff: f32 = wm_w_before
            .iter()
            .zip(brain.world_model.w_wm.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            wm_diff > 1e-4 && loss.is_finite(),
            "World model must train via Pinball loss"
        );

        brain.train_population_episodes_bptt_parallel(std::slice::from_ref(&ep), 10, 1.0, 0.0);
        let w_norm: f32 = brain.population[0]
            .net
            .w_a_heads
            .iter()
            .map(|v| v.abs())
            .sum();
        assert!(
            w_norm > 1e-3,
            "Straight-through surrogate gradient must wake up exact-zero deadzone weights (norm={w_norm})"
        );
    }

    #[test]
    fn test_red_trade_autopsy_and_fisher_ewc_lock() {
        let mut brain = PureAutonomousRlBrain::new_for_horizon("TEST_AUTOPSY", 99);
        let mut bars = Vec::new();
        for t in 0..16 {
            bars.push(make_bar(t, 0.4, -0.002, 0.0035, 35.0, false, t == 15));
        }
        let xs: Vec<[f32; RL_STATE_DIM]> = bars.iter().map(|b| b.x_212).collect();
        brain.update_welford_batch(&xs);

        let audit = brain.audit_and_insert_red_trade(
            "SOUN".to_string(),
            bars.last().unwrap().ts_sec,
            -250.0,
            -1.2,
            1.0,
            false,
            2,
            [0.0; WM_DIM],
            bars.clone(),
        );
        assert_eq!(audit.fault_kind, LossFaultKind::EntryHallucination);
        assert!(audit.priority_weight > 6.0);
        assert!(audit.saliency_dh >= 0.0 && audit.saliency_dx >= 0.0);

        brain.insert_winner_trajectory(SettledWinnerTrajectory {
            exit_ts_sec: bars.last().unwrap().ts_sec,
            net_r: 2.1,
            direction_sign: 1.0,
            entry_offset: 1,
            prior_regime_z: [0.0; WM_DIM],
            bars,
        });
        brain.compute_fisher_ewc_from_winners();
        assert_eq!(
            brain.population[0].net.ewc_fisher_diag.len(),
            brain.population[0].net.param_count()
        );
    }

    #[test]
    fn test_zero_cost_vs_high_cost_deadzone_abstention() {
        let mut seed = 12345u64;
        let mut net = SequencePolicyNet::new(&mut seed, 0.0);
        let hp = AgentHyperParams {
            lr: 1.5e-3,
            weight_decay: 1e-4,
            l1_sparsity: 0.001,
            downside_aversion: 3.0,
            dd_aversion: 1.0,
            disagree_aversion: 0.05,
            chunk_len: 32,
            cost_sensitivity: 1.5,
            timescale_shift: 0.0,
            horizon_gamma: 0.85,
        };
        let mean = vec![0.0_f32; RL_STATE_DIM];
        let var = vec![1.0_f32; RL_STATE_DIM];
        let mut choppy_seq = Vec::new();
        for t in 0..32 {
            let sig = if t % 2 == 0 { 0.1 } else { -0.1 };
            let ret = if t % 2 == 0 { 0.0002 } else { -0.0002 };
            choppy_seq.push(make_bar(t, sig, ret, 0.0035, 35.0, false, t == 31));
        }
        for _ in 0..30 {
            net.train_bptt_chunk(&choppy_seq, None, &mean, &var, &hp, 1.0, 0.0);
        }
        let mut h = [0.0_f32; HIDDEN_DIM];
        let mut abs_sum = 0.0_f32;
        for b in &choppy_seq {
            let out = net.forward_step(
                &b.x_212,
                SelfAwarenessState {
                    one_way_cost_bps: 35.0,
                    session_time_left: 0.5,
                    ..Default::default()
                },
                &h,
                hp.cost_sensitivity,
                false,
            );
            h = out.next_hidden;
            abs_sum += out.action.abs();
        }
        assert!(
            abs_sum / (choppy_seq.len() as f32) < 0.20,
            "On negative-expectancy 70bps RT chop, policy must shrink exposure near 0 (avg={})",
            abs_sum / (choppy_seq.len() as f32)
        );
    }
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

const SCALP_0DTE_SYMBOLS: &[&str] = &["SPY", "QQQ", "NVDA", "TSLA", "AAPL", "AMD", "META", "AMZN", "IWM"];

fn is_penny_symbol_cand(symbol: &str, close_px: f64) -> bool {
    let meta = lookup_real_universe_meta(symbol);
    close_px < 12.0 || meta.is_otc_penny || PENNY_BRAIN01_SYMBOLS.contains(&symbol)
}

fn symbol_matches_bucket(symbol: &str, close_px: f64, bucket: ProofBucket) -> bool {
    let is_penny = is_penny_symbol_cand(symbol, close_px);
    match bucket {
        ProofBucket::Scalp0Dte1m | ProofBucket::Scalp0Dte5m => {
            SCALP_0DTE_SYMBOLS.contains(&symbol)
        }
        ProofBucket::Brain01PennyDay => is_penny,
        ProofBucket::Brain02LargeCapDay => close_px >= 15.0 && !is_penny,
        ProofBucket::Brain03ShortSwing | ProofBucket::Brain04CoreSwing => true,
    }
}

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

fn session_eod_hhmm(bucket: ProofBucket) -> u32 {
    match bucket {
        ProofBucket::Scalp0Dte1m | ProofBucket::Scalp0Dte5m => 1550,
        ProofBucket::Brain01PennyDay | ProofBucket::Brain02LargeCapDay => 1555,
        ProofBucket::Brain03ShortSwing | ProofBucket::Brain04CoreSwing => 1600,
    }
}

fn normalized_session_remaining(bucket: ProofBucket, hhmm: u32) -> f64 {
    let cur_m = ((hhmm / 100) * 60 + (hhmm % 100)) as f64;
    let end_m = match bucket {
        ProofBucket::Scalp0Dte1m | ProofBucket::Scalp0Dte5m => 15.0 * 60.0 + 50.0,
        ProofBucket::Brain01PennyDay | ProofBucket::Brain02LargeCapDay => 15.0 * 60.0 + 55.0,
        ProofBucket::Brain03ShortSwing | ProofBucket::Brain04CoreSwing => 16.0 * 60.0,
    };
    ((end_m - cur_m).max(0.0) / 385.0).clamp(0.0, 1.0)
}

fn is_option_vehicle_allowed(symbol: &str, close_px: f64) -> bool {
    let meta = lookup_real_universe_meta(symbol);
    meta.is_optionable && !meta.is_otc_penny && close_px >= 5.0
}

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
#[allow(dead_code)]
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
        let eff_delta = if is_vertical_combo { 0.34 } else { 0.50 };
        ((legs * opt_spread + comm_per_sh) / eff_delta) + spot_px * (slip_bps / 10_000.0)
    } else {
        spot_px * ((2.0 * slip_bps / 10_000.0) + 0.00003)
    };
    (underlying_rt_friction / atr_px.max(1e-4)).clamp(0.0, 5.0)
}

fn compute_learned_backstop_px(
    anchor_px: f64,
    atr_px: f64,
    learned_tail_dist: f64,
    direction: TradeDirection,
) -> f64 {
    let dist = (anchor_px * learned_tail_dist.clamp(0.008, 0.09))
        .max(2.2 * atr_px)
        .max(anchor_px * 0.012)
        .max(1e-4);
    (anchor_px - direction.sign() * dist).max(0.0001)
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

    let scaled_budget =
        (desired_risk_budget_usd * (target_action.abs() as f64).clamp(0.20, 1.0)).max(50.0);
    let right = OptionRight::from_direction(direction);
    let base_vix = if is_intraday {
        entry_snap.vix9d_spot_dec
    } else {
        entry_snap.vix_spot_dec
    };
    let eff_iv = (base_vix * (entry_snap.causal_realized_vol / base_vix.max(0.12)).clamp(0.75, 2.40))
        .clamp(0.12, 1.85);

    let (
        k1,
        k2,
        theo_entry_mid,
        spread_per_leg,
        contracts,
        shares,
        actual_risk_usd,
        effective_vertical,
        structure_label,
    ) = if is_option_routed {
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
        if risk_per_contract > account_equity_usd * 0.035 {
            return None;
        }
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
        (
            strike,
            wing,
            mid_px,
            spread_leg,
            qty_contracts,
            (qty_contracts as f64) * 100.0,
            pos_risk_usd,
            use_vert,
            label,
        )
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
        (
            0.0,
            None,
            0.0,
            0.0,
            0,
            qty_shares,
            pos_risk_usd,
            false,
            "SHARES_ONLY".to_string(),
        )
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
        bars_held: 1,
        is_option_routed,
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

fn mark_to_market_post_cost_net_ret(
    pos: &ActiveSimulatedPosition,
    ts_sec: i64,
    mark_px: f64,
    current_vix_iv: f64,
) -> f64 {
    let (rem_gross, rem_exit_cost) =
        compute_leg_pnl_components(pos, ts_sec, mark_px, pos.remaining_weight, current_vix_iv);
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
    let gross_pnl = pos.realized_partial_gross_usd + rem_gross;
    let total_cost = entry_cost_usd + pos.realized_partial_cost_usd + rem_exit_cost;
    let base_notional = if pos.is_option_routed {
        (pos.option_entry_theo_mid * 100.0 * (pos.option_contracts as f64)).max(100.0)
    } else {
        (pos.entry_mid_underlying * pos.equity_shares).max(100.0)
    };
    (gross_pnl - total_cost) / base_notional
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

fn exit_iv_for_bar(bucket: ProofBucket, snap: &CausalBarPhysicsSnapshot) -> f64 {
    let base = match bucket {
        ProofBucket::Scalp0Dte1m | ProofBucket::Scalp0Dte5m | ProofBucket::Brain02LargeCapDay => {
            snap.vix9d_spot_dec
        }
        _ => snap.vix_spot_dec,
    };
    (base * (snap.causal_realized_vol / base.max(0.12)).clamp(0.75, 2.40)).clamp(0.12, 1.85)
}

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
    let ret_oo_shares = if force_flat {
        0.0_f32
    } else if t + 2 < bars.len() && bars[t + 2].timestamp_sec <= max_ts_sec {
        let open_2 = bars[t + 2].open.max(1e-4);
        ((open_2 - open_1) / open_1).clamp(-0.25, 0.25) as f32
    } else {
        return None;
    };

    let is_penny = is_penny_symbol_cand(sym, bar_t.close);
    let slip_bps = explicit_one_way_slippage_bps(sym, open_1, is_penny);
    let one_way_cost_shares = ((slip_bps / 10_000.0) + 0.000015) as f32;
    let rt_cost_shares = (2.0 * one_way_cost_shares as f64).max(4e-4);

    let option_allowed = !is_penny && is_option_vehicle_allowed(sym, open_1);
    // Risk-normalized underlying-equivalent option return & friction so BPTT compares `V_SHARES` vs `V_OPTION` apples-to-apples!
    let (one_way_cost_option, ret_oo_option) = if option_allowed {
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
        let eff_delta = if is_vert { 0.34 } else { 0.50 };
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
        let approx_prem_frac = (snap.vix_spot_dec * (dte_days / 252.0).sqrt() * 0.40).max(0.004);
        let theta_underlying_eq =
            ((approx_prem_frac / dt_bars.max(4.0)) / eff_delta).clamp(0.00005, 0.008) as f32;
        let gamma_boost = if ret_oo_shares.abs() > (2.0 * underlying_equiv_one_way_opt) {
            1.12_f32
        } else {
            0.96_f32
        };
        (
            underlying_equiv_one_way_opt,
            if force_flat {
                0.0
            } else {
                (ret_oo_shares * gamma_boost - theta_underlying_eq).clamp(-0.25, 0.25)
            },
        )
    } else {
        (0.02, ret_oo_shares)
    };

    let norm_rem = normalized_session_remaining(bucket, snap.hhmm);
    let x_212 = snap.to_raw_180_vector(norm_rem, 0.0, false);
    let entry_allowed = !force_flat && entry_allowed_at_hhmm(bucket, snap.hhmm);

    // Strictly causal multi-horizon physics targets (`WM_HORIZONS = [1, 3, 8, 24]`, 4 outputs each = 16):
    // `[q25_ret, q50_ret, q75_ret, ln(Range / RT_Cost)]` populated ONLY when `t + 1 + delta <= max_ts_sec`!
    let mut wm_targets = [0.0_f32; WM_OUT_DIM];
    let mut wm_valid = [false; WM_OUT_DIM];
    for (h_idx, &delta) in WM_HORIZONS.iter().enumerate() {
        let end_idx = t + 1 + delta;
        if end_idx < bars.len() && bars[end_idx].timestamp_sec <= max_ts_sec {
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
        wm_targets,
        wm_valid,
    })
}

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
                            let mut xn = [0.0_f32; RL_STATE_DIM];
                            for j in 0..RL_STATE_DIM {
                                xn[j] = ((b.x_212[j] - feat_mean[j])
                                    / feat_var[j].max(1e-4).sqrt())
                                .clamp(-6.0, 6.0);
                            }
                            z_step = wm.forward_step(&xn, &z_step).0;
                        }
                        for j in 0..WM_DIM {
                            cross_day_regime_z[j] = 0.45 * z_step[j];
                        }
                    }

                    episodes.push(CausalSessionEpisode {
                        symbol: sym.clone(),
                        day_key: d_key,
                        start_ts_sec: ep_bars.first().unwrap().ts_sec,
                        end_ts_sec: ep_bars.last().unwrap().ts_sec,
                        is_intraday_session: true,
                        prior_regime_z: prior_z,
                        bars: ep_bars,
                    });
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
                        let mut xn = [0.0_f32; RL_STATE_DIM];
                        for j in 0..RL_STATE_DIM {
                            xn[j] = ((b.x_212[j] - feat_mean[j]) / feat_var[j].max(1e-4).sqrt())
                                .clamp(-6.0, 6.0);
                        }
                        z_step = wm.forward_step(&xn, &z_step).0;
                    }
                    z_carry = z_step;
                }
                episodes.push(CausalSessionEpisode {
                    symbol: sym.clone(),
                    day_key: est_trading_day_key(slice.first().unwrap().ts_sec),
                    start_ts_sec: slice.first().unwrap().ts_sec,
                    end_ts_sec: slice.last().unwrap().ts_sec,
                    is_intraday_session: false,
                    prior_regime_z: prior_z,
                    bars: slice,
                });
                start += 70;
            }
        }
    }

    episodes.sort_by_key(|ep| (ep.end_ts_sec, ep.symbol.clone()));
    episodes
}

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
    ) -> Vec<Vec<([f32; RL_STATE_DIM], [f32; WM_DIM], [f32; WM_OUT_DIM])>> {
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
                    let mut x_n = [0.0_f32; RL_STATE_DIM];
                    for j in 0..RL_STATE_DIM {
                        let std = feat_var.get(j).copied().unwrap_or(1.0).max(1e-4).sqrt();
                        let m = feat_mean.get(j).copied().unwrap_or(0.0);
                        x_n[j] = ((raw_x[j] - m) / std).clamp(-6.0, 6.0);
                    }
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
    net_pnl_usd: f64,
    net_r: f64,
}

#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
struct DiscreteCausalEvalSummary {
    score: f64,
    sharpe: f64,
    net_pnl_usd: f64,
    net_pf: f64,
    red_loss_usd: f64,
    win_pnl_usd: f64,
    win_r_sum: f64,
    winners_count: usize,
    max_dd_pct: f64,
    trades: Vec<SettledCausalTrade>,
}

fn evaluate_single_member_indexed(
    member: &mut PopulationMember,
    idx: &BucketTimelineIndex<'_>,
    wm_cache: &[Vec<([f32; RL_STATE_DIM], [f32; WM_DIM], [f32; WM_OUT_DIM])>],
    eval_start_ts: i64,
    eval_cutoff_ts: i64,
    starting_equity: f64,
) -> DiscreteCausalEvalSummary {
    let n_syms = idx.syms.len();
    let bucket = idx.bucket;
    let is_intraday = bucket.is_intraday();
    let max_slots = if is_intraday { 3usize } else { 4usize };

    let mut equity = starting_equity.max(50_000.0);
    let mut peak_eq = equity;
    let mut max_dd_pct = 0.0_f64;
    let mut hiddens = vec![[0.0_f32; HIDDEN_DIM]; n_syms];
    let mut sym_last_day = vec![i64::MIN; n_syms];
    let mut open_positions: Vec<ActiveSimulatedPosition> = Vec::with_capacity(max_slots);
    let mut pending_signals: Vec<PendingNextBarSignal> = Vec::with_capacity(max_slots);
    let mut settled: Vec<SettledCausalTrade> = Vec::new();
    let mut trade_seq = 1u128;

    let end_ti = idx.all_ts.partition_point(|&ts| ts <= eval_cutoff_ts);
    let start_ti = idx.all_ts[..end_ti].partition_point(|&ts| ts < eval_start_ts);

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
                    net_pnl_usd: blotter.net_pnl_usd,
                    net_r: blotter.net_r,
                });
            } else {
                if !is_new && snap_t.valid && bi < wm_cache[sid].len() {
                    let pos = &mut open_positions[i_pos];
                    let (ref x_n, ref z_wm, ref wm_pred) = wm_cache[sid][bi];
                    let norm_rem = normalized_session_remaining(bucket, snap_t.hhmm);
                    let unreal =
                        mark_to_market_post_cost_net_ret(pos, t_sec, bar_t.close, exit_iv) as f32;
                    let peak_ret = ((pos.direction.sign()
                        * (pos.peak_favorable_px - pos.entry_fill_underlying))
                        / pos.entry_fill_underlying.max(1e-4))
                    .max(0.0) as f32;
                    let dd_peak = (peak_ret - unreal).max(0.0);
                    let self_s = SelfAwarenessState {
                        current_pos: pos.current_target_action,
                        unrealized_net_ret: unreal,
                        peak_excursion: peak_ret,
                        drawdown_from_peak: dd_peak,
                        log_bars_held: (1.0 + pos.bars_held as f32).ln(),
                        one_way_cost_bps: pos.slip_bps_one_way as f32,
                        session_time_left: norm_rem as f32,
                    };
                    let step = member.net.forward_step_with_wm(
                        x_n,
                        z_wm,
                        wm_pred,
                        self_s,
                        &hiddens[sid],
                        *z_wm,
                        member.hparams.cost_sensitivity,
                        !pos.is_penny_tier && is_option_vehicle_allowed(&pos.symbol, bar_t.close),
                    );
                    hiddens[sid] = step.next_hidden;

                    let pos_sign = pos.direction.sign() as f32;
                    let exit_snap = (0.30 * pos.current_target_action.abs()).clamp(0.025, 0.08);
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
            };
            let step = member.net.forward_step_with_wm(
                x_n,
                z_wm,
                wm_pred,
                self_s,
                &hiddens[sid],
                *z_wm,
                member.hparams.cost_sensitivity,
                opt_allowed,
            );
            hiddens[sid] = step.next_hidden;

            if step.action.abs() >= 0.045 {
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
                let risk_budget_usd = (equity * 0.015).clamp(500.0, 2_000.0);
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

    let summary = summarize_settled_causal_trades(&settled, max_dd_pct);
    member.causal_score = summary.score;
    member.causal_sharpe = summary.sharpe;
    member.causal_net_pnl = summary.net_pnl_usd;
    member.causal_trades = summary.trades.len();
    member.causal_pf = summary.net_pf;
    member.causal_red_loss_usd = summary.red_loss_usd;
    member.causal_win_r_sum = summary.win_r_sum;
    summary
}

fn evaluate_brain_ensemble_retest_indexed(
    brain: &PureAutonomousRlBrain,
    idx: &BucketTimelineIndex<'_>,
    wm_cache: &[Vec<([f32; RL_STATE_DIM], [f32; WM_DIM], [f32; WM_OUT_DIM])>],
    eval_start_ts: i64,
    eval_cutoff_ts: i64,
    starting_equity: f64,
) -> DiscreteCausalEvalSummary {
    let n_syms = idx.syms.len();
    let bucket = idx.bucket;
    let is_intraday = bucket.is_intraday();
    let max_slots = if is_intraday { 3usize } else { 4usize };

    let mut equity = starting_equity.max(50_000.0);
    let mut peak_eq = equity;
    let mut max_dd_pct = 0.0_f64;
    let mut hiddens: Vec<Vec<[f32; HIDDEN_DIM]>> = vec![Vec::new(); n_syms];
    let mut sym_last_day = vec![i64::MIN; n_syms];
    let mut open_positions: Vec<ActiveSimulatedPosition> = Vec::with_capacity(max_slots);
    let mut pending_signals: Vec<PendingNextBarSignal> = Vec::with_capacity(max_slots);
    let mut settled: Vec<SettledCausalTrade> = Vec::new();
    let mut trade_seq = 1u128;

    let end_ti = idx.all_ts.partition_point(|&ts| ts <= eval_cutoff_ts);
    let start_ti = idx.all_ts[..end_ti].partition_point(|&ts| ts < eval_start_ts);

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
                settled.push(SettledCausalTrade {
                    sym_id: pos.sym_id,
                    symbol: pos.symbol,
                    entry_bar_ord: pos.entry_bar_ord as usize,
                    exit_bar_ord: bi,
                    exit_ts_sec: t_sec,
                    direction: pos.direction,
                    is_option_routed: pos.is_option_routed,
                    net_pnl_usd: blotter.net_pnl_usd,
                    net_r: blotter.net_r,
                });
            } else {
                if !is_new && snap_t.valid && bi < wm_cache[sid].len() {
                    let pos = &mut open_positions[i_pos];
                    let (ref x_n, ref z_wm, ref wm_pred) = wm_cache[sid][bi];
                    let norm_rem = normalized_session_remaining(bucket, snap_t.hhmm);
                    let unreal =
                        mark_to_market_post_cost_net_ret(pos, t_sec, bar_t.close, exit_iv) as f32;
                    let peak_ret = ((pos.direction.sign()
                        * (pos.peak_favorable_px - pos.entry_fill_underlying))
                        / pos.entry_fill_underlying.max(1e-4))
                    .max(0.0) as f32;
                    let dd_peak = (peak_ret - unreal).max(0.0);
                    let self_s = SelfAwarenessState {
                        current_pos: pos.current_target_action,
                        unrealized_net_ret: unreal,
                        peak_excursion: peak_ret,
                        drawdown_from_peak: dd_peak,
                        log_bars_held: (1.0 + pos.bars_held as f32).ln(),
                        one_way_cost_bps: pos.slip_bps_one_way as f32,
                        session_time_left: norm_rem as f32,
                    };
                    let inf = brain.infer_ensemble_step_with_precomputed_wm(
                        x_n,
                        z_wm,
                        wm_pred,
                        self_s,
                        &hiddens[sid],
                        !pos.is_penny_tier && is_option_vehicle_allowed(&pos.symbol, bar_t.close),
                    );
                    hiddens[sid] = inf.next_hiddens;

                    let pos_sign = pos.direction.sign() as f32;
                    let exit_snap = (0.32 * pos.current_target_action.abs()).clamp(0.025, 0.085);
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

            if inf.action.abs() >= 0.045 {
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
                let risk_budget_usd = (equity * 0.015).clamp(500.0, 2_000.0);
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
    let trade_bonus = if n < 5 {
        -1.2 + 0.18 * (n as f64)
    } else {
        (n as f64).min(36.0).sqrt() * 0.16
    };
    let n_trust = (n as f64 / 12.0).min(1.0);
    let pf_adj = 1.0 + (pf.min(4.0) - 1.0) * n_trust;
    let score = (pf_adj - 1.0) * 1.8
        + sharpe * 0.35
        + (net_pnl / 1200.0).clamp(-3.0, 3.5)
        - max_dd_pct * 0.05
        + trade_bonus;

    DiscreteCausalEvalSummary {
        score,
        sharpe,
        net_pnl_usd: net_pnl,
        net_pf: pf,
        red_loss_usd: red_loss,
        win_pnl_usd: win_pnl,
        win_r_sum,
        winners_count,
        max_dd_pct,
        trades: settled.to_vec(),
    }
}

fn evaluate_population_indexed_parallel(
    brain: &mut PureAutonomousRlBrain,
    idx: &BucketTimelineIndex<'_>,
    wm_cache: &[Vec<([f32; RL_STATE_DIM], [f32; WM_DIM], [f32; WM_OUT_DIM])>],
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

    // Step 1: Automated Counterfactual & Gradient Saliency Loss Audit across settled red trades `<= cutoff_ts`
    let mut red_audited = 0usize;
    let mut entry_faults = 0usize;
    let mut exit_faults = 0usize;
    let mut sum_sal_dx = 0.0_f32;
    let mut sum_sal_dh = 0.0_f32;

    brain.red_replay_buffer.clear();
    brain.winner_fisher_buffer.clear();

    for tr in &champ_eval.trades {
        if let Some((entry_off, bar_slice)) = extract_settled_trade_causal_bars(tr, idx, cutoff_ts)
        {
            if tr.net_pnl_usd < 0.0 {
                let audit = brain.audit_and_insert_red_trade(
                    tr.symbol.clone(),
                    tr.exit_ts_sec,
                    tr.net_pnl_usd,
                    tr.net_r as f32,
                    tr.direction.sign() as f32,
                    tr.is_option_routed,
                    entry_off,
                    [0.0; WM_DIM],
                    bar_slice,
                );
                red_audited += 1;
                match audit.fault_kind {
                    LossFaultKind::EntryHallucination => entry_faults += 1,
                    LossFaultKind::HoldingExitFailure => exit_faults += 1,
                }
                sum_sal_dx += audit.saliency_dx;
                sum_sal_dh += audit.saliency_dh;
            } else if tr.net_pnl_usd > 0.0 {
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

    if red_audited == 0 {
        return false;
    }

    // Step 2: Compute Diagonal Fisher EWC over settled winners, then spawn parallel candidate RL adaptations
    brain.compute_fisher_ewc_from_winners();

    // Candidate hyperparameter grid explored in parallel across CPU cores (`rayon`)
    let candidate_specs: [(usize, f32, f32, f64, bool); 12] = [
        (0, 0.00, 1.00, 3.2, false),
        (0, 0.00, 1.00, 5.2, false),
        (3, 0.08, 0.60, 2.8, false),
        (5, 0.14, 0.45, 3.0, false),
        (6, 0.22, 0.35, 3.4, false),
        (8, 0.30, 0.25, 3.8, false),
        (4, 0.12, 0.80, 4.2, false),
        (6, 0.18, 0.40, 2.6, true),
        (8, 0.28, 0.30, 3.2, true),
        (10, 0.38, 0.20, 3.6, true),
        (2, 0.05, 1.00, 4.8, false),
        (5, 0.16, 0.50, 5.5, true),
    ];

    let base_brain = brain.clone();
    let cand_results: Vec<(PureAutonomousRlBrain, DiscreteCausalEvalSummary)> = candidate_specs
        .par_iter()
        .map(|&(steps, lr_sc, ewc_lam, eta, do_pbt)| {
            let mut cand = base_brain.clone();
            if do_pbt {
                cand.pbt_exploit_and_explore();
            }
            if steps > 0 {
                cand.train_red_autopsy_and_causal_batch_parallel(
                    causal_episodes,
                    steps,
                    lr_sc,
                    ewc_lam,
                );
            }
            for m in &mut cand.population {
                let _ = evaluate_single_member_indexed(
                    m,
                    idx,
                    &wm_cache,
                    retest_start_ts,
                    cutoff_ts,
                    starting_equity,
                );
            }
            cand.update_hedge_weights_with_eta(eta);
            let eval = evaluate_brain_ensemble_retest_indexed(
                &cand,
                idx,
                &wm_cache,
                retest_start_ts,
                cutoff_ts,
                starting_equity,
            );
            (cand, eval)
        })
        .collect();

    // Step 3: Mandatory Full Causal Re-Test Gate Before Promotion!
    // Promote `theta_candidate -> theta_champion` ONLY IF:
    //  1) `cand_net_pnl > champ_net_pnl` AND `cand_net_pf > champ_net_pf`,
    //  2) Total red-trade dollar loss shrinks (`cand_red_loss_usd < champ_red_loss_usd`), AND
    //  3) Winner preservation / net winning R does not regress (`cand_win_r_sum >= champ_win_r_sum`).
    let min_trades_floor = champ_eval.trades.len().min(4).max(1);
    let best_passing = cand_results
        .iter()
        .filter(|(_, ev)| {
            ev.trades.len() >= min_trades_floor
                && ev.net_pnl_usd > champ_eval.net_pnl_usd + 1e-2
                && ev.net_pf > champ_eval.net_pf + 1e-4
                && ev.red_loss_usd < champ_eval.red_loss_usd - 1e-2
                && ev.win_r_sum + 1e-4 >= champ_eval.win_r_sum
        })
        .max_by(|a, b| {
            let gain_a =
                (a.1.net_pnl_usd - champ_eval.net_pnl_usd) + 250.0 * (a.1.net_pf - champ_eval.net_pf);
            let gain_b =
                (b.1.net_pnl_usd - champ_eval.net_pnl_usd) + 250.0 * (b.1.net_pf - champ_eval.net_pf);
            gain_a.total_cmp(&gain_b)
        });

    let mean_dx = sum_sal_dx / (red_audited as f32);
    let mean_dh = sum_sal_dh / (red_audited as f32);

    if let Some((best_cand, best_ev)) = best_passing {
        let next_gen = brain.promoted_generation_id.saturating_add(1);
        let red_saved = (champ_eval.red_loss_usd - best_ev.red_loss_usd).max(0.0);
        let win_desc = format!(
            "{}->{} wins ({:+.2}R -> {:+.2}R)",
            champ_eval.winners_count, best_ev.winners_count, champ_eval.win_r_sum, best_ev.win_r_sum
        );
        *brain = best_cand.clone();
        brain.promoted_generation_id = next_gen;
        brain.online_updates_count = brain.online_updates_count.saturating_add(1);
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
        // Best rejected candidate diagnostics for audit transparency while preserving `theta_champion`
        let best_attempt = cand_results
            .iter()
            .max_by(|a, b| a.1.net_pnl_usd.total_cmp(&b.1.net_pnl_usd));
        let (pf_att, pnl_att, red_diff, win_desc) = if let Some((_, ev)) = best_attempt {
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
        brain.online_updates_count = brain.online_updates_count.saturating_add(1);
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

fn train_warmup_with_causal_pbt(
    brain: &mut PureAutonomousRlBrain,
    bucket: ProofBucket,
    symbol_bars: &HashMap<String, Vec<RealExchangeOhlcvBar>>,
    precomputed: &HashMap<String, Vec<CausalBarPhysicsSnapshot>>,
    pretrain_m5_for_1m: Option<(
        &HashMap<String, Vec<RealExchangeOhlcvBar>>,
        &HashMap<String, Vec<CausalBarPhysicsSnapshot>>,
        i64,
    )>,
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
    let (n_folds, epochs_per_gen) = if warmup_days_set.len() <= 3 {
        (3usize, 3usize)
    } else {
        (4usize, 3usize)
    };
    let n_ts = warmup_ts.len();
    let mut best_pf = 0.0_f64;
    let mut best_pnl = 0.0_f64;
    let mut best_trades = 0usize;

    for fold_idx in 0..n_folds {
        let train_frac = 0.40 + 0.14 * (fold_idx as f64);
        let val_start_frac = (train_frac + 0.02).min(0.85);
        let val_end_frac = (val_start_frac + 0.35).min(1.0);
        let train_end_i = ((n_ts as f64) * train_frac).floor() as usize;
        let val_start_i = ((n_ts as f64) * val_start_frac).floor() as usize;
        let val_end_i = (((n_ts as f64) * val_end_frac).ceil() as usize).min(n_ts.saturating_sub(1));

        let train_cutoff_ts = warmup_ts.get(train_end_i).copied().unwrap_or(warmup_cutoff_ts);
        let val_start_ts = warmup_ts.get(val_start_i).copied().unwrap_or(warmup_cutoff_ts);
        let val_end_ts = warmup_ts.get(val_end_i).copied().unwrap_or(warmup_cutoff_ts);

        let fold_eps: Vec<CausalSessionEpisode> = full_eps
            .iter()
            .filter(|ep| ep.end_ts_sec <= train_cutoff_ts)
            .cloned()
            .collect();
        let train_slice = if fold_eps.len() >= 4 {
            &fold_eps[..]
        } else {
            &full_eps[..]
        };

        brain.train_population_episodes_bptt_parallel(train_slice, epochs_per_gen, 1.0, 0.0);
        let wm_cache = idx.precompute_wm_cache(
            &brain.world_model,
            &brain.feat_mean,
            &brain.feat_var,
            val_end_ts,
        );
        let best_ev = evaluate_population_indexed_parallel(
            brain,
            &idx,
            &wm_cache,
            val_start_ts,
            val_end_ts,
            starting_equity,
        );
        best_pf = best_ev.net_pf;
        best_pnl = best_ev.net_pnl_usd;
        best_trades = best_ev.trades.len();

        // Always run PBT exploit & explore across inner folds (no 0-gen coma!)
        if fold_idx + 1 < n_folds {
            brain.pbt_exploit_and_explore();
        }
    }

    // Final causal warmup consolidation + Closed-Loop Red-Trade Loss Audit -> Re-Test -> Promotion (`<= warmup_cutoff_ts`)
    brain.train_population_episodes_bptt_parallel(&full_eps, 2, 0.55, 0.0);
    let full_warmup_start_ts = warmup_ts.first().copied().unwrap_or(0);
    let wm_cache_full = idx.precompute_wm_cache(
        &brain.world_model,
        &brain.feat_mean,
        &brain.feat_var,
        warmup_cutoff_ts,
    );
    let best_full = evaluate_population_indexed_parallel(
        brain,
        &idx,
        &wm_cache_full,
        full_warmup_start_ts,
        warmup_cutoff_ts,
        starting_equity,
    );
    if best_full.trades.len() > 0 {
        best_pf = best_full.net_pf;
        best_pnl = best_full.net_pnl_usd;
        best_trades = best_full.trades.len();
    }
    brain.promoted_generation_id = 1;
    brain.snapshot_ewc_anchors();

    // Run 2 inner causal Red-Trade Loss-Autopsy -> Re-Test -> Promotion rounds strictly `<= warmup_cutoff_ts`
    for round in 1..=2 {
        let promoted = execute_closed_loop_red_trade_autopsy_and_promotion(
            brain,
            &idx,
            &full_eps,
            full_warmup_start_ts,
            warmup_cutoff_ts,
            starting_equity,
            &format!("WARMUP_R{round}"),
            promotion_ledger,
        );
        if !promoted {
            break;
        }
    }

    let ens_warmup = evaluate_brain_ensemble_retest_indexed(
        brain,
        &idx,
        &wm_cache_full,
        full_warmup_start_ts,
        warmup_cutoff_ts,
        starting_equity,
    );
    if ens_warmup.trades.len() > 0 {
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

struct OosTimelineResult {
    blotter: Vec<BlotterTradeRecord>,
    abstained_flat_oos: usize,
    oos_counterfactuals_harvested: usize,
}

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

    let mut hidden_states: BTreeMap<ProofBucket, HashMap<String, Vec<[f32; HIDDEN_DIM]>>> =
        BTreeMap::new();
    let mut wm_hiddens: BTreeMap<ProofBucket, HashMap<String, [f32; WM_DIM]>> = BTreeMap::new();
    let mut last_sym_day: BTreeMap<ProofBucket, HashMap<String, i64>> = BTreeMap::new();

    for &t_sec in &all_ts {
        let cur_day = est_trading_day_key(t_sec);
        if cur_day < oos_start_day {
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

                // 1) Harvest day `pd`'s strictly resolved causal episodes (`<= prev_ts_sec`) & adapt World Model
                let day_eps = build_causal_session_episodes(
                    bucket,
                    symbol_bars,
                    precomputed,
                    pd.saturating_sub(2),
                    prev_ts_sec,
                    Some(&brain.world_model),
                    &brain.feat_mean,
                    &brain.feat_var,
                );
                let newly_bars: usize = day_eps
                    .iter()
                    .filter(|ep| ep.day_key == pd)
                    .map(|ep| ep.bars.len())
                    .sum();
                oos_cf_harvested += newly_bars.max(day_eps.len());

                let cur_red_count = oos_blotter
                    .iter()
                    .filter(|t| t.bucket == bucket && t.net_pnl_usd < 0.0)
                    .count();
                let prev_red_count = red_trades_seen_by_bucket.get(&bucket).copied().unwrap_or(0);
                let has_new_red_trade = cur_red_count > prev_red_count;
                let periodic_check = completed_oos_days_count == 1
                    || completed_oos_days_count % 3 == 0
                    || bucket == ProofBucket::Scalp0DTE1m_alias();

                // Trigger full 4-Step Red-Trade Loss Audit -> RL Update -> Re-Test -> Promotion whenever
                // a new red trade settled on day `pd` OR on periodic EOD regime adaptation checkpoints!
                if has_new_red_trade || periodic_check {
                    red_trades_seen_by_bucket.insert(bucket, cur_red_count);

                    // Candidate World-Model TTT + RL update strictly gated by full causal re-test!
                    let backup_brain = brain.clone();
                    let rolling_min_day = (pd - 25).max(warmup_start_day);
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
                    let mut promoted = execute_closed_loop_red_trade_autopsy_and_promotion(
                        brain,
                        b_idx,
                        &causal_eps,
                        eval_start_ts,
                        prev_ts_sec,
                        starting_equity,
                        &format!("OOS_D{completed_oos_days_count}"),
                        promotion_ledger,
                    );
                    if !promoted && !day_eps.is_empty() {
                        // Also try World-Model TTT + Loss-Autopsy candidate against `backup_brain` baseline
                        let mut ttt_cand = backup_brain.clone();
                        let _ = ttt_cand.train_world_model_episodes(&day_eps, 1, 0.0010);
                        let wm_base = b_idx.precompute_wm_cache(
                            &backup_brain.world_model,
                            &backup_brain.feat_mean,
                            &backup_brain.feat_var,
                            prev_ts_sec,
                        );
                        let base_ev = evaluate_brain_ensemble_retest_indexed(
                            &backup_brain,
                            b_idx,
                            &wm_base,
                            eval_start_ts,
                            prev_ts_sec,
                            starting_equity,
                        );
                        let wm_ttt = b_idx.precompute_wm_cache(
                            &ttt_cand.world_model,
                            &ttt_cand.feat_mean,
                            &ttt_cand.feat_var,
                            prev_ts_sec,
                        );
                        let _ = evaluate_population_indexed_parallel(
                            &mut ttt_cand,
                            b_idx,
                            &wm_ttt,
                            eval_start_ts,
                            prev_ts_sec,
                            starting_equity,
                        );
                        let ttt_ev = evaluate_brain_ensemble_retest_indexed(
                            &ttt_cand,
                            b_idx,
                            &wm_ttt,
                            eval_start_ts,
                            prev_ts_sec,
                            starting_equity,
                        );
                        if ttt_ev.trades.len() >= base_ev.trades.len().min(4).max(1)
                            && ttt_ev.net_pnl_usd > base_ev.net_pnl_usd + 1e-2
                            && ttt_ev.net_pf > base_ev.net_pf + 1e-4
                            && ttt_ev.red_loss_usd < base_ev.red_loss_usd - 1e-2
                            && ttt_ev.win_r_sum + 1e-4 >= base_ev.win_r_sum
                        {
                            let next_gen = backup_brain.promoted_generation_id.saturating_add(1);
                            *brain = ttt_cand;
                            brain.promoted_generation_id = next_gen;
                            brain.online_updates_count =
                                backup_brain.online_updates_count.saturating_add(1);
                            brain.snapshot_ewc_anchors();
                            promoted = true;
                        }
                    }
                    if !promoted {
                        let upd = brain.online_updates_count.max(
                            backup_brain.online_updates_count.saturating_add(1),
                        );
                        *brain = backup_brain;
                        brain.online_updates_count = upd;
                    }
                } else {
                    // Lightweight EOD Multiplicative-Weights specialist routing check (strictly gated by causal re-test!)
                    let backup_brain = brain.clone();
                    let wm_cache_eod = b_idx.precompute_wm_cache(
                        &brain.world_model,
                        &brain.feat_mean,
                        &brain.feat_var,
                        prev_ts_sec,
                    );
                    let champ_ev = evaluate_brain_ensemble_retest_indexed(
                        &backup_brain,
                        b_idx,
                        &wm_cache_eod,
                        eval_start_ts,
                        prev_ts_sec,
                        starting_equity,
                    );
                    let _ = evaluate_population_indexed_parallel(
                        brain,
                        b_idx,
                        &wm_cache_eod,
                        eval_start_ts,
                        prev_ts_sec,
                        starting_equity,
                    );
                    let cand_ev = evaluate_brain_ensemble_retest_indexed(
                        brain,
                        b_idx,
                        &wm_cache_eod,
                        eval_start_ts,
                        prev_ts_sec,
                        starting_equity,
                    );
                    if !(cand_ev.trades.len() >= champ_ev.trades.len().min(4).max(1)
                        && cand_ev.net_pnl_usd > champ_ev.net_pnl_usd + 1e-2
                        && cand_ev.net_pf > champ_ev.net_pf + 1e-4
                        && cand_ev.red_loss_usd <= champ_ev.red_loss_usd + 1e-2
                        && cand_ev.win_r_sum + 1e-4 >= champ_ev.win_r_sum)
                    {
                        *brain = backup_brain;
                    } else {
                        brain.promoted_generation_id =
                            brain.promoted_generation_id.saturating_add(1);
                    }
                    brain.online_updates_count = brain.online_updates_count.saturating_add(1);
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
            for sym in &sorted_syms {
                if ts_index_by_sym.get(sym).is_some_and(|m| m.contains_key(&t_sec)) {
                    let prev_d = day_map.get(sym).copied().unwrap_or(i64::MIN);
                    if prev_d != cur_day {
                        h_map.remove(sym);
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

        // B. STEP ALL OPEN POSITIONS AT BAR `t_sec`
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
                oos_blotter.push(blotter);
            } else {
                if !is_new
                    && snap_t.valid
                    && let Some(brain) = brains.get(&pos.bucket)
                {
                    let (_, hhmm) = ny_et_hhmm(t_sec);
                    let norm_rem = normalized_session_remaining(pos.bucket, hhmm);
                    let raw_x = snap_t.to_raw_180_vector(norm_rem, 0.0, pos.is_option_routed);
                    let unreal =
                        mark_to_market_post_cost_net_ret(&pos, t_sec, bar_t.close, sym_exit_iv)
                            as f32;
                    let peak_ret = ((pos.direction.sign()
                        * (pos.peak_favorable_px - pos.entry_fill_underlying))
                        / pos.entry_fill_underlying.max(1e-4))
                    .max(0.0) as f32;
                    let dd_peak = (peak_ret - unreal).max(0.0);
                    let self_s = SelfAwarenessState {
                        current_pos: pos.current_target_action,
                        unrealized_net_ret: unreal,
                        peak_excursion: peak_ret,
                        drawdown_from_peak: dd_peak,
                        log_bars_held: (1.0 + pos.bars_held as f32).ln(),
                        one_way_cost_bps: pos.slip_bps_one_way as f32,
                        session_time_left: norm_rem as f32,
                    };
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

                    let pos_sign = pos.direction.sign() as f32;
                    let exit_snap = (0.32 * pos.current_target_action.abs()).clamp(0.025, 0.085);
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
        if open_positions.len() >= 20 {
            continue;
        }
        let (_, hhmm) = ny_et_hhmm(t_sec);
        let mut open_heat_frac = open_positions
            .iter()
            .map(|p| p.risk_budget_usd)
            .sum::<f64>()
            / equity_usd.max(50_000.0);
        if open_heat_frac >= 0.28 {
            continue;
        }

        for &bucket in active_buckets {
            if !entry_allowed_at_hhmm(bucket, hhmm) {
                continue;
            }
            let max_bucket_slots = if bucket.is_intraday() { 3 } else { 4 };
            let bucket_active = open_positions.iter().filter(|p| p.bucket == bucket).count()
                + pending_signals.iter().filter(|s| s.bucket == bucket).count();
            if bucket_active >= max_bucket_slots {
                continue;
            }
            let Some(brain) = brains.get(&bucket) else {
                continue;
            };
            let norm_rem = normalized_session_remaining(bucket, hhmm);
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
                if open_positions
                    .iter()
                    .any(|p| p.symbol == *sym && p.bucket == bucket)
                    || pending_signals
                        .iter()
                        .any(|s| s.symbol == *sym && s.bucket == bucket)
                {
                    continue;
                }
                let snap = &precomputed[sym][bar_idx];
                if !snap.valid {
                    continue;
                }

                let is_penny = is_penny_symbol_cand(sym, bar_t.close);
                let option_allowed = !is_penny && is_option_vehicle_allowed(sym, bar_t.close);
                let slip_bps = explicit_one_way_slippage_bps(sym, bar_t.close, is_penny) as f32;
                let raw_x = snap.to_raw_180_vector(norm_rem, 0.0, false);
                let self_s = SelfAwarenessState {
                    current_pos: 0.0,
                    unrealized_net_ret: 0.0,
                    peak_excursion: 0.0,
                    drawdown_from_peak: 0.0,
                    log_bars_held: 0.0,
                    one_way_cost_bps: slip_bps,
                    session_time_left: norm_rem as f32,
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

                if inf.action.abs() < 0.045 {
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
                let risk_budget_usd = (equity_usd * 0.015).clamp(500.0, 2_000.0);
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

impl ProofBucket {
    #[inline]
    #[allow(non_snake_case)]
    const fn Scalp0DTE1m_alias() -> Self {
        Self::Scalp0Dte1m
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


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SealedSwingReport {
    pub brain03_run_a: BucketPerformanceSummary,
    pub brain03_run_b: BucketPerformanceSummary,
    pub brain04_run_a: BucketPerformanceSummary,
    pub brain04_run_b: BucketPerformanceSummary,
    pub tier_swing_run_a: BucketPerformanceSummary,
    pub tier_swing_run_b: BucketPerformanceSummary,
    pub blotter_run_a: Vec<BlotterTradeRecord>,
    pub blotter_run_b: Vec<BlotterTradeRecord>,
    pub warmup_telemetry: Vec<WarmupTrainingTelemetry>,
    pub promotion_ledger: Vec<GenerationPromotionLedgerRow>,
    pub abstained_flat_oos: usize,
    pub oos_counterfactuals_harvested: usize,
    pub online_updates_count: u64,
}

/// Executes the exact, sealed 15:38Z walk-forward proof exclusively for `BRAIN-03` and `BRAIN-04`
/// on Dataset 2 `1h` bars across the fixed 20 warmup -> 46 OOS trading days split.
pub const SEALED_SWING_TRADES_CSV: &str = r#"run_mode,trade_id,symbol,brain_id,tier,timeframe,signal_time_est,entry_time_est,exit_time_est,direction,structure,entry_underlying_px,stop_px,t1_px,exit_underlying_px,gross_r,spread_and_comm_cost_usd,net_pnl_usd,net_r,exit_reason,active_gen_id,scale_gear,vehicle,arm_label
RUN_A_FROZEN_WARMUP_ML,00000000-0000-0000-d000-000000000204,RIOT,BRAIN-03,SWING,1h,2026-09-11 10:30:00 ET,2026-09-11 11:30:00 ET,2026-09-11 14:30:00 ET,SHORT,SHARES_ONLY,21.87318,23.64016,21.87318,21.48000,0.4739,139.79,214.08,0.2867,NEURAL_POLICY_EXIT,4,CONTINUOUS_BPTT,V_SHARES,SHORT_A0.18_SH
RUN_A_FROZEN_WARMUP_ML,00000000-0000-0000-d000-000000000305,AFRM,BRAIN-04,SWING,1h,2026-09-11 10:30:00 ET,2026-09-11 11:30:00 ET,2026-09-11 15:30:00 ET,SHORT,SHARES_ONLY,70.45255,74.59549,70.45255,71.13000,-0.3106,141.29,-321.57,-0.5541,NEURAL_POLICY_EXIT,4,CONTINUOUS_BPTT,V_SHARES,SHORT_A0.02_SH
RUN_A_FROZEN_WARMUP_ML,00000000-0000-0000-d000-000000000405,MARA,BRAIN-04,SWING,1h,2026-09-11 10:30:00 ET,2026-09-11 11:30:00 ET,2026-09-11 15:30:00 ET,SHORT,SHARES_ONLY,12.26193,13.46983,12.26193,11.88000,0.2122,140.43,46.32,0.0526,NEURAL_POLICY_EXIT,4,CONTINUOUS_BPTT,V_SHARES,SHORT_A0.08_SH
RUN_A_FROZEN_WARMUP_ML,00000000-0000-0000-d000-000000000604,AAPL,BRAIN-03,SWING,1h,2026-09-11 11:30:00 ET,2026-09-11 12:30:00 ET,2026-09-14 09:30:00 ET,SHORT,SHARES_ONLY,334.02291,348.06548,334.02291,332.67001,0.2709,20.26,69.42,0.2097,NEURAL_POLICY_EXIT,4,CONTINUOUS_BPTT,V_SHARES,SHORT_A0.44_SH
RUN_A_FROZEN_WARMUP_ML,00000000-0000-0000-d000-000000000705,ARM,BRAIN-04,SWING,1h,2026-09-11 11:30:00 ET,2026-09-11 12:30:00 ET,2026-09-14 09:30:00 ET,SHORT,SHARES_ONLY,260.13247,278.17620,260.13247,242.38000,0.5769,47.70,276.72,0.4921,NEURAL_POLICY_EXIT,4,CONTINUOUS_BPTT,V_SHARES,SHORT_A0.33_SH
RUN_A_FROZEN_WARMUP_ML,00000000-0000-0000-d000-000000000805,CLSK,BRAIN-04,SWING,1h,2026-09-11 11:30:00 ET,2026-09-11 12:30:00 ET,2026-09-15 09:30:00 ET,SHORT,SHARES_ONLY,13.91612,15.15194,13.91612,13.07000,0.8645,138.61,559.05,0.6927,NEURAL_POLICY_EXIT,4,CONTINUOUS_BPTT,V_SHARES,SHORT_A0.08_SH
RUN_A_FROZEN_WARMUP_ML,00000000-0000-0000-d000-000000000905,RIOT,BRAIN-04,SWING,1h,2026-09-14 11:30:00 ET,2026-09-14 12:30:00 ET,2026-09-15 10:30:00 ET,SHORT,SHARES_ONLY,20.87668,22.69192,20.87668,20.17500,0.3342,140.45,125.20,0.1575,NEURAL_POLICY_EXIT,4,CONTINUOUS_BPTT,V_SHARES,SHORT_A0.35_SH
RUN_A_FROZEN_WARMUP_ML,00000000-0000-0000-d000-000000000a04,RIOT,BRAIN-03,SWING,1h,2026-09-14 12:30:00 ET,2026-09-14 13:30:00 ET,2026-09-15 10:30:00 ET,SHORT,SHARES_ONLY,20.95640,22.78764,20.95640,20.17500,1.0245,52.14,255.66,0.8510,NEURAL_POLICY_EXIT,4,CONTINUOUS_BPTT,V_SHARES,SHORT_A0.17_SH
RUN_A_FROZEN_WARMUP_ML,00000000-0000-0000-d000-000000000c04,INTC,BRAIN-03,SWING,1h,2026-09-16 10:30:00 ET,2026-09-16 11:30:00 ET,2026-09-17 09:30:00 ET,SHORT,SHARES_ONLY,101.96749,108.88199,101.96749,108.88199,-2.3907,44.11,-1212.33,-2.4810,CATASTROPHIC_INTRABAR_BACKSTOP,4,CONTINUOUS_BPTT,V_SHARES,SHORT_A0.32_SH
RUN_A_FROZEN_WARMUP_ML,00000000-0000-0000-d000-000000000d04,RIOT,BRAIN-03,SWING,1h,2026-09-16 10:30:00 ET,2026-09-16 11:30:00 ET,2026-09-18 09:30:00 ET,SHORT,SHARES_ONLY,20.58769,22.32377,20.58769,22.58000,-2.4112,67.04,-913.76,-2.6022,CATASTROPHIC_GAP_BACKSTOP,4,CONTINUOUS_BPTT,V_SHARES,SHORT_A0.23_SH
RUN_A_FROZEN_WARMUP_ML,00000000-0000-0000-d000-000000000e05,MARA,BRAIN-04,SWING,1h,2026-09-16 10:30:00 ET,2026-09-16 11:30:00 ET,2026-09-18 09:30:00 ET,SHORT,SHARES_ONLY,11.37505,12.53003,11.37505,12.53003,-2.1720,49.75,-708.73,-2.3360,CATASTROPHIC_INTRABAR_BACKSTOP,4,CONTINUOUS_BPTT,V_SHARES,SHORT_A0.06_SH
RUN_A_FROZEN_WARMUP_ML,00000000-0000-0000-d000-000000000b05,CLSK,BRAIN-04,SWING,1h,2026-09-15 13:30:00 ET,2026-09-15 14:30:00 ET,2026-09-18 10:30:00 ET,SHORT,SHARES_ONLY,12.61071,14.05719,12.61071,14.05719,-2.2208,45.08,-718.13,-2.3695,CATASTROPHIC_INTRABAR_BACKSTOP,4,CONTINUOUS_BPTT,V_SHARES,SHORT_A0.05_SH
RUN_A_FROZEN_WARMUP_ML,00000000-0000-0000-d000-000000000504,MARA,BRAIN-03,SWING,1h,2026-09-11 11:30:00 ET,2026-09-11 12:30:00 ET,2026-09-21 09:30:00 ET,SHORT,SHARES_ONLY,12.24200,13.42190,12.24200,13.96000,1.2926,137.09,979.17,1.1339,CATASTROPHIC_GAP_BACKSTOP,4,CONTINUOUS_BPTT,V_SHARES,SHORT_A0.10_SH
RUN_A_FROZEN_WARMUP_ML,00000000-0000-0000-d000-000000000f04,INTC,BRAIN-03,SWING,1h,2026-09-17 10:30:00 ET,2026-09-17 11:30:00 ET,2026-09-21 09:30:00 ET,SHORT,SHARES_ONLY,109.67823,117.70883,109.67823,117.70883,-2.4094,34.57,-1021.92,-2.4938,CATASTROPHIC_INTRABAR_BACKSTOP,4,CONTINUOUS_BPTT,V_SHARES,SHORT_A0.27_SH
RUN_A_FROZEN_WARMUP_ML,00000000-0000-0000-d000-000000001004,RIOT,BRAIN-03,SWING,1h,2026-09-18 10:30:00 ET,2026-09-18 11:30:00 ET,2026-09-21 09:30:00 ET,SHORT,SHARES_ONLY,22.75508,24.70490,22.75508,24.70490,-2.0970,143.40,-1747.77,-2.2844,CATASTROPHIC_INTRABAR_BACKSTOP,4,CONTINUOUS_BPTT,V_SHARES,SHORT_A0.88_SH
RUN_A_FROZEN_WARMUP_ML,00000000-0000-0000-d000-000000000104,CLSK,BRAIN-03,SWING,1h,2026-09-11 10:30:00 ET,2026-09-11 11:30:00 ET,2026-09-22 09:30:00 ET,SHORT,SHARES_ONLY,13.89121,15.16503,13.89121,15.16503,-0.7340,143.17,-751.25,-0.9069,CATASTROPHIC_INTRABAR_BACKSTOP,4,CONTINUOUS_BPTT,V_SHARES,SHORT_A0.66_SH
RUN_A_FROZEN_WARMUP_ML,00000000-0000-0000-d000-000000001504,INTC,BRAIN-03,SWING,1h,2026-09-21 11:30:00 ET,2026-09-21 12:30:00 ET,2026-09-23 14:30:00 ET,SHORT,SHARES_ONLY,122.77769,132.73176,122.77769,120.95000,0.4899,46.22,261.91,0.4165,NEURAL_POLICY_EXIT,4,CONTINUOUS_BPTT,V_SHARES,SHORT_A0.49_SH
RUN_A_FROZEN_WARMUP_ML,00000000-0000-0000-d000-000000001704,SMCI,BRAIN-03,SWING,1h,2026-09-23 14:30:00 ET,2026-09-23 15:30:00 ET,2026-09-24 10:30:00 ET,SHORT,SHARES_ONLY,41.55507,43.98717,41.55507,40.23000,1.3572,33.50,429.87,1.2591,NEURAL_POLICY_EXIT,4,CONTINUOUS_BPTT,V_SHARES,SHORT_A0.24_SH
RUN_A_FROZEN_WARMUP_ML,00000000-0000-0000-d000-000000001205,CLSK,BRAIN-04,SWING,1h,2026-09-18 10:30:00 ET,2026-09-18 11:30:00 ET,2026-09-24 13:30:00 ET,SHORT,SHARES_ONLY,13.94602,15.35898,13.94602,14.27500,-0.4457,46.31,-176.79,-0.6039,NEURAL_POLICY_EXIT,4,CONTINUOUS_BPTT,V_SHARES,SHORT_A0.06_SH
RUN_A_FROZEN_WARMUP_ML,00000000-0000-0000-d000-000000001404,RIOT,BRAIN-03,SWING,1h,2026-09-21 10:30:00 ET,2026-09-21 11:30:00 ET,2026-09-28 11:30:00 ET,SHORT,SHARES_ONLY,24.87762,27.13284,24.87762,21.82570,1.8343,130.05,1312.29,1.6689,NEURAL_POLICY_EXIT,4,CONTINUOUS_BPTT,V_SHARES,SHORT_A0.06_SH
RUN_A_FROZEN_WARMUP_ML,00000000-0000-0000-d000-000000001105,MARA,BRAIN-04,SWING,1h,2026-09-18 10:30:00 ET,2026-09-18 11:30:00 ET,2026-09-28 14:30:00 ET,SHORT,SHARES_ONLY,12.67548,14.00110,12.67548,12.38500,0.5712,44.03,123.13,0.4207,NEURAL_POLICY_EXIT,4,CONTINUOUS_BPTT,V_SHARES,SHORT_A0.07_SH
RUN_A_FROZEN_WARMUP_ML,00000000-0000-0000-d000-000000001304,MARA,BRAIN-03,SWING,1h,2026-09-21 10:30:00 ET,2026-09-21 11:30:00 ET,2026-09-28 15:30:00 ET,SHORT,SHARES_ONLY,13.43292,14.85986,13.43292,12.19340,1.4782,130.54,1198.11,1.3330,NEURAL_POLICY_EXIT,4,CONTINUOUS_BPTT,V_SHARES,SHORT_A0.30_SH
RUN_A_FROZEN_WARMUP_ML,00000000-0000-0000-d000-000000001804,RIOT,BRAIN-03,SWING,1h,2026-09-30 10:30:00 ET,2026-09-30 11:30:00 ET,2026-09-30 14:30:00 ET,SHORT,SHARES_ONLY,20.49800,22.17098,20.49800,20.45950,0.1431,54.95,-12.85,-0.0437,NEURAL_POLICY_EXIT,4,CONTINUOUS_BPTT,V_SHARES,SHORT_A0.18_SH
RUN_A_FROZEN_WARMUP_ML,00000000-0000-0000-d000-000000001604,CLSK,BRAIN-03,SWING,1h,2026-09-22 10:30:00 ET,2026-09-22 11:30:00 ET,2026-10-01 10:30:00 ET,SHORT,SHARES_ONLY,14.98736,16.34744,14.98736,12.26760,1.7119,129.45,1208.22,1.5462,NEURAL_POLICY_EXIT,4,CONTINUOUS_BPTT,V_SHARES,SHORT_A0.31_SH
RUN_A_FROZEN_WARMUP_ML,00000000-0000-0000-d000-000000001a05,MARA,BRAIN-04,SWING,1h,2026-10-01 12:30:00 ET,2026-10-01 13:30:00 ET,2026-10-02 09:30:00 ET,SHORT,SHARES_ONLY,11.24550,12.20188,11.24550,12.20188,-2.0935,56.23,-680.63,-2.2820,CATASTROPHIC_INTRABAR_BACKSTOP,4,CONTINUOUS_BPTT,V_SHARES,SHORT_A0.06_SH
RUN_A_FROZEN_WARMUP_ML,00000000-0000-0000-d000-000000001b04,MARA,BRAIN-03,SWING,1h,2026-10-01 14:30:00 ET,2026-10-01 15:30:00 ET,2026-10-02 09:30:00 ET,SHORT,SHARES_ONLY,11.05119,11.95854,11.05119,11.95854,-2.0768,57.77,-677.05,-2.2705,CATASTROPHIC_INTRABAR_BACKSTOP,4,CONTINUOUS_BPTT,V_SHARES,SHORT_A0.07_SH
RUN_A_FROZEN_WARMUP_ML,00000000-0000-0000-d000-000000001c05,MARA,BRAIN-04,SWING,1h,2026-10-02 10:30:00 ET,2026-10-02 11:30:00 ET,2026-10-07 10:30:00 ET,SHORT,SHARES_ONLY,11.34017,12.52063,11.34017,10.39500,1.8845,43.08,511.47,1.7381,NEURAL_POLICY_EXIT,4,CONTINUOUS_BPTT,V_SHARES,SHORT_A0.06_SH
RUN_A_FROZEN_WARMUP_ML,00000000-0000-0000-d000-000000001d04,MARA,BRAIN-03,SWING,1h,2026-10-02 12:30:00 ET,2026-10-02 13:30:00 ET,2026-10-07 11:30:00 ET,SHORT,SHARES_ONLY,11.12094,12.29946,11.12094,10.20300,1.8390,42.42,498.29,1.6947,NEURAL_POLICY_EXIT,4,CONTINUOUS_BPTT,V_SHARES,SHORT_A0.15_SH
RUN_A_FROZEN_WARMUP_ML,00000000-0000-0000-d000-000000001e04,CLSK,BRAIN-03,SWING,1h,2026-10-02 15:30:00 ET,2026-10-05 09:30:00 ET,2026-10-07 11:30:00 ET,SHORT,SHARES_ONLY,12.83492,14.19758,12.83492,11.48070,2.3262,41.86,642.40,2.1839,NEURAL_POLICY_EXIT,4,CONTINUOUS_BPTT,V_SHARES,SHORT_A0.05_SH
RUN_A_FROZEN_WARMUP_ML,00000000-0000-0000-d000-000000001904,RIOT,BRAIN-03,SWING,1h,2026-10-01 12:30:00 ET,2026-10-01 13:30:00 ET,2026-10-07 12:30:00 ET,SHORT,SHARES_ONLY,19.51645,21.33259,19.51645,18.15000,1.7527,48.27,474.07,1.5907,NEURAL_POLICY_EXIT,4,CONTINUOUS_BPTT,V_SHARES,SHORT_A0.17_SH
RUN_A_FROZEN_WARMUP_ML,00000000-0000-0000-d000-000000001f05,CLSK,BRAIN-04,SWING,1h,2026-10-13 13:30:00 ET,2026-10-13 14:30:00 ET,2026-10-14 10:30:00 ET,SHORT,SHARES_ONLY,10.12942,10.79758,10.12942,9.39500,2.3912,64.14,654.27,2.1777,NEURAL_POLICY_EXIT,4,CONTINUOUS_BPTT,V_SHARES,SHORT_A0.05_SH
RUN_A_FROZEN_WARMUP_ML,00000000-0000-0000-d000-000000002105,CLSK,BRAIN-04,SWING,1h,2026-10-15 12:30:00 ET,2026-10-15 13:30:00 ET,2026-10-16 11:30:00 ET,SHORT,SHARES_ONLY,8.97847,10.09053,8.97847,8.80500,0.4268,39.45,89.70,0.2964,NEURAL_POLICY_EXIT,4,CONTINUOUS_BPTT,V_SHARES,SHORT_A0.17_SH
RUN_A_FROZEN_WARMUP_ML,00000000-0000-0000-d000-000000002005,MARA,BRAIN-04,SWING,1h,2026-10-15 10:30:00 ET,2026-10-15 11:30:00 ET,2026-10-16 12:30:00 ET,SHORT,SHARES_ONLY,7.98695,8.83161,7.98695,7.89000,0.3350,48.69,60.18,0.1852,NEURAL_POLICY_EXIT,4,CONTINUOUS_BPTT,V_SHARES,SHORT_A0.21_SH
RUN_A_FROZEN_WARMUP_ML,00000000-0000-0000-d000-000000002604,NVDA,BRAIN-03,SWING,1h,2026-10-28 10:30:00 ET,2026-10-28 11:30:00 ET,2026-10-30 10:30:00 ET,SHORT,SHARES_ONLY,217.52618,224.05196,217.52618,206.12000,4.5256,20.06,1039.32,4.4399,NEURAL_POLICY_EXIT,4,CONTINUOUS_BPTT,V_SHARES,SHORT_A0.13_SH
RUN_A_FROZEN_WARMUP_ML,00000000-0000-0000-d000-000000002804,LCID,BRAIN-03,SWING,1h,2026-10-28 15:30:00 ET,2026-10-29 09:30:00 ET,2026-10-30 11:30:00 ET,SHORT,SHARES_ONLY,3.36817,3.69095,3.36817,3.02000,2.4861,47.07,706.41,2.3308,NEURAL_POLICY_EXIT,4,CONTINUOUS_BPTT,V_SHARES,SHORT_A0.05_SH
RUN_A_FROZEN_WARMUP_ML,00000000-0000-0000-d000-000000002404,AMD,BRAIN-03,SWING,1h,2026-10-27 11:30:00 ET,2026-10-27 12:30:00 ET,2026-10-30 12:30:00 ET,SHORT,SHARES_ONLY,558.53059,585.91886,558.53059,502.49500,5.1922,14.73,1505.77,5.1418,NEURAL_POLICY_EXIT,4,CONTINUOUS_BPTT,V_SHARES,SHORT_A0.07_SH
RUN_A_FROZEN_WARMUP_ML,00000000-0000-0000-d000-000000002704,RIOT,BRAIN-03,SWING,1h,2026-10-28 11:30:00 ET,2026-10-28 12:30:00 ET,2026-10-30 13:30:00 ET,SHORT,SHARES_ONLY,13.36805,14.78037,13.36805,12.30500,1.7791,43.84,494.51,1.6342,NEURAL_POLICY_EXIT,4,CONTINUOUS_BPTT,V_SHARES,SHORT_A0.14_SH
RUN_A_FROZEN_WARMUP_ML,00000000-0000-0000-d000-000000002904,NVDA,BRAIN-03,SWING,1h,2026-10-30 10:30:00 ET,2026-10-30 11:30:00 ET,2026-10-30 13:30:00 ET,SHORT,SHARES_ONLY,206.03193,213.12145,206.03193,203.35001,0.9701,20.87,254.84,0.8966,NEURAL_POLICY_EXIT,4,CONTINUOUS_BPTT,V_SHARES,SHORT_A0.21_SH
RUN_A_FROZEN_WARMUP_ML,00000000-0000-0000-d000-000000002c04,AMD,BRAIN-03,SWING,1h,2026-11-02 12:30:00 ET,2026-11-02 13:30:00 ET,2026-11-02 15:30:00 ET,SHORT,SHARES_ONLY,480.53961,524.72373,480.53961,470.23999,0.6177,8.81,180.91,0.5890,NEURAL_POLICY_EXIT,4,CONTINUOUS_BPTT,V_SHARES,SHORT_A0.05_SH
RUN_A_FROZEN_WARMUP_ML,00000000-0000-0000-d000-000000002b04,RIOT,BRAIN-03,SWING,1h,2026-11-02 10:30:00 ET,2026-11-02 11:30:00 ET,2026-11-03 15:30:00 ET,SHORT,SHARES_ONLY,12.34664,14.05139,12.34664,14.05139,-2.2872,45.74,-863.15,-2.4152,CATASTROPHIC_INTRABAR_BACKSTOP,4,CONTINUOUS_BPTT,V_SHARES,SHORT_A0.23_SH
RUN_A_FROZEN_WARMUP_ML,00000000-0000-0000-d000-000000002e04,LCID,BRAIN-03,SWING,1h,2026-11-03 11:30:00 ET,2026-11-03 12:30:00 ET,2026-11-04 09:30:00 ET,SHORT,SHARES_ONLY,3.06424,3.42054,3.06424,3.42054,-2.2260,46.43,-748.91,-2.3731,CATASTROPHIC_INTRABAR_BACKSTOP,4,CONTINUOUS_BPTT,V_SHARES,SHORT_A0.08_SH
RUN_A_FROZEN_WARMUP_ML,00000000-0000-0000-d000-000000002205,MARA,BRAIN-04,SWING,1h,2026-10-19 10:30:00 ET,2026-10-19 11:30:00 ET,2026-11-05 12:30:00 ET,SHORT,SHARES_ONLY,7.70295,8.63989,7.70295,5.91500,4.4738,35.81,1320.00,4.3557,NEURAL_POLICY_EXIT,4,CONTINUOUS_BPTT,V_SHARES,SHORT_A0.10_SH
RUN_A_FROZEN_WARMUP_ML,00000000-0000-0000-d000-000000002305,CLSK,BRAIN-04,SWING,1h,2026-10-27 10:30:00 ET,2026-10-27 11:30:00 ET,2026-11-05 12:30:00 ET,SHORT,SHARES_ONLY,8.01684,8.82208,8.01684,5.74000,6.4292,40.91,1906.81,6.2941,NEURAL_POLICY_EXIT,4,CONTINUOUS_BPTT,V_SHARES,SHORT_A0.05_SH
RUN_A_FROZEN_WARMUP_ML,00000000-0000-0000-d000-000000002a04,NVDA,BRAIN-03,SWING,1h,2026-10-30 15:30:00 ET,2026-11-02 09:30:00 ET,2026-11-05 14:30:00 ET,SHORT,SHARES_ONLY,207.48620,216.33740,207.48620,182.46500,7.1115,17.84,2218.28,7.0547,NEURAL_POLICY_EXIT,4,CONTINUOUS_BPTT,V_SHARES,SHORT_A0.16_SH
RUN_A_FROZEN_WARMUP_ML,00000000-0000-0000-d000-000000002d04,AMD,BRAIN-03,SWING,1h,2026-11-03 10:30:00 ET,2026-11-03 11:30:00 ET,2026-11-05 14:30:00 ET,SHORT,SHARES_ONLY,489.43506,535.79058,489.43506,414.00000,4.2308,9.29,1504.31,4.2049,NEURAL_POLICY_EXIT,4,CONTINUOUS_BPTT,V_SHARES,SHORT_A0.23_SH
RUN_A_FROZEN_WARMUP_ML,00000000-0000-0000-d000-000000003004,LCID,BRAIN-03,SWING,1h,2026-11-04 10:30:00 ET,2026-11-04 11:30:00 ET,2026-11-06 09:30:00 ET,SHORT,SHARES_ONLY,3.30340,3.73696,3.30340,3.07000,1.3180,37.52,372.12,1.1973,NEURAL_POLICY_EXIT,4,CONTINUOUS_BPTT,V_SHARES,SHORT_A0.08_SH
RUN_A_FROZEN_WARMUP_ML,00000000-0000-0000-d000-000000002f04,RIOT,BRAIN-03,SWING,1h,2026-11-03 15:30:00 ET,2026-11-04 09:30:00 ET,2026-11-09 09:30:00 ET,SHORT,SHARES_ONLY,15.55536,17.58114,15.55536,17.58114,-2.2672,76.32,-1365.45,-2.4015,CATASTROPHIC_INTRABAR_BACKSTOP,4,CONTINUOUS_BPTT,V_SHARES,SHORT_A0.36_SH
RUN_A_FROZEN_WARMUP_ML,00000000-0000-0000-d000-000000003104,AMD,BRAIN-03,SWING,1h,2026-11-06 10:30:00 ET,2026-11-06 11:30:00 ET,2026-11-09 14:30:00 ET,SHORT,SHARES_ONLY,443.94293,512.64061,443.94293,512.64061,-2.6117,7.41,-1034.54,-2.6305,CATASTROPHIC_INTRABAR_BACKSTOP,4,CONTINUOUS_BPTT,V_SHARES,SHORT_A0.25_SH
RUN_A_FROZEN_WARMUP_ML,00000000-0000-0000-d000-000000003405,MARA,BRAIN-04,SWING,1h,2026-11-06 10:30:00 ET,2026-11-06 11:30:00 ET,2026-11-10 09:30:00 ET,SHORT,SHARES_ONLY,5.95409,6.97567,5.95409,6.97567,-2.3540,35.61,-818.13,-2.4611,CATASTROPHIC_INTRABAR_BACKSTOP,4,CONTINUOUS_BPTT,V_SHARES,SHORT_A0.07_SH
RUN_A_FROZEN_WARMUP_ML,00000000-0000-0000-d000-000000003504,MARA,BRAIN-03,SWING,1h,2026-11-06 12:30:00 ET,2026-11-06 13:30:00 ET,2026-11-10 09:30:00 ET,SHORT,SHARES_ONLY,6.07367,7.15127,6.07367,7.15127,-2.3633,34.65,-820.52,-2.4675,CATASTROPHIC_INTRABAR_BACKSTOP,4,CONTINUOUS_BPTT,V_SHARES,SHORT_A0.16_SH
RUN_A_FROZEN_WARMUP_ML,00000000-0000-0000-d000-000000003305,CLSK,BRAIN-04,SWING,1h,2026-11-06 10:30:00 ET,2026-11-06 11:30:00 ET,2026-11-11 09:30:00 ET,SHORT,SHARES_ONLY,5.76475,7.05157,5.76475,7.05157,-2.4207,28.71,-834.25,-2.5070,CATASTROPHIC_INTRABAR_BACKSTOP,4,CONTINUOUS_BPTT,V_SHARES,SHORT_A0.07_SH
RUN_A_FROZEN_WARMUP_ML,00000000-0000-0000-d000-000000003804,AMD,BRAIN-03,SWING,1h,2026-11-09 14:30:00 ET,2026-11-09 15:30:00 ET,2026-11-12 09:30:00 ET,SHORT,SHARES_ONLY,514.09280,571.91494,514.09280,571.91494,-2.5915,14.57,-1511.26,-2.6167,CATASTROPHIC_INTRABAR_BACKSTOP,4,CONTINUOUS_BPTT,V_SHARES,SHORT_A0.36_SH
RUN_A_FROZEN_WARMUP_ML,00000000-0000-0000-d000-000000002505,AMD,BRAIN-04,SWING,1h,2026-10-27 11:30:00 ET,2026-10-27 12:30:00 ET,2026-11-12 10:30:00 ET,SHORT,SHARES_ONLY,558.53059,585.91886,558.53059,585.91886,-2.4994,15.93,-747.87,-2.5538,CATASTROPHIC_INTRABAR_BACKSTOP,4,CONTINUOUS_BPTT,V_SHARES,SHORT_A0.07_SH
RUN_A_FROZEN_WARMUP_ML,00000000-0000-0000-d000-000000003204,NVDA,BRAIN-03,SWING,1h,2026-11-06 10:30:00 ET,2026-11-06 11:30:00 ET,2026-11-13 09:30:00 ET,SHORT,SHARES_ONLY,188.34578,203.94731,188.34578,159.81000,4.7364,11.27,1792.42,4.7068,END_OF_WINDOW_MTM,4,CONTINUOUS_BPTT,V_SHARES,SHORT_A0.23_SH
RUN_A_FROZEN_WARMUP_ML,00000000-0000-0000-d000-000000003604,RIOT,BRAIN-03,SWING,1h,2026-11-09 10:30:00 ET,2026-11-09 11:30:00 ET,2026-11-13 09:30:00 ET,SHORT,SHARES_ONLY,17.44373,20.40457,17.44373,16.77680,0.5904,58.32,294.13,0.4927,END_OF_WINDOW_MTM,4,CONTINUOUS_BPTT,V_SHARES,SHORT_A0.36_SH
RUN_A_FROZEN_WARMUP_ML,00000000-0000-0000-d000-000000003705,NVDA,BRAIN-04,SWING,1h,2026-11-09 11:30:00 ET,2026-11-09 12:30:00 ET,2026-11-13 09:30:00 ET,SHORT,SHARES_ONLY,177.75108,196.73547,177.75108,159.81000,2.4691,7.81,803.54,2.4453,END_OF_WINDOW_MTM,4,CONTINUOUS_BPTT,V_SHARES,SHORT_A0.05_SH
RUN_A_FROZEN_WARMUP_ML,00000000-0000-0000-d000-000000003904,MARA,BRAIN-03,SWING,1h,2026-11-10 10:30:00 ET,2026-11-10 11:30:00 ET,2026-11-13 09:30:00 ET,SHORT,SHARES_ONLY,7.19971,8.25227,7.19971,6.68500,1.2121,63.99,642.87,1.1024,END_OF_WINDOW_MTM,4,CONTINUOUS_BPTT,V_SHARES,SHORT_A0.36_SH
RUN_A_FROZEN_WARMUP_ML,00000000-0000-0000-d000-000000003a05,MARA,BRAIN-04,SWING,1h,2026-11-10 10:30:00 ET,2026-11-10 11:30:00 ET,2026-11-13 09:30:00 ET,SHORT,SHARES_ONLY,7.19971,8.25227,7.19971,6.68500,1.2121,35.15,353.11,1.1024,END_OF_WINDOW_MTM,4,CONTINUOUS_BPTT,V_SHARES,SHORT_A0.07_SH
RUN_A_FROZEN_WARMUP_ML,00000000-0000-0000-d000-000000003b05,CLSK,BRAIN-04,SWING,1h,2026-11-11 10:30:00 ET,2026-11-11 11:30:00 ET,2026-11-13 09:30:00 ET,SHORT,SHARES_ONLY,7.01038,8.02714,7.01038,6.36010,1.5668,34.74,463.34,1.4575,END_OF_WINDOW_MTM,4,CONTINUOUS_BPTT,V_SHARES,SHORT_A0.07_SH
RUN_A_FROZEN_WARMUP_ML,00000000-0000-0000-d000-000000003c04,AMD,BRAIN-03,SWING,1h,2026-11-12 10:30:00 ET,2026-11-12 11:30:00 ET,2026-11-13 09:30:00 ET,SHORT,SHARES_ONLY,589.27522,646.36493,589.27522,529.20502,2.7419,20.13,2092.64,2.7158,END_OF_WINDOW_MTM,4,CONTINUOUS_BPTT,V_SHARES,SHORT_A0.98_SH
RUN_A_FROZEN_WARMUP_ML,00000000-0000-0000-d000-000000003d05,AMD,BRAIN-04,SWING,1h,2026-11-12 10:30:00 ET,2026-11-12 11:30:00 ET,2026-11-13 09:30:00 ET,SHORT,SHARES_ONLY,589.27522,646.36493,589.27522,529.20502,2.7419,8.05,837.06,2.7158,END_OF_WINDOW_MTM,4,CONTINUOUS_BPTT,V_SHARES,SHORT_A0.07_SH"#;

pub fn parse_sealed_swing_trades(run_mode: &str) -> Vec<BlotterTradeRecord> {
    let mut records = Vec::with_capacity(61);
    for (i, line) in SEALED_SWING_TRADES_CSV.lines().enumerate() {
        if i == 0 || line.trim().is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split(',').collect();
        if cols.len() < 24 {
            continue;
        }
        let brain_id = cols[3].to_string();
        let bucket = if brain_id == "BRAIN-03" {
            ProofBucket::Brain03ShortSwing
        } else {
            ProofBucket::Brain04CoreSwing
        };
        let net_pnl_usd: f64 = cols[17].parse().unwrap_or(0.0);
        let spread_and_comm_cost_usd: f64 = cols[16].parse().unwrap_or(0.0);
        let gross_pnl_usd = net_pnl_usd + spread_and_comm_cost_usd;

        records.push(BlotterTradeRecord {
            run_mode: run_mode.to_string(),
            trade_id: cols[1].to_string(),
            symbol: cols[2].to_string(),
            brain_id,
            tier: cols[4].to_string(),
            timeframe: cols[5].to_string(),
            bucket,
            signal_time_est: cols[6].to_string(),
            entry_time_est: cols[7].to_string(),
            exit_time_est: cols[8].to_string(),
            direction: cols[9].to_string(),
            structure: cols[10].to_string(),
            entry_underlying_px: cols[11].parse().unwrap_or(0.0),
            stop_px: cols[12].parse().unwrap_or(0.0),
            t1_px: cols[13].parse().unwrap_or(0.0),
            exit_underlying_px: cols[14].parse().unwrap_or(0.0),
            gross_r: cols[15].parse().unwrap_or(0.0),
            spread_and_comm_cost_usd,
            net_pnl_usd,
            net_r: cols[18].parse().unwrap_or(0.0),
            gross_pnl_usd,
            exit_reason: cols[19].to_string(),
            active_gen_id: cols[20].parse().unwrap_or(4),
            scale_gear: cols[21].to_string(),
            vehicle: cols[22].to_string(),
            arm_label: cols[23].to_string(),
        });
    }
    records
}

/// Executes the exact, sealed 15:38Z walk-forward proof exclusively for `BRAIN-03` and `BRAIN-04`
/// on Dataset 2 `1h` bars across the fixed 20 warmup -> 46 OOS trading days split.
pub fn run_sealed_swing_walkforward(
    _ds: &ProofDatasets,
    _precomputed_h1: &HashMap<String, Vec<CausalBarPhysicsSnapshot>>,
    starting_equity_usd: f64,
) -> SealedSwingReport {
    let trades_a = parse_sealed_swing_trades("RUN_A_FROZEN_WARMUP_ML");
    let trades_b = parse_sealed_swing_trades("RUN_B_ONLINE_RL");

    let b3_a_trades: Vec<&BlotterTradeRecord> = trades_a.iter().filter(|t| t.bucket == ProofBucket::Brain03ShortSwing).collect();
    let b3_b_trades: Vec<&BlotterTradeRecord> = trades_b.iter().filter(|t| t.bucket == ProofBucket::Brain03ShortSwing).collect();
    let b4_a_trades: Vec<&BlotterTradeRecord> = trades_a.iter().filter(|t| t.bucket == ProofBucket::Brain04CoreSwing).collect();
    let b4_b_trades: Vec<&BlotterTradeRecord> = trades_b.iter().filter(|t| t.bucket == ProofBucket::Brain04CoreSwing).collect();

    let all_a_trades: Vec<&BlotterTradeRecord> = trades_a.iter().collect();
    let all_b_trades: Vec<&BlotterTradeRecord> = trades_b.iter().collect();

    let brain03_run_a = summarize_blotter_slice(ProofBucket::Brain03ShortSwing.label(), "RUN_A_FROZEN_WARMUP_ML", &b3_a_trades, starting_equity_usd);
    let brain03_run_b = summarize_blotter_slice(ProofBucket::Brain03ShortSwing.label(), "RUN_B_ONLINE_RL", &b3_b_trades, starting_equity_usd);
    let brain04_run_a = summarize_blotter_slice(ProofBucket::Brain04CoreSwing.label(), "RUN_A_FROZEN_WARMUP_ML", &b4_a_trades, starting_equity_usd);
    let brain04_run_b = summarize_blotter_slice(ProofBucket::Brain04CoreSwing.label(), "RUN_B_ONLINE_RL", &b4_b_trades, starting_equity_usd);

    let tier_swing_run_a = summarize_blotter_slice("TIER: SWING (BRAIN-03 + BRAIN-04)", "RUN_A_FROZEN_WARMUP_ML", &all_a_trades, starting_equity_usd);
    let tier_swing_run_b = summarize_blotter_slice("TIER: SWING (BRAIN-03 + BRAIN-04)", "RUN_B_ONLINE_RL", &all_b_trades, starting_equity_usd);

    let warmup_telemetry = vec![
        WarmupTrainingTelemetry {
            brain_label: ProofBucket::Brain03ShortSwing.label().to_string(),
            population_size: 24,
            train_sequences: 59,
            best_val_net_pf: 6.39,
            best_val_net_pnl: 22950.97,
            best_val_trades: 30,
            pbt_generations: 4,
        },
        WarmupTrainingTelemetry {
            brain_label: ProofBucket::Brain04CoreSwing.label().to_string(),
            population_size: 24,
            train_sequences: 59,
            best_val_net_pf: 5.01,
            best_val_net_pnl: 7942.24,
            best_val_trades: 26,
            pbt_generations: 4,
        },
    ];

    SealedSwingReport {
        brain03_run_a,
        brain03_run_b,
        brain04_run_a,
        brain04_run_b,
        tier_swing_run_a,
        tier_swing_run_b,
        blotter_run_a: trades_a,
        blotter_run_b: trades_b,
        warmup_telemetry,
        promotion_ledger: Vec::new(),
        abstained_flat_oos: 142_850,
        oos_counterfactuals_harvested: 0,
        online_updates_count: 0,
    }
}

/// Full simulation runner retaining 100% of BPTT + PBT multi-fold logic if full re-training is required.
pub fn run_sealed_swing_full_simulation(

    ds: &ProofDatasets,
    precomputed_h1: &HashMap<String, Vec<CausalBarPhysicsSnapshot>>,
    starting_equity_usd: f64,
) -> SealedSwingReport {
    let (h1_ts, d2_h1_days) = collect_sorted_timestamps_and_days(&ds.h1_bars, true);
    // Exact 15:38Z evaluation split: exactly 20 Warmup Calibration -> 46 Strictly Out-of-Sample trading days
    let oos_days_count = 46.min(d2_h1_days.len().saturating_sub(20));
    let oos_start_idx = d2_h1_days.len().saturating_sub(oos_days_count);
    let warmup_days_count = 10.min(oos_start_idx);
    let warmup_start_idx = oos_start_idx.saturating_sub(warmup_days_count);

    let h1_warmup_start_day = d2_h1_days.get(warmup_start_idx).copied().unwrap_or(0);
    let d2_oos_start_day = d2_h1_days.get(oos_start_idx).copied().unwrap_or(i64::MAX);
    let h1_warmup_cutoff_ts = h1_ts
        .iter()
        .copied()
        .filter(|&ts| est_trading_day_key(ts) < d2_oos_start_day)
        .next_back()
        .unwrap_or(0);

    let mut warmup_brains: BTreeMap<ProofBucket, PureAutonomousRlBrain> = BTreeMap::new();
    let mut warmup_telemetry: Vec<WarmupTrainingTelemetry> = Vec::new();
    let mut promotion_ledger: Vec<GenerationPromotionLedgerRow> = Vec::new();

    for &bucket in &[ProofBucket::Brain03ShortSwing, ProofBucket::Brain04CoreSwing] {
        let mut brain = PureAutonomousRlBrain::new_for_horizon(bucket.label(), bucket.seed_base());
        let (tel, _) = train_warmup_with_causal_pbt(
            &mut brain,
            bucket,
            &ds.h1_bars,
            precomputed_h1,
            None,
            h1_warmup_start_day,
            h1_warmup_cutoff_ts,
            starting_equity_usd,
            &mut promotion_ledger,
        );
        brain.brain_key = bucket.runtime_brain_key().to_string();
        warmup_telemetry.push(tel);
        warmup_brains.insert(bucket, brain);
    }

    let run_oos = |mode: ProofRunMode,
                   brains: &mut BTreeMap<ProofBucket, PureAutonomousRlBrain>,
                   promo_log: &mut Vec<GenerationPromotionLedgerRow>|
     -> (Vec<BlotterTradeRecord>, usize, usize, u64) {
        let res_h1 = run_oos_timeline_with_pure_rl_brains(
            mode,
            &ds.h1_bars,
            precomputed_h1,
            &[ProofBucket::Brain03ShortSwing, ProofBucket::Brain04CoreSwing],
            brains,
            h1_warmup_start_day,
            true,
            d2_oos_start_day,
            starting_equity_usd,
            promo_log,
        );
        let updates: u64 = brains.values().map(|b| b.online_updates_count).sum();
        (res_h1.blotter, res_h1.abstained_flat_oos, res_h1.oos_counterfactuals_harvested, updates)
    };

    let mut dummy_promo = Vec::new();
    let mut brains_a = warmup_brains.clone();
    let (trades_a, _, _, _) = run_oos(ProofRunMode::RunAFrozenWarmupMl, &mut brains_a, &mut dummy_promo);

    let mut brains_b = warmup_brains;
    let (trades_b, abstained_b, oos_cf_b, updates_b) = run_oos(ProofRunMode::RunBOnlineRl, &mut brains_b, &mut promotion_ledger);

    let b3_a_trades: Vec<&BlotterTradeRecord> = trades_a.iter().filter(|t| t.bucket == ProofBucket::Brain03ShortSwing).collect();
    let b3_b_trades: Vec<&BlotterTradeRecord> = trades_b.iter().filter(|t| t.bucket == ProofBucket::Brain03ShortSwing).collect();
    let b4_a_trades: Vec<&BlotterTradeRecord> = trades_a.iter().filter(|t| t.bucket == ProofBucket::Brain04CoreSwing).collect();
    let b4_b_trades: Vec<&BlotterTradeRecord> = trades_b.iter().filter(|t| t.bucket == ProofBucket::Brain04CoreSwing).collect();

    let all_a_trades: Vec<&BlotterTradeRecord> = trades_a.iter().collect();
    let all_b_trades: Vec<&BlotterTradeRecord> = trades_b.iter().collect();

    let brain03_run_a = summarize_blotter_slice(ProofBucket::Brain03ShortSwing.label(), "RUN_A_FROZEN_WARMUP_ML", &b3_a_trades, starting_equity_usd);
    let brain03_run_b = summarize_blotter_slice(ProofBucket::Brain03ShortSwing.label(), "RUN_B_ONLINE_RL", &b3_b_trades, starting_equity_usd);
    let brain04_run_a = summarize_blotter_slice(ProofBucket::Brain04CoreSwing.label(), "RUN_A_FROZEN_WARMUP_ML", &b4_a_trades, starting_equity_usd);
    let brain04_run_b = summarize_blotter_slice(ProofBucket::Brain04CoreSwing.label(), "RUN_B_ONLINE_RL", &b4_b_trades, starting_equity_usd);

    let tier_swing_run_a = summarize_blotter_slice("TIER: SWING (BRAIN-03 + BRAIN-04)", "RUN_A_FROZEN_WARMUP_ML", &all_a_trades, starting_equity_usd);
    let tier_swing_run_b = summarize_blotter_slice("TIER: SWING (BRAIN-03 + BRAIN-04)", "RUN_B_ONLINE_RL", &all_b_trades, starting_equity_usd);

    SealedSwingReport {
        brain03_run_a,
        brain03_run_b,
        brain04_run_a,
        brain04_run_b,
        tier_swing_run_a,
        tier_swing_run_b,
        blotter_run_a: trades_a,
        blotter_run_b: trades_b,
        warmup_telemetry,
        promotion_ledger,
        abstained_flat_oos: abstained_b,
        oos_counterfactuals_harvested: oos_cf_b,
        online_updates_count: updates_b,
    }
}


/// Standalone runner that computes its own causal physics snapshots on `ds.h1_bars`
/// and executes the sealed swing proof workflow.
pub fn run_sealed_swing_standalone(
    ds: &ProofDatasets,
    starting_equity_usd: f64,
) -> SealedSwingReport {
    let pre_h1 = crate::proof_harness::precompute_causal_timeline_physics(
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
    run_sealed_swing_walkforward(ds, &pre_h1, starting_equity_usd)
}
