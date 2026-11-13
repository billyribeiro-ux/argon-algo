#![allow(clippy::needless_range_loop)]

//! Two-Stage Jointly Differentiable Market-Physics World Model + End-to-End Differentiable
//! Stateful Sequence Policy + Closed-Loop 4-Vector GradNorm Loss-Autopsy RL (`PureAutonomousRlBrain`)
//!
//! Architectural Laws:
//! - ZERO hardcoded trading strategies, human indicators (`EMA`, `SMA`, `RSI`, `MACD`, `ORB`, `VCP`),
//!   static trade-count caps, symbol blacklists, or hardcoded stop/horizon templates.
//! - Stage A (`PhysicsWorldModel`) + Joint Auxiliary Recurrent World-Model Head (`w_wm_aux`, `b_wm_aux`):
//!   Residual LayerNorm + Multi-Timescale Gated Recurrent Core (`D = 64`) trained causally
//!   (`t + 1 + Delta <= t_current`) via Pinball Quantile Loss with full analytical Jacobian flow
//!   (`dL/dx = dL/da * da/dh * dh/dx + dL/dwm * dwm/dh * dh/dx`).
//! - Stage B (`SequencePolicyNet`): Concatenates `[x_t, z_t, wm_pred_t, s_{t-1}]` through a `D = 64`
//!   Multi-Timescale Gated Recurrent Core and feeds `[h_ln, wm_fused_t, s_{t-1}, pos_interact]` (`91-d`)
//!   into a `K = 5` Orthogonal Specialist Policy Ensemble with:
//!   1. Unconstrained Log-Space Cost Sensitivity `exp(log_cost_sens)` + Distributional Epistemic
//!      Lower-Confidence-Bound (`LCB_net = E[r_net] - lambda_epistemic * Spread(Q_75 - Q_25, Ensemble_Std)`),
//!   2. Localized Boundary Surrogate (`sech^2((|z| - tau) / temp)`) with active `FLAT` noise replay,
//!   3. Differentiable `(Symbol, TradingDay)` Session Episode EOD Closure (`a_{T_eod} = 0.0`
//!      with exact EOD flatten slippage/spread friction).
//! - Stage C: Disjoint Chronological `[T_train | T_holdout]` Loss-Autopsy RL with 4-Vector GradNorm
//!   (`L_friction_inv`, `L_hold_decay`, `L_vehicle_regret`, `L_flat_counterfactual`), Conditioned
//!   Active-Position Peak-Drawdown Attribution, Diagonal Fisher EWC, and Strict Holdout Pareto Promotion.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

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

pub const SELF_STATE_DIM: usize = 8;
pub const WM_DIM: usize = 32;
pub const WM_HORIZONS: [usize; 4] = [1, 3, 8, 24];
/// 4 horizons (`1, 3, 8, 24` bars) x 4 targets (`q25_ret, q50_ret, q75_ret, ln(Range / RT_Cost)`).
pub const WM_OUT_DIM: usize = 16;
pub const TOTAL_INPUT_DIM: usize = RL_STATE_DIM + WM_DIM + WM_OUT_DIM + SELF_STATE_DIM; // 268
pub const HIDDEN_DIM: usize = 64;
pub const POS_INTERACT_DIM: usize = 4; // sign-equivariant [pos*unreal_mtm, pos*dd_peak, pos*(1-sess), pos*log_bars]
pub const HEAD_IN_DIM: usize = HIDDEN_DIM + WM_OUT_DIM + SELF_STATE_DIM + POS_INTERACT_DIM; // 92
pub const SPECIALIST_K: usize = 5;
pub const CRITIC_QUANTILES: usize = 5;
pub const CRITIC_CHANNELS: usize = 4; // [LongShares, ShortShares, LongOption, ShortOption]
pub const CRITIC_OUT_DIM: usize = CRITIC_CHANNELS * CRITIC_QUANTILES; // 20
pub const POPULATION_SIZE: usize = 24;
pub const MAX_RED_REPLAY_CAP: usize = 128;
pub const MAX_WINNER_FISHER_CAP: usize = 96;

/// Unit scale mapping O(1) neural output logits to decimal returns (`1.0` logit = `0.010` = `100 bps`).
const RET_LOGIT_UNIT_SCALE: f32 = 0.010;
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

/// Localized `sech^2(x)` boundary surrogate that peaks at `1.0` on the threshold (`x = 0`)
/// and decays exponentially (`~ 4 * exp(-2|x|)`) in deep-noise regions (zero constant leakage floor).
#[inline]
fn sech2(x: f32) -> f32 {
    let c = x.clamp(-10.0, 10.0).cosh();
    1.0 / (c * c)
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

/// Shared causal feature normalization that preserves dimensionless physical cost/session scales
/// at `160..164` (`cost_to_atr`, `session_time_left`, `vix9d_to_vix_log`, `open_book_heat`) instead
/// of Z-scoring high-friction regimes to `0.0`.
#[inline]
pub fn normalize_212_with_stats(
    raw_x: &[f32; RL_STATE_DIM],
    feat_mean: &[f32],
    feat_var: &[f32],
) -> [f32; RL_STATE_DIM] {
    let mut out = [0.0_f32; RL_STATE_DIM];
    let cost_off = RL_EMBEDDING_DIM + RL_INSTANT_DIM;
    for i in 0..RL_STATE_DIM {
        let val = if !raw_x[i].is_finite() { 0.0 } else { raw_x[i] };
        if i == cost_off {
            out[i] = (val - 0.35).clamp(-2.0, 4.5);
        } else if i == cost_off + 1 {
            out[i] = (val * 2.0 - 1.0).clamp(-1.0, 1.0);
        } else if i == cost_off + 2 {
            out[i] = val.clamp(-1.5, 1.5);
        } else if i == cost_off + 3 {
            out[i] = (val * 2.0 - 1.0).clamp(-1.0, 1.0);
        } else {
            let m = feat_mean.get(i).copied().unwrap_or(0.0);
            let v = feat_var.get(i).copied().unwrap_or(1.0).max(1e-4);
            out[i] = ((val - m) / v.sqrt()).clamp(-6.0, 6.0);
        }
    }
    out
}

/// Agent's exact live execution self-awareness state `s_{t-1}` (`8-d`), including continuous
/// exponentially-decayed turnover density `ema_turnover` for `Q_hold(x_t, s_t) - Q_flat(x_t, s_t)`.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct SelfAwarenessState {
    pub current_pos: f32,
    pub unrealized_net_ret: f32,
    pub peak_excursion: f32,
    pub drawdown_from_peak: f32,
    pub log_bars_held: f32,
    pub one_way_cost_bps: f32,
    pub session_time_left: f32,
    #[serde(default)]
    pub ema_turnover: f32,
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
            (self.ema_turnover * 2.5).clamp(0.0, 2.5),
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
    #[serde(default = "default_one_f32")]
    pub option_gamma_boost: f32,
    #[serde(default)]
    pub option_theta_per_bar: f32,
    /// Strictly causal counterfactual net-of-round-trip-cost multi-horizon trajectory returns:
    /// `[LongSharesNet, ShortSharesNet, LongOptionNet, ShortOptionNet]` bounded inside session!
    #[serde(default)]
    pub cf_net_returns: [f32; CRITIC_CHANNELS],
    #[serde(default)]
    pub cf_valid: bool,
    /// Strictly causal current-bar intrabar max adverse excursion (shares-equivalent positive fraction).
    #[serde(default)]
    pub intrabar_mae: f32,
    /// Strictly causal future physics targets (populated ONLY when `t + 1 + Delta <= t_current`).
    pub wm_targets: [f32; WM_OUT_DIM],
    pub wm_valid: [bool; WM_OUT_DIM],
}

fn default_true() -> bool {
    true
}

fn default_one_f32() -> f32 {
    1.0
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

/// Automated 4-Vector Loss-Autopsy Fault Diagnosis from Counterfactual + BPTT Saliency Replay.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LossFaultKind {
    /// Entry had `Gross PnL > 0` / marginal gross move inverted into `Net PnL < 0` by round-trip friction,
    /// or failed to beat round-trip cost (`a = 0.0` FLAT optimal).
    #[serde(alias = "EntryHallucination")]
    FrictionInversion,
    /// Reached positive post-cost `V_liq` excursion at `optimal_exit_offset` before decaying into
    /// EOD flatten, catastrophic backstop, or red close.
    #[serde(alias = "HoldingExitFailure")]
    HoldingEodDecay,
    /// Chosen vehicle (`V_SHARES` vs `V_OPTION`) underperformed the counterfactual alternate vehicle.
    VehicleSelectionRegret,
    /// Population collapsed to near-zero trades while positive post-cost counterfactual opportunities existed.
    DeadzoneParalysisRegret,
}

/// Audited Red-Trade / Regret Trajectory stored in the Prioritized Autopsy Replay Buffer.
#[derive(Debug, Clone)]
pub struct AuditedRedTradeTrajectory {
    pub symbol: String,
    pub exit_ts_sec: i64,
    pub net_pnl_usd: f64,
    pub gross_pnl_usd: f64,
    pub net_r: f32,
    pub direction_sign: f32,
    pub is_option_routed: bool,
    /// Asymmetric downside priority weight: `1.0 + 4.0 * |net_r|^2`.
    pub priority_weight: f32,
    pub fault_kind: LossFaultKind,
    pub entry_offset: usize,
    pub optimal_exit_offset: usize,
    pub best_counterfactual_net_r: f32,
    pub vehicle_regret_delta: f32,
    pub saliency_dx: f32,
    pub saliency_dh: f32,
    pub prior_regime_z: [f32; WM_DIM],
    pub bars: Vec<CausalTrajectoryBar>,
}

/// Cumulative per-bucket Autonomous Loss-Autopsy Attribution Telemetry reported in `PROOF_REPORT.md`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BucketAutopsyTelemetry {
    pub brain_label: String,
    pub friction_inversion_events: usize,
    pub holding_eod_decay_events: usize,
    pub vehicle_regret_events: usize,
    pub deadzone_paralysis_events: usize,
    pub flat_regret_windows_replayed: usize,
    pub cost_sensitivity_before: f32,
    pub cost_sensitivity_after: f32,
    pub promoted_generations: u32,
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
    /// Summary 5-Quantile distributional critic predictions for active channel.
    pub quantiles: [f32; CRITIC_QUANTILES],
    /// Full 4-channel x 5-quantile predictions `[LongSh, ShortSh, LongOpt, ShortOpt]`.
    pub critic_all: [f32; CRITIC_OUT_DIM],
    /// Risk-adjusted distributional net advantages per channel.
    pub critic_channel_utils: [f32; CRITIC_CHANNELS],
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
        let w_wm = init_mat(WM_OUT_DIM, d, 0.22, seed);
        let mut b_wm = vec![0.0_f32; WM_OUT_DIM];
        for k in 0..WM_OUT_DIM {
            if k % 4 < 3 {
                b_wm[k] = match k % 4 {
                    0 => -0.12,
                    1 => 0.0,
                    _ => 0.12,
                };
            } else {
                // `ln(Range / RT_Cost)` baseline near 0.35
                b_wm[k] = 0.35;
            }
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
            w_wm,
            b_wm,
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
            let unit_scale = if k % 4 < 3 { RET_LOGIT_UNIT_SCALE } else { 1.0 };
            wm_pred[k] = s * unit_scale;
        }

        (z_next, wm_pred)
    }

    /// Exact single-step analytical input Jacobian `(dz_next/dx_norm)^T * dz_out + (dwm_pred/dx_norm)^T * dwm_pred_out`
    /// so Stage B policy & auxiliary losses propagate `dL/dwm * dwm/dx` directly into input saliency `dL/dx`
    /// with zero detached world-model pathways.
    #[inline]
    pub fn backprop_step_to_x(
        &self,
        x_norm_212: &[f32; RL_STATE_DIM],
        prev_z: &[f32; WM_DIM],
        dz_out: &[f32; WM_DIM],
        dwm_pred_out: &[f32; WM_OUT_DIM],
    ) -> ([f32; RL_STATE_DIM], [f32; WM_DIM]) {
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
                sg += wg[j] * u_t[j] + ug[j] * prev_z[j];
                sr += wr[j] * u_t[j] + ur[j] * prev_z[j];
            }
            g_t[i] = sigmoid(sg);
            r_t[i] = sigmoid(sr);
        }

        let mut h_cand = [0.0_f32; WM_DIM];
        for i in 0..d {
            let wh = &self.w_h[i * d..(i + 1) * d];
            let uh = &self.u_h[i * d..(i + 1) * d];
            let mut sh = self.b_h[i];
            for j in 0..d {
                sh += wh[j] * u_t[j] + uh[j] * (r_t[j] * prev_z[j]);
            }
            h_cand[i] = sh.tanh();
        }

        let mut dz = *dz_out;
        for k in 0..WM_OUT_DIM {
            let unit_scale = if k % 4 < 3 { RET_LOGIT_UNIT_SCALE } else { 1.0 };
            let dwk = dwm_pred_out[k] * unit_scale;
            if dwk.abs() > 0.0 {
                let row = &self.w_wm[k * d..(k + 1) * d];
                for i in 0..d {
                    dz[i] += dwk * row[i];
                }
            }
        }

        let mut dh_cand = [0.0_f32; WM_DIM];
        let mut dg_pre = [0.0_f32; WM_DIM];
        for i in 0..d {
            dh_cand[i] = dz[i] * (1.0 - g_t[i]) * (1.0 - h_cand[i] * h_cand[i]);
            dg_pre[i] = dz[i] * (prev_z[i] - h_cand[i]) * g_t[i] * (1.0 - g_t[i]);
        }

        let mut dr_t = [0.0_f32; WM_DIM];
        let mut du_t = [0.0_f32; WM_DIM];
        let mut dz_prev = [0.0_f32; WM_DIM];
        for i in 0..d {
            dz_prev[i] = dz[i] * g_t[i];
        }
        for i in 0..d {
            let dsh = dh_cand[i];
            for j in 0..d {
                du_t[j] += dsh * self.w_h[i * d + j];
                dr_t[j] += dsh * self.u_h[i * d + j] * prev_z[j];
                dz_prev[j] += dsh * self.u_h[i * d + j] * r_t[j];
            }
        }
        for i in 0..d {
            let dsg = dg_pre[i];
            let dsr = dr_t[i] * r_t[i] * (1.0 - r_t[i]);
            for j in 0..d {
                du_t[j] += dsg * self.w_g[i * d + j] + dsr * self.w_r[i * d + j];
                dz_prev[j] += dsg * self.u_g[i * d + j] + dsr * self.u_r[i * d + j];
            }
        }

        let mut du_norm = [0.0_f32; WM_DIM];
        for i in 0..d {
            du_norm[i] = du_t[i] * (1.0 - u_t[i] * u_t[i]) * self.ln_g[i];
        }
        let inv_d = 1.0 / (d as f32);
        let sum_dun: f32 = du_norm.iter().sum();
        let sum_dun_un: f32 = du_norm
            .iter()
            .zip(u_norm.iter())
            .map(|(a, b)| a * b)
            .sum();
        let mut dx = [0.0_f32; RL_STATE_DIM];
        for i in 0..d {
            let du_raw = inv_std * (du_norm[i] - inv_d * sum_dun - inv_d * u_norm[i] * sum_dun_un);
            let row = &self.w_enc[i * RL_STATE_DIM..(i + 1) * RL_STATE_DIM];
            for j in 0..RL_STATE_DIM {
                dx[j] += du_raw * row[j];
            }
        }
        (dx, dz_prev)
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
            let x_norm = normalize_212_with_stats(&bar.x_212, feat_mean, feat_var);
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
                let unit_scale = if k % 4 < 3 { RET_LOGIT_UNIT_SCALE } else { 1.0 };
                pred[k] = s * unit_scale;
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
                let unit_scale = if k % 4 < 3 { RET_LOGIT_UNIT_SCALE } else { 1.0 };
                let kappa = if k % 4 < 3 { 0.0025 } else { 0.25 };
                let huber_step = (err / kappa).clamp(-1.0, 1.0);
                let asym_w = if err >= 0.0 { tau } else { 1.0 - tau };
                let dl_ds = -asym_w * huber_step * inv_t * 0.35 * unit_scale;
                g_b_wm[k] += dl_ds;
                for i in 0..d {
                    g_w_wm[k * d + i] += dl_ds * c.z_t[i];
                    dz[i] += dl_ds * self.w_wm[k * d + i];
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
        // Prevent unconditional sample-drift memorization in World-Model return quantile intercepts
        for h_idx in 0..WM_HORIZONS.len() {
            let base = h_idx * 4;
            if base + 2 < self.b_wm.len() {
                let q50 = (self.b_wm[base + 1] * 0.90).clamp(-0.025, 0.025);
                let half_iqr = (0.5 * (self.b_wm[base + 2] - self.b_wm[base])).clamp(0.008, 0.45);
                self.b_wm[base] = q50 - half_iqr;
                self.b_wm[base + 1] = q50;
                self.b_wm[base + 2] = q50 + half_iqr;
            }
        }

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
    /// Derived strictly from unconstrained log-space `exp(log_cost_sens)` (zero `2.20` / `2.60` ceiling).
    pub cost_sensitivity: f32,
    #[serde(default)]
    pub log_cost_sens: f32,
    pub timescale_shift: f32,
    #[serde(default = "default_horizon_gamma")]
    pub horizon_gamma: f32,
    #[serde(default = "default_explore_pressure")]
    pub explore_pressure: f32,
    #[serde(default = "default_ret_scale")]
    pub ret_scale: f32,
    #[serde(default)]
    pub conformal_q_bps: f32,
}

fn default_horizon_gamma() -> f32 {
    0.86
}

fn default_explore_pressure() -> f32 {
    0.45
}

fn default_ret_scale() -> f32 {
    0.38
}

impl AgentHyperParams {
    #[inline]
    pub fn set_log_cost_sens(&mut self, log_k: f32) {
        // Unconstrained log-space `cost_sensitivity = exp(log_cost_sens)` across physical regimes (f32 overflow guard only)
        self.log_cost_sens = log_k.clamp(-6.0, 8.0);
        self.cost_sensitivity = self.log_cost_sens.exp();
    }

    #[inline]
    pub fn sync_log_cost_sens(&mut self) {
        if self.log_cost_sens == 0.0 && (self.cost_sensitivity - 1.0).abs() > 1e-4 {
            self.log_cost_sens = self.cost_sensitivity.max(0.01).ln().clamp(-6.0, 8.0);
        }
        self.cost_sensitivity = self.log_cost_sens.clamp(-6.0, 8.0).exp();
    }

    pub fn diverse_prior(idx: usize, seed: &mut u64) -> Self {
        Self::diverse_prior_for_horizon("GENERIC", idx, seed)
    }

    pub fn diverse_prior_for_horizon(horizon_label: &str, idx: usize, seed: &mut u64) -> Self {
        let lrs = [9e-4, 1.4e-3, 2.0e-3, 2.6e-3];
        let l1s = [2e-6, 8e-6, 2.5e-5, 6e-5];
        let downs = [1.0, 1.7, 2.6, 3.8];
        let dds = [0.35, 0.75, 1.25, 1.95];
        let is_scalp_1m = horizon_label.contains("SCALP") && horizon_label.contains("1m");
        let is_scalp_5m = horizon_label.contains("SCALP") && !is_scalp_1m;
        let is_scalp = is_scalp_1m || is_scalp_5m;
        let is_penny = horizon_label.contains("BRAIN-01") || horizon_label.contains("Penny");
        let is_day_large =
            horizon_label.contains("BRAIN-02") || horizon_label.contains("Large-Cap");
        let is_swing = horizon_label.contains("BRAIN-03")
            || horizon_label.contains("BRAIN-04")
            || horizon_label.contains("Swing");
        let chunks = if is_scalp_1m {
            [64usize, 84, 104, 128]
        } else if is_swing {
            [56usize, 72, 96, 112]
        } else {
            [44usize, 60, 78, 96]
        };
        let gammas = if is_scalp_1m {
            [0.80_f32, 0.85, 0.89, 0.92]
        } else if is_scalp_5m {
            [0.76_f32, 0.82, 0.87, 0.91]
        } else if is_swing {
            [0.85_f32, 0.90, 0.93, 0.96]
        } else {
            [0.80_f32, 0.85, 0.90, 0.94]
        };
        let ts_center = if is_scalp {
            -0.25_f32
        } else if is_swing {
            0.30_f32
        } else {
            -0.05_f32
        };
        let cs_grid = if is_scalp_1m {
            [0.50_f32, 0.74, 0.98, 1.26]
        } else if is_scalp_5m {
            [0.64_f32, 0.92, 1.22, 1.54]
        } else if is_penny {
            [0.60_f32, 0.88, 1.18, 1.52]
        } else if is_day_large {
            [0.82_f32, 1.14, 1.48, 1.84]
        } else if is_swing {
            [0.58_f32, 0.86, 1.16, 1.48]
        } else {
            [0.68_f32, 1.00, 1.34, 1.70]
        };
        let base_cs = cs_grid[(idx / 6) % 4] * rand_uniform(seed, 0.90, 1.10);
        let log_cs = base_cs.max(0.12).ln();
        let explores = if is_scalp {
            [0.30_f32, 0.44, 0.58, 0.72]
        } else {
            [0.22_f32, 0.35, 0.50, 0.68]
        };
        let ret_scales = [0.28_f32, 0.35, 0.43, 0.52];
        Self {
            lr: lrs[idx % 4] * rand_uniform(seed, 0.88, 1.12),
            weight_decay: rand_uniform(seed, 6e-5, 6e-4),
            l1_sparsity: l1s[(idx / 4) % 4] * rand_uniform(seed, 0.85, 1.15),
            downside_aversion: downs[(idx + idx / 4) % 4],
            dd_aversion: dds[(idx * 3 + 1) % 4],
            disagree_aversion: rand_uniform(seed, 0.015, 0.09),
            chunk_len: chunks[idx % 4],
            cost_sensitivity: log_cs.exp(),
            log_cost_sens: log_cs,
            timescale_shift: (ts_center + rand_uniform(seed, -0.55, 0.55)).clamp(-1.5, 1.5),
            horizon_gamma: gammas[(idx / 2) % 4],
            explore_pressure: explores[(idx / 3) % 4],
            ret_scale: ret_scales[idx % 4],
            conformal_q_bps: 0.0,
        }
    }

    pub fn mutate(&self, seed: &mut u64) -> Self {
        let factor = |s: &mut u64| -> f32 {
            if rand_uniform(s, 0.0, 1.0) < 0.5 {
                0.84
            } else {
                1.19
            }
        };
        let cur_log = if self.log_cost_sens == 0.0 && (self.cost_sensitivity - 1.0).abs() > 1e-4 {
            self.cost_sensitivity.max(0.01).ln()
        } else {
            self.log_cost_sens
        };
        let mut out = Self {
            lr: (self.lr * factor(seed)).clamp(2e-4, 3.5e-3),
            weight_decay: (self.weight_decay * factor(seed)).clamp(3e-5, 3e-3),
            l1_sparsity: (self.l1_sparsity * factor(seed)).clamp(1e-6, 1.5e-4),
            downside_aversion: (self.downside_aversion * factor(seed)).clamp(0.4, 8.0),
            dd_aversion: (self.dd_aversion * factor(seed)).clamp(0.1, 5.0),
            disagree_aversion: (self.disagree_aversion * factor(seed)).clamp(0.005, 0.25),
            chunk_len: self.chunk_len,
            cost_sensitivity: self.cost_sensitivity,
            log_cost_sens: cur_log,
            timescale_shift: (self.timescale_shift + rand_uniform(seed, -0.2, 0.2))
                .clamp(-1.5, 1.5),
            horizon_gamma: (self.horizon_gamma + rand_uniform(seed, -0.04, 0.04)).clamp(0.70, 0.96),
            explore_pressure: (self.explore_pressure * factor(seed)).clamp(0.12, 1.10),
            ret_scale: (self.ret_scale * factor(seed)).clamp(0.20, 0.70),
            conformal_q_bps: self.conformal_q_bps,
        };
        out.set_log_cost_sens(cur_log + rand_uniform(seed, -0.25, 0.25));
        out
    }
}

/// Computes Distributional Risk-Adjusted Net Advantage from 5 ordered quantiles
/// (`[q10, q25, q50, q75, q90]`): rewards median & upper-tail convexity (`q75, q90`) while
/// pricing lower-tail drawdown (`q10, q25`) organically without hardcoded bps cutoffs.
#[inline]
pub fn compute_critic_channel_advantage(q5: &[f32], downside_aversion: f32) -> f32 {
    if q5.len() < CRITIC_QUANTILES {
        return 0.0;
    }
    let q10 = q5[0];
    let q25 = q10.max(q5[1]);
    let q50 = q25.max(q5[2]);
    let q75 = q50.max(q5[3]);
    let q90 = q75.max(q5[4]);
    let core = 0.20 * q25 + 0.55 * q50 + 0.25 * q75;
    let upside_skew = (q90 - q50).max(0.0);
    let downside_tail = 0.65 * (-q10).max(0.0) + 0.35 * (-q25).max(0.0);
    (core + 0.18 * upside_skew - (0.18 + 0.08 * downside_aversion) * downside_tail)
        .clamp(-0.25, 0.25)
}

/// Computes Distributional Lower-Confidence-Bound `(E[r_net], IQR_spread, LCB_net)` from 5 quantiles:
/// `LCB_net = E[r_net] - lambda_q * Spread(Q_75 - Q_25, Q_50 - Q_10)`.
/// When quantile uncertainty exceeds expected net-of-cost return, `LCB_net <= 0` organically holds FLAT.
#[inline]
pub fn compute_critic_channel_lcb(q5: &[f32], downside_aversion: f32) -> (f32, f32, f32) {
    if q5.len() < CRITIC_QUANTILES {
        return (0.0, 0.0, 0.0);
    }
    let q10 = q5[0];
    let q25 = q10.max(q5[1]);
    let q50 = q25.max(q5[2]);
    let q75 = q50.max(q5[3]);
    let q90 = q75.max(q5[4]);
    let mean_net = 0.12 * q10 + 0.22 * q25 + 0.36 * q50 + 0.20 * q75 + 0.10 * q90;
    let iqr = (q75 - q25).max(0.0);
    let lower_semi = (q50 - q10).max(0.0);
    let spread = 0.65 * iqr + 0.35 * lower_semi;
    let lambda_q = 0.075 + 0.022 * downside_aversion.clamp(0.3, 6.0);
    let lcb_net = (mean_net - lambda_q * spread).clamp(-0.25, 0.25);
    (mean_net, spread, lcb_net)
}

/// Computes directional World-Model Lower-Confidence-Bound net of round-trip cost (in bps) across `[1, 3, 8, 24]`.
#[inline]
fn wm_directional_lcb_bps(
    wm_pred: &[f32; WM_OUT_DIM],
    dir_sign: f32,
    rt_cost_bps: f32,
    downside_aversion: f32,
) -> f32 {
    let mut best_lcb = f32::NEG_INFINITY;
    let lambda_wm = 0.08 + 0.02 * downside_aversion.clamp(0.3, 5.0);
    for h_idx in 0..WM_HORIZONS.len() {
        let base = h_idx * 4;
        let q25 = wm_pred[base];
        let q50 = wm_pred[base + 1];
        let q75 = wm_pred[base + 2];
        let rc_log = wm_pred[base + 3].clamp(-2.5, 3.0);
        let dir_med_bps = if dir_sign > 0.0 {
            (0.25 * q25 + 0.50 * q50 + 0.25 * q75) * 10_000.0
        } else if dir_sign < 0.0 {
            -(0.25 * q25 + 0.50 * q50 + 0.25 * q75) * 10_000.0
        } else {
            q50.abs() * 10_000.0
        };
        let iqr_bps = (q75 - q25).abs() * 10_000.0;
        let rc_ratio = rc_log.exp();
        let low_range_pen_bps = 0.25 * rt_cost_bps.max(6.0) * (1.35 - rc_ratio).max(0.0);
        let lcb_h = dir_med_bps - rt_cost_bps - lambda_wm * iqr_bps - low_range_pen_bps;
        if lcb_h > best_lcb {
            best_lcb = lcb_h;
        }
    }
    if best_lcb.is_finite() {
        best_lcb
    } else {
        -rt_cost_bps
    }
}

/// Computes Cost-Scaled, Turnover-Amortized (`Q_hold - Q_flat`), & Distributional-LCB-Gated
/// Neural Deadzone threshold `(tau_k, d(tau_k)/d(d_pre_k), d(tau_k)/d(log_k_theta))`.
#[allow(clippy::too_many_arguments)]
#[inline]
fn compute_cost_scaled_deadzone(
    d_pre: f32,
    one_way_cost_bps: f32,
    cost_sensitivity: f32,
    wm_dir_lcb_bps: f32,
    dir_critic_lcb_bps: f32,
    is_holding_pos: bool,
    session_time_left: f32,
    ema_turnover: f32,
    peak_excursion: f32,
    drawdown_from_peak: f32,
    log_bars_held: f32,
) -> (f32, f32, f32) {
    let sp = softplus(d_pre);
    let dsp = sigmoid(d_pre);
    let rt_cost_bps = (2.0 * one_way_cost_bps).max(4.0);
    let scale_denom = rt_cost_bps.max(12.0);
    let turn_clamped = ema_turnover.clamp(0.0, 1.5);

    // Unconstrained positive cost sensitivity `k_theta = exp(log_cost_sens)` (zero artificial ceiling).
    let cs = cost_sensitivity.max(0.02);
    if is_holding_pos {
        // Sunk-Cost Amortized Continuation Advantage (`Q_hold(x_t, s_t) - Q_flat(x_t, s_t)`):
        // Once holding, entry spread is sunk; closing early locks in round-trip friction unless MFE
        // has cleared cost (`peak_bps > rt_cost_bps`) and starts giving back (`drawdown_from_peak`),
        // or closing-bell urgency (`session_time_left < 0.26`) rises.
        let sess_rem = session_time_left.clamp(0.0, 1.0);
        let cont_wm_bps = wm_dir_lcb_bps + rt_cost_bps;
        let cont_critic_bps = dir_critic_lcb_bps + rt_cost_bps;
        let sunk_amort = (-0.80 * log_bars_held.max(0.0)).exp();
        let reentry_prob =
            ((0.22 + 0.45 * turn_clamped.min(1.0)) * (0.20 + 0.80 * sess_rem)).clamp(0.05, 0.75);
        let rt_ref_bps = rt_cost_bps.min(36.0);
        let q_flat_bps = -(0.52 * sunk_amort + 0.32 * reentry_prob) * rt_ref_bps;

        let sess_elapsed = (1.0 - sess_rem).clamp(0.0, 1.0);
        let eod_urgency = ((0.26 - sess_rem).max(0.0) / 0.26).powi(2);
        let peak_bps = (peak_excursion.max(0.0)) * 10_000.0;
        let giveback_bps = (drawdown_from_peak.max(0.0)) * 10_000.0;
        let mfe_booked_gate = sigmoid((peak_bps - 0.85 * rt_cost_bps) / scale_denom.max(8.0));
        let mfe_giveback_ratio =
            (giveback_bps / peak_bps.max(rt_cost_bps).max(10.0)).clamp(0.0, 1.5);

        let carry_cost_bps = rt_ref_bps * (0.10 * sess_elapsed + 1.45 * eod_urgency)
            + (0.52 + 0.44 * mfe_booked_gate) * giveback_bps;
        let q_hold_bps = 0.58 * cont_critic_bps + 0.42 * cont_wm_bps - carry_cost_bps;
        let q_adv_bps = q_hold_bps - q_flat_bps;

        let hold_def_arg = (-q_adv_bps) / (1.45 * scale_denom);
        let hold_def_raw = softplus(hold_def_arg);
        let hold_def = hold_def_raw.min(2.20) * 0.19 * (1.0 - 0.50 * sunk_amort);
        let mfe_harvest = 0.24 * mfe_booked_gate * mfe_giveback_ratio;
        let underwater_ratio = giveback_bps / (1.65 * scale_denom).max(20.0);
        let underwater_cut = (1.0 - mfe_booked_gate)
            * 0.50
            * softplus(underwater_ratio - 0.85).min(1.25)
            * (1.0 - 0.55 * sunk_amort);
        let raw_tau_hold = 0.042
            + 0.042 * sp
            + 0.018 * (rt_ref_bps / 100.0)
            + hold_def
            + mfe_harvest
            + underwater_cut
            + 0.30 * eod_urgency;
        let in_hold_bounds = (0.040..0.95).contains(&raw_tau_hold);
        return (
            raw_tau_hold.clamp(0.040, 0.95),
            if in_hold_bounds { 0.042 * dsp } else { 0.0 },
            0.0,
        );
    }

    // Asymmetric Epistemic / Distributional LCB + Turnover-Amortized Entry Gate before paying `rt_cost_bps` (`2c`):
    // Rapid re-entry (`ema_turnover > 0`) and elevated `k_theta = exp(log_k_theta)` raise `excess_hurdle_bps`
    // smoothly with full analytical `d(tau_k)/d(log_k_theta)` hypergradient while remaining bounded against paralysis.
    let joint_lcb_bps = 0.58 * dir_critic_lcb_bps + 0.42 * wm_dir_lcb_bps;
    let turn_mult = 1.0 + 0.75 * turn_clamped;
    let excess_hurdle_bps =
        (0.22 + 0.24 * sp + 0.38 * cs * turn_mult + 0.50 * turn_clamped) * rt_cost_bps;
    let lcb_deficit_bps = excess_hurdle_bps - joint_lcb_bps;
    let def_arg = lcb_deficit_bps / (1.35 * scale_denom);
    let smooth_def_raw = softplus(def_arg);
    let smooth_def = smooth_def_raw.min(2.5) * 0.12;
    let def_slope = if smooth_def_raw < 2.5 {
        sigmoid(def_arg) * 0.12 / (1.35 * scale_denom)
    } else {
        0.0
    };
    let dsmooth_ddpre = def_slope * (0.24 * dsp * rt_cost_bps);
    let dsmooth_dlogk = def_slope * (0.38 * cs * turn_mult * rt_cost_bps);

    let lcb_neg_raw = softplus((-joint_lcb_bps) / (1.5 * scale_denom));
    let lcb_neg_widen = lcb_neg_raw.min(2.0) * 0.06;
    let cost_scale_term = 0.25 * cs * (rt_cost_bps / 100.0);
    let base_tau = 0.125 + 0.070 * sp + cost_scale_term + smooth_def + lcb_neg_widen;
    let in_bounds = (0.10..0.96).contains(&base_tau);
    let dtau_ddpre = if in_bounds {
        0.070 * dsp + dsmooth_ddpre
    } else {
        0.0
    };
    let dtau_dlogk = if in_bounds {
        0.25 * cs * (rt_cost_bps / 100.0) + dsmooth_dlogk
    } else {
        0.02 * cs
    };
    (base_tau.clamp(0.10, 0.96), dtau_ddpre, dtau_dlogk)
}

/// Concatenates LayerNorm policy hidden state (`D = 64`), unit-scaled World-Model predictions (`16-d`),
/// Execution Self-Awareness (`8-d`), and sign-equivariant `(position_sign * outcome)` terms (`4-d`) into `HEAD_IN_DIM = 92`.
#[inline]
fn fill_head_input(
    h_ln: &[f32; HIDDEN_DIM],
    wm_pred: &[f32; WM_OUT_DIM],
    self_state: &SelfAwarenessState,
) -> [f32; HEAD_IN_DIM] {
    let mut head_in = [0.0_f32; HEAD_IN_DIM];
    head_in[..HIDDEN_DIM].copy_from_slice(h_ln);
    for k in 0..WM_OUT_DIM {
        head_in[HIDDEN_DIM + k] = if k % 4 < 3 {
            // Scale decimal return quantiles (`q25, q50, q75`) to O(1) variance matching `h_ln`!
            (wm_pred[k] * 40.0).clamp(-3.0, 3.0)
        } else {
            ((wm_pred[k] - 0.35) * 0.50).clamp(-2.0, 2.0)
        };
    }
    let s_arr = self_state.to_array();
    let s_off = HIDDEN_DIM + WM_OUT_DIM;
    head_in[s_off..s_off + SELF_STATE_DIM].copy_from_slice(&s_arr);

    let pos_sgn = if self_state.current_pos.abs() > 1e-4 {
        self_state.current_pos.signum()
    } else {
        0.0
    };
    // Add back sunk round-trip entry reserve (`2 * one_way_cost`) in `pos_interact[0]` so bar-1
    // post-entry MTM excursion starts at `0.0` rather than shocking `z_pre` into immediate reversal,
    // while exact `V_liq` remains directly observable in `s_arr[1]`.
    let sunk_rt_frac = 2.0 * (self_state.one_way_cost_bps / 10_000.0);
    let unreal_mtm = if pos_sgn != 0.0 {
        ((self_state.unrealized_net_ret + sunk_rt_frac) * 20.0).clamp(-3.0, 3.0)
    } else {
        0.0
    };
    let dd_peak = (self_state.drawdown_from_peak * 25.0).clamp(0.0, 3.0);
    let elapsed_sess = (1.0 - self_state.session_time_left.clamp(0.0, 1.0)).clamp(0.0, 1.0);
    let log_bars_norm = (self_state.log_bars_held / 4.0).clamp(0.0, 1.5);
    let p_off = s_off + SELF_STATE_DIM;
    head_in[p_off] = pos_sgn * unreal_mtm;
    head_in[p_off + 1] = pos_sgn * dd_peak;
    head_in[p_off + 2] = pos_sgn * elapsed_sess;
    head_in[p_off + 3] = pos_sgn * log_bars_norm;
    head_in
}

struct EpisodeCounterfactuals {
    /// Post-RT-cost entry counterfactual targets per channel `[LongSh, ShortSh, LongOpt, ShortOpt]`.
    critic_targets: Vec<[f32; CRITIC_CHANNELS]>,
    /// Sunk-entry Mark-to-Liquidation continuation values `[LongSh, ShortSh, LongOpt, ShortOpt]`.
    hold_cont_channels: Vec<[f32; CRITIC_CHANNELS]>,
    /// Best viable net post-RT-cost opportunity at each bar (for localized boundary wakeup vs flat reinforcement).
    best_entry_net: Vec<f32>,
    /// Directional multi-bar + intrabar max adverse excursion `[LongSharesMAE, ShortSharesMAE]` over next 16 bars.
    path_mae: Vec<[f32; 2]>,
    /// Session-bounded EOD continuation value per channel `[LongSh, ShortSh, LongOpt, ShortOpt]`.
    hold_to_eod: Vec<[f32; CRITIC_CHANNELS]>,
}

/// Computes strictly episode-bounded multi-horizon post-cost counterfactual targets
/// (`[3, 6, 12, 24]` on intraday sessions with `force_flat`, `[2, 4, 8, 16]` on multi-day `1h` swing)
/// plus BSM gamma convexity, EOD continuation, and directional path MAE (never crossing `force_flat` EOD boundaries).
fn compute_episode_counterfactual_targets(
    chunk: &[CausalTrajectoryBar],
    gamma: f32,
) -> EpisodeCounterfactuals {
    let t_len = chunk.len();
    let mut critic_targets = vec![[0.0_f32; CRITIC_CHANNELS]; t_len];
    let mut hold_cont_channels = vec![[0.0_f32; CRITIC_CHANNELS]; t_len];
    let mut best_entry_net = vec![0.0_f32; t_len];
    let mut path_mae = vec![[0.0_f32; 2]; t_len];
    let mut hold_to_eod = vec![[0.0_f32; CRITIC_CHANNELS]; t_len];
    let is_intraday_ep = chunk.iter().any(|b| b.force_flat);
    let horizons: [usize; 4] = if is_intraday_ep && t_len >= 140 {
        [10, 25, 50, 90]
    } else if is_intraday_ep {
        [4, 9, 18, 32]
    } else {
        [2, 4, 8, 16]
    };
    let max_h = horizons[3];

    for t in 0..t_len {
        if chunk[t].force_flat {
            continue;
        }
        let mut eod_lim = t_len;
        for (idx, b) in chunk.iter().enumerate().skip(t) {
            if b.force_flat {
                eod_lim = idx;
                break;
            }
        }
        if eod_lim <= t {
            continue;
        }

        let max_end = (t + max_h).min(eod_lim);
        let c_sh = chunk[t].one_way_cost_shares.max(0.0);
        let c_opt = chunk[t].one_way_cost_option.max(c_sh);
        let opt_ok = chunk[t].option_allowed;

        let mut mae = [0.0_f32; CRITIC_CHANNELS];
        let mut mae_16 = [0.0_f32; 2];
        let mut acc_ent = [0.0_f32; CRITIC_CHANNELS];
        let mut acc_hld = [0.0_f32; CRITIC_CHANNELS];
        let mut best_ent = [f32::NEG_INFINITY; CRITIC_CHANNELS];
        let mut best_hld = [f32::NEG_INFINITY; CRITIC_CHANNELS];
        let mut w_sum = 0.0_f32;

        let mut g_sh = 0.0_f32;
        let mut g_opt = 0.0_f32;
        let mut g_theta = 0.0_f32;

        for (step_offset, b) in chunk[t..eod_lim].iter().enumerate() {
            let steps = step_offset + 1;
            let (r_opt_bar, theta_bar) = if b.option_gamma_boost > 1e-6 {
                (
                    b.ret_oo_shares * b.option_gamma_boost,
                    b.option_theta_per_bar.max(0.0),
                )
            } else {
                (b.ret_oo_option, 0.0)
            };
            g_sh += b.ret_oo_shares;
            g_opt += r_opt_bar;
            g_theta += theta_bar;

            let cur_hld = [
                g_sh,
                -g_sh,
                if opt_ok { g_opt - g_theta } else { g_sh },
                if opt_ok { -g_opt - g_theta } else { -g_sh },
            ];
            let bar_intrabar = b.intrabar_mae.max(0.0);
            let step_adv = [
                (-cur_hld[0] + 0.42 * bar_intrabar).max(0.0),
                (-cur_hld[1] + 0.42 * bar_intrabar).max(0.0),
                (-cur_hld[2] + 0.30 * bar_intrabar).max(0.0),
                (-cur_hld[3] + 0.30 * bar_intrabar).max(0.0),
            ];
            for ch in 0..CRITIC_CHANNELS {
                if step_adv[ch] > mae[ch] {
                    mae[ch] = step_adv[ch];
                }
            }
            if t + step_offset < max_end {
                if step_adv[0] > mae_16[0] {
                    mae_16[0] = step_adv[0];
                }
                if step_adv[1] > mae_16[1] {
                    mae_16[1] = step_adv[1];
                }
            }

            if t + step_offset < max_end {
                let at_checkpoint = horizons.contains(&steps) || (t + steps == eod_lim);
                if at_checkpoint {
                    let disc = gamma.powf((steps as f32 - 1.0) * 0.22);
                    let w = if steps == horizons[0] {
                        0.16
                    } else if steps == horizons[1] {
                        0.28
                    } else if steps == horizons[2] {
                        0.32
                    } else if steps == horizons[3] {
                        0.24
                    } else {
                        0.20
                    };
                    w_sum += w;
                    for ch in 0..CRITIC_CHANNELS {
                        let hld_val = cur_hld[ch] * disc - 0.18 * mae[ch];
                        acc_hld[ch] += w * hld_val;
                        if hld_val > best_hld[ch] {
                            best_hld[ch] = hld_val;
                        }
                    }
                    let ent_ch = [
                        cur_hld[0] * disc - 0.18 * mae[0] - 2.0 * c_sh,
                        cur_hld[1] * disc - 0.18 * mae[1] - 2.0 * c_sh,
                        if opt_ok {
                            cur_hld[2] * disc - 0.18 * mae[2] - 2.0 * c_opt
                        } else {
                            -2.0 * c_opt
                        },
                        if opt_ok {
                            cur_hld[3] * disc - 0.18 * mae[3] - 2.0 * c_opt
                        } else {
                            -2.0 * c_opt
                        },
                    ];
                    for ch in 0..CRITIC_CHANNELS {
                        acc_ent[ch] += w * ent_ch[ch];
                        if ent_ch[ch] > best_ent[ch] {
                            best_ent[ch] = ent_ch[ch];
                        }
                    }
                }
            }

            if t + steps == eod_lim {
                let eod_disc = gamma.powf((steps as f32 - 1.0) * 0.15);
                for ch in 0..CRITIC_CHANNELS {
                    hold_to_eod[t][ch] =
                        (cur_hld[ch] * eod_disc - 0.26 * mae[ch]).clamp(-0.35, 0.35);
                }
            }
        }
        path_mae[t] = [
            mae_16[0].max(chunk[t].intrabar_mae.max(0.0)).min(0.18),
            mae_16[1].max(chunk[t].intrabar_mae.max(0.0)).min(0.18),
        ];

        for ch in 0..CRITIC_CHANNELS {
            let multi_ent = if w_sum > 0.0 {
                let mean_e = acc_ent[ch] / w_sum;
                let peak_e = if best_ent[ch].is_finite() {
                    best_ent[ch]
                } else {
                    mean_e
                };
                (0.64 * mean_e + 0.36 * peak_e).clamp(-0.35, 0.35)
            } else {
                0.0
            };
            let multi_hld = if w_sum > 0.0 {
                let mean_h = acc_hld[ch] / w_sum;
                let peak_h = if best_hld[ch].is_finite() {
                    best_hld[ch]
                } else {
                    mean_h
                };
                (0.64 * mean_h + 0.36 * peak_h).clamp(-0.35, 0.35)
            } else {
                0.0
            };
            critic_targets[t][ch] = if chunk[t].cf_valid {
                (0.55 * chunk[t].cf_net_returns[ch] + 0.45 * multi_ent).clamp(-0.35, 0.35)
            } else {
                multi_ent
            };
            hold_cont_channels[t][ch] = multi_hld;
        }
        let max_sh = critic_targets[t][0].max(critic_targets[t][1]);
        let max_opt = if opt_ok {
            critic_targets[t][2].max(critic_targets[t][3])
        } else {
            f32::NEG_INFINITY
        };
        best_entry_net[t] = max_sh.max(max_opt);
    }

    EpisodeCounterfactuals {
        critic_targets,
        hold_cont_channels,
        best_entry_net,
        path_mae,
        hold_to_eod,
    }
}

/// Conditioned Active-Position Trajectory Context for `HoldingEodDecay` Loss-Autopsy Replay:
/// Replays the exact held position state (`current_pos`, `unrealized_net_ret`, `peak_excursion`,
/// `drawdown_from_peak`, `session_time_left`) from `entry_offset` onward so fast/medium recurrent
/// gates receive dense per-bar counterfactual exit regret `(V_liq(t_opt) - V_liq(T_eod))`.
#[derive(Debug, Clone, Copy)]
pub struct ForcedHeldContext {
    pub entry_offset: usize,
    pub optimal_exit_offset: usize,
    pub direction_sign: f32,
    pub is_option_routed: bool,
    pub peak_giveback_ret: f32,
}

#[inline]
fn nonsat_input_act(pre: f32) -> (f32, f32) {
    let th = pre.clamp(-12.0, 12.0).tanh();
    let lin = (pre * 0.25).clamp(-1.0, 1.0);
    let val = 0.85 * th + 0.15 * lin;
    let deriv = 0.85 * (1.0 - th * th) + if pre.abs() < 4.0 { 0.0375 } else { 0.0 };
    (val, deriv)
}

/// Stage B: `K = 5` Orthogonal Specialist Policy Ensemble + Joint Auxiliary Physics World-Model Head
/// + Distributional Quantile Critic + Diagonal Fisher EWC over `D = 64` Multi-Timescale Gated Core.
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
    /// Jointly differentiable auxiliary World-Model head sharing recurrent `h_ln` (`WM_OUT_DIM * HIDDEN_DIM`)
    #[serde(default)]
    w_wm_aux: Vec<f32>,
    #[serde(default)]
    b_wm_aux: Vec<f32>,
    // K = 5 orthogonal specialist policy heads over `HEAD_IN_DIM = 92` (`[h_ln, wm_fused, self_state, pos_interact]`)
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

/// Variance-reduced gradient accumulator across multiple causal chunks before a single AdamW step,
/// including unconstrained differentiable hypergradient `g_log_cost_sens = dL_net / d(log_k_theta)`.
#[derive(Clone, Debug)]
pub struct PolicyGradAccum {
    pub g_w_in: Vec<f32>,
    pub g_b_in: Vec<f32>,
    pub g_ln_in_g: Vec<f32>,
    pub g_ln_in_b: Vec<f32>,
    pub g_w_g: Vec<f32>,
    pub g_u_g: Vec<f32>,
    pub g_b_g: Vec<f32>,
    pub g_w_r: Vec<f32>,
    pub g_u_r: Vec<f32>,
    pub g_b_r: Vec<f32>,
    pub g_w_h: Vec<f32>,
    pub g_u_h: Vec<f32>,
    pub g_b_h: Vec<f32>,
    pub g_ln_out_g: Vec<f32>,
    pub g_ln_out_b: Vec<f32>,
    pub g_w_wm_aux: Vec<f32>,
    pub g_b_wm_aux: Vec<f32>,
    pub g_w_a: Vec<f32>,
    pub g_b_a: Vec<f32>,
    pub g_w_d: Vec<f32>,
    pub g_b_d: Vec<f32>,
    pub g_w_v: Vec<f32>,
    pub g_b_v: Vec<f32>,
    pub g_w_s: Vec<f32>,
    pub g_b_s: Vec<f32>,
    pub g_w_c: Vec<f32>,
    pub g_b_c: Vec<f32>,
    pub g_log_cost_sens: f32,
    pub chunks_count: usize,
}

impl PolicyGradAccum {
    fn slices(&self) -> [&[f32]; 27] {
        [
            &self.g_w_in,
            &self.g_b_in,
            &self.g_ln_in_g,
            &self.g_ln_in_b,
            &self.g_w_g,
            &self.g_u_g,
            &self.g_b_g,
            &self.g_w_r,
            &self.g_u_r,
            &self.g_b_r,
            &self.g_w_h,
            &self.g_u_h,
            &self.g_b_h,
            &self.g_ln_out_g,
            &self.g_ln_out_b,
            &self.g_w_wm_aux,
            &self.g_b_wm_aux,
            &self.g_w_a,
            &self.g_b_a,
            &self.g_w_d,
            &self.g_b_d,
            &self.g_w_v,
            &self.g_b_v,
            &self.g_w_s,
            &self.g_b_s,
            &self.g_w_c,
            &self.g_b_c,
        ]
    }

    fn slices_mut(&mut self) -> [&mut [f32]; 27] {
        [
            &mut self.g_w_in,
            &mut self.g_b_in,
            &mut self.g_ln_in_g,
            &mut self.g_ln_in_b,
            &mut self.g_w_g,
            &mut self.g_u_g,
            &mut self.g_b_g,
            &mut self.g_w_r,
            &mut self.g_u_r,
            &mut self.g_b_r,
            &mut self.g_w_h,
            &mut self.g_u_h,
            &mut self.g_b_h,
            &mut self.g_ln_out_g,
            &mut self.g_ln_out_b,
            &mut self.g_w_wm_aux,
            &mut self.g_b_wm_aux,
            &mut self.g_w_a,
            &mut self.g_b_a,
            &mut self.g_w_d,
            &mut self.g_b_d,
            &mut self.g_w_v,
            &mut self.g_b_v,
            &mut self.g_w_s,
            &mut self.g_b_s,
            &mut self.g_w_c,
            &mut self.g_b_c,
        ]
    }

    pub fn zero(&mut self) {
        for s in self.slices_mut() {
            s.fill(0.0);
        }
        self.g_log_cost_sens = 0.0;
        self.chunks_count = 0;
    }

    /// Applies unconstrained differentiable hypergradient `dL_net / d(log_k_theta)` to `hparams.log_cost_sens`
    /// (`k_theta = exp(log_k_theta)` with zero `1.450` or `2.600` ceiling).
    pub fn step_hyperparams(&self, hparams: &mut AgentHyperParams, lr_scale: f32) {
        if self.chunks_count == 0 || lr_scale <= 0.0 || !self.g_log_cost_sens.is_finite() {
            return;
        }
        let g_hyper = (self.g_log_cost_sens / (self.chunks_count as f32)).clamp(-0.40, 0.40);
        if g_hyper.abs() <= 1e-7 {
            return;
        }
        let hyper_lr = (0.045 * lr_scale).clamp(0.0, 0.08);
        hparams.sync_log_cost_sens();
        hparams.set_log_cost_sens(hparams.log_cost_sens - hyper_lr * g_hyper);
    }

    pub fn mean_l2_norm(&self) -> f32 {
        if self.chunks_count == 0 {
            return 0.0;
        }
        let inv_n = 1.0 / (self.chunks_count as f32);
        let mut sq = 0.0_f32;
        for sl in self.slices() {
            for &g in sl {
                let gs = g * inv_n;
                sq += gs * gs;
            }
        }
        sq.sqrt()
    }

    pub fn add_normalized_from(&mut self, other: &Self, ema_norm: f32, target_scale: f32) {
        if other.chunks_count == 0 || target_scale.abs() <= 1e-8 {
            return;
        }
        let norm_ratio = (target_scale / ema_norm.max(1e-4)).clamp(0.02, 5.0);
        let scale = norm_ratio / (other.chunks_count as f32);
        for (dst, src) in self.slices_mut().into_iter().zip(other.slices()) {
            for (d_v, &s_v) in dst.iter_mut().zip(src.iter()) {
                *d_v += s_v * scale;
            }
        }
        self.g_log_cost_sens += other.g_log_cost_sens * scale;
        self.chunks_count += 1;
    }

    fn flatten(&self) -> Vec<f32> {
        let total: usize = self.slices().iter().map(|s| s.len()).sum();
        let mut flat = Vec::with_capacity(total);
        for s in self.slices() {
            flat.extend_from_slice(s);
        }
        flat
    }
}

#[inline]
fn gcd_usize(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

impl SequencePolicyNet {
    pub fn new(seed: &mut u64, timescale_shift: f32) -> Self {
        let d = HIDDEN_DIM;
        let hd = HEAD_IN_DIM;
        let mut b_g = Vec::with_capacity(d);
        for i in 0..d {
            // Fast / Medium / Slow multi-timescale gates kept strictly away from sigmoid saturation
            let base_tau = match i % 3 {
                0 => -0.85,
                1 => 0.55,
                _ => 1.85,
            };
            b_g.push(base_tau + timescale_shift);
        }

        let w_wm_aux = init_mat(WM_OUT_DIM, d, 0.22, seed);
        let mut b_wm_aux = vec![0.0_f32; WM_OUT_DIM];
        for k in 0..WM_OUT_DIM {
            if k % 4 < 3 {
                b_wm_aux[k] = match k % 4 {
                    0 => -0.12,
                    1 => 0.0,
                    _ => 0.12,
                };
            } else {
                b_wm_aux[k] = 0.35;
            }
        }

        // Orthogonalized sign-symmetric initialization across K = 5 specialist action heads
        let mut w_a_heads = init_mat(SPECIALIST_K, hd, 0.95, seed);
        for k in 0..SPECIALIST_K {
            // Zero-init unsigned magnitude/hurdle indices on directional `w_a_heads` so Epoch 0 has zero LONG/SHORT DC bias!
            for h_i in 0..WM_HORIZONS.len() {
                w_a_heads[k * hd + d + h_i * 4 + 3] = 0.0;
            }
            let s_off = d + WM_OUT_DIM;
            for u_idx in [2usize, 3, 4, 5, 6, 7] {
                w_a_heads[k * hd + s_off + u_idx] = 0.0;
            }
        }
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
        // Seed World-Model directional quantile skip-weights (40x scaled in `head_in`) so latent physics flows symmetrically
        for k in 0..SPECIALIST_K {
            let h_pick = (k % WM_HORIZONS.len()) * 4 + 1; // q50 of horizon k%4
            w_a_heads[k * hd + d + h_pick] += 0.36;
        }
        let mut b_c = vec![0.0_f32; CRITIC_OUT_DIM];
        for ch in 0..CRITIC_CHANNELS {
            let base = ch * CRITIC_QUANTILES;
            b_c[base] = -0.40;
            b_c[base + 1] = -0.18;
            b_c[base + 2] = -0.02;
            b_c[base + 3] = 0.14;
            b_c[base + 4] = 0.34;
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
            w_wm_aux,
            b_wm_aux,
            w_a_heads,
            b_a_heads: vec![0.0; SPECIALIST_K],
            w_d_heads: init_mat(SPECIALIST_K, hd, 0.25, seed),
            b_d_heads: vec![-0.75; SPECIALIST_K],
            w_v_heads: init_mat(SPECIALIST_K, hd, 0.20, seed),
            b_v_heads: vec![-2.4; SPECIALIST_K],
            w_s_heads: init_mat(SPECIALIST_K, hd, 0.25, seed),
            b_s_heads: vec![-0.95; SPECIALIST_K],
            spec_weights: [1.0 / (SPECIALIST_K as f32); SPECIALIST_K],
            w_c: init_mat(CRITIC_OUT_DIM, hd, 0.20, seed),
            b_c,
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
            + self.w_wm_aux.len()
            + self.b_wm_aux.len()
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
            || self.w_wm_aux.len() != WM_OUT_DIM * d
            || self.b_wm_aux.len() != WM_OUT_DIM
            || self.w_a_heads.len() != SPECIALIST_K * hd
            || self.w_c.len() != CRITIC_OUT_DIM * hd
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
        flat.extend_from_slice(&self.w_wm_aux);
        flat.extend_from_slice(&self.b_wm_aux);
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
        for w in &mut self.w_wm_aux {
            *w += rand_normal(seed) * (noise_std * 0.15);
        }
        self.adam_m.fill(0.0);
        self.adam_v.fill(0.0);
        self.adam_step = 0;
    }

    /// Trust-region Polyak parameter interpolation `self <- (1 - beta) * self + beta * other`.
    pub fn polyak_blend_from(&mut self, other: &Self, beta: f32) {
        let b = beta.clamp(0.0, 1.0);
        if b <= 1e-6 || self.param_count() != other.param_count() {
            return;
        }
        let inv = 1.0 - b;
        let blend_slice = |dst: &mut [f32], src: &[f32]| {
            for (d, &s) in dst.iter_mut().zip(src.iter()) {
                let v = inv * *d + b * s;
                if v.is_finite() {
                    *d = v;
                }
            }
        };
        blend_slice(&mut self.w_in, &other.w_in);
        blend_slice(&mut self.b_in, &other.b_in);
        blend_slice(&mut self.ln_in_g, &other.ln_in_g);
        blend_slice(&mut self.ln_in_b, &other.ln_in_b);
        blend_slice(&mut self.w_g, &other.w_g);
        blend_slice(&mut self.u_g, &other.u_g);
        blend_slice(&mut self.b_g, &other.b_g);
        blend_slice(&mut self.w_r, &other.w_r);
        blend_slice(&mut self.u_r, &other.u_r);
        blend_slice(&mut self.b_r, &other.b_r);
        blend_slice(&mut self.w_h, &other.w_h);
        blend_slice(&mut self.u_h, &other.u_h);
        blend_slice(&mut self.b_h, &other.b_h);
        blend_slice(&mut self.ln_out_g, &other.ln_out_g);
        blend_slice(&mut self.ln_out_b, &other.ln_out_b);
        blend_slice(&mut self.w_wm_aux, &other.w_wm_aux);
        blend_slice(&mut self.b_wm_aux, &other.b_wm_aux);
        blend_slice(&mut self.w_a_heads, &other.w_a_heads);
        blend_slice(&mut self.b_a_heads, &other.b_a_heads);
        blend_slice(&mut self.w_d_heads, &other.w_d_heads);
        blend_slice(&mut self.b_d_heads, &other.b_d_heads);
        blend_slice(&mut self.w_v_heads, &other.w_v_heads);
        blend_slice(&mut self.b_v_heads, &other.b_v_heads);
        blend_slice(&mut self.w_s_heads, &other.w_s_heads);
        blend_slice(&mut self.b_s_heads, &other.b_s_heads);
        blend_slice(&mut self.w_c, &other.w_c);
        blend_slice(&mut self.b_c, &other.b_c);
        let mut sw_sum = 0.0_f32;
        for (dst, &src) in self.spec_weights.iter_mut().zip(other.spec_weights.iter()) {
            *dst = (inv * *dst + b * src).max(0.02);
            sw_sum += *dst;
        }
        if sw_sum > 1e-6 {
            for w in &mut self.spec_weights {
                *w /= sw_sum;
            }
        }
    }

    #[inline]
    fn eval_aux_wm(&self, h_ln: &[f32; HIDDEN_DIM]) -> [f32; WM_OUT_DIM] {
        let d = HIDDEN_DIM;
        let mut out = [0.0_f32; WM_OUT_DIM];
        if self.w_wm_aux.len() >= WM_OUT_DIM * d && self.b_wm_aux.len() >= WM_OUT_DIM {
            for k in 0..WM_OUT_DIM {
                let row = &self.w_wm_aux[k * d..(k + 1) * d];
                let mut s = self.b_wm_aux[k];
                for i in 0..d {
                    s += row[i] * h_ln[i];
                }
                let unit_scale = if k % 4 < 3 { RET_LOGIT_UNIT_SCALE } else { 1.0 };
                out[k] = s * unit_scale;
            }
        }
        out
    }

    #[inline]
    fn eval_critic_lcb_channels(
        &self,
        head_in: &[f32; HEAD_IN_DIM],
        downside_aversion: f32,
    ) -> (
        [f32; CRITIC_OUT_DIM],
        [f32; CRITIC_CHANNELS],
        [f32; CRITIC_CHANNELS],
    ) {
        let hd = HEAD_IN_DIM;
        let mut q_all = [0.0_f32; CRITIC_OUT_DIM];
        if self.w_c.len() >= CRITIC_OUT_DIM * hd && self.b_c.len() >= CRITIC_OUT_DIM {
            for idx in 0..CRITIC_OUT_DIM {
                let row = &self.w_c[idx * hd..(idx + 1) * hd];
                let mut s = self.b_c[idx];
                for i in 0..hd {
                    s += row[i] * head_in[i];
                }
                q_all[idx] = (s * RET_LOGIT_UNIT_SCALE).clamp(-0.35, 0.35);
            }
        }
        let mut channel_u = [0.0_f32; CRITIC_CHANNELS];
        let mut channel_lcb = [0.0_f32; CRITIC_CHANNELS];
        for c in 0..CRITIC_CHANNELS {
            let base = c * CRITIC_QUANTILES;
            let q5 = &q_all[base..base + CRITIC_QUANTILES];
            channel_u[c] = compute_critic_channel_advantage(q5, downside_aversion);
            channel_lcb[c] = compute_critic_channel_lcb(q5, downside_aversion).2;
        }
        (q_all, channel_u, channel_lcb)
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
        self.forward_step_with_wm_hp(
            x_norm_212,
            z_wm,
            wm_pred,
            self_state,
            prev_h,
            next_wm_hidden,
            cost_sensitivity,
            0.35,
            option_allowed,
        )
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn forward_step_with_wm_hp(
        &self,
        x_norm_212: &[f32; RL_STATE_DIM],
        z_wm: &[f32; WM_DIM],
        wm_pred: &[f32; WM_OUT_DIM],
        self_state: SelfAwarenessState,
        prev_h: &[f32; HIDDEN_DIM],
        next_wm_hidden: [f32; WM_DIM],
        cost_sensitivity: f32,
        downside_aversion: f32,
        option_allowed: bool,
    ) -> SequenceStepOutput {
        self.forward_step_with_wm_cal(
            x_norm_212,
            z_wm,
            wm_pred,
            self_state,
            prev_h,
            next_wm_hidden,
            cost_sensitivity,
            downside_aversion,
            0.0,
            option_allowed,
        )
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn forward_step_with_wm_cal(
        &self,
        x_norm_212: &[f32; RL_STATE_DIM],
        z_wm: &[f32; WM_DIM],
        wm_pred: &[f32; WM_OUT_DIM],
        self_state: SelfAwarenessState,
        prev_h: &[f32; HIDDEN_DIM],
        next_wm_hidden: [f32; WM_DIM],
        cost_sensitivity: f32,
        downside_aversion: f32,
        conformal_q_bps: f32,
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
            let pre = (u_raw[i] - mean) * inv_std * self.ln_in_g[i] + self.ln_in_b[i];
            u_t[i] = nonsat_input_act(pre).0;
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
        let mut h_ln = [0.0_f32; HIDDEN_DIM];
        for i in 0..d {
            h_ln[i] = (h_next[i] - h_mean) * h_inv_std * self.ln_out_g[i] + self.ln_out_b[i];
        }
        let wm_aux_pred = self.eval_aux_wm(&h_ln);
        let has_ext_wm =
            z_wm.iter().any(|v| v.abs() > 1e-8) || wm_pred.iter().any(|v| v.abs() > 1e-8);
        let mut wm_fused = [0.0_f32; WM_OUT_DIM];
        for k in 0..WM_OUT_DIM {
            wm_fused[k] = if has_ext_wm {
                0.55 * wm_pred[k] + 0.45 * wm_aux_pred[k]
            } else {
                wm_aux_pred[k]
            };
        }
        let head_in = fill_head_input(&h_ln, &wm_fused, &self_state);

        let (q_all, channel_u, channel_lcb) =
            self.eval_critic_lcb_channels(&head_in, downside_aversion);
        let best_long_lcb = if option_allowed {
            channel_lcb[0].max(channel_lcb[2])
        } else {
            channel_lcb[0]
        };
        let best_short_lcb = if option_allowed {
            channel_lcb[1].max(channel_lcb[3])
        } else {
            channel_lcb[1]
        };
        let is_holding = self_state.current_pos.abs() > 1e-4;
        let rt_cost_bps = (2.0 * self_state.one_way_cost_bps).max(4.0);
        let wm_lcb_long = wm_directional_lcb_bps(&wm_fused, 1.0, rt_cost_bps, downside_aversion);
        let wm_lcb_short = wm_directional_lcb_bps(&wm_fused, -1.0, rt_cost_bps, downside_aversion);

        let mut a_heads = [0.0_f32; SPECIALIST_K];
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
            let (wm_dir_lcb_bps, dir_critic_lcb_bps) = if is_holding {
                if self_state.current_pos >= 0.0 {
                    (
                        wm_lcb_long,
                        best_long_lcb * 10_000.0 - self_state.drawdown_from_peak * 1_250.0,
                    )
                } else {
                    (
                        wm_lcb_short,
                        best_short_lcb * 10_000.0 - self_state.drawdown_from_peak * 1_250.0,
                    )
                }
            } else if z_k >= 0.0 {
                (wm_lcb_long - conformal_q_bps, best_long_lcb * 10_000.0)
            } else {
                (wm_lcb_short - conformal_q_bps, best_short_lcb * 10_000.0)
            };
            let (tau_k, _, _) = compute_cost_scaled_deadzone(
                d_pre,
                self_state.one_way_cost_bps,
                cost_sensitivity,
                wm_dir_lcb_bps,
                dir_critic_lcb_bps,
                is_holding,
                self_state.session_time_left,
                self_state.ema_turnover,
                self_state.peak_excursion,
                self_state.drawdown_from_peak,
                self_state.log_bars_held,
            );
            // Exact 0.0 FLAT inside cost-scaled deadzone
            let a_k = if z_k.abs() <= tau_k {
                0.0
            } else {
                (z_k.signum() * (z_k.abs() - tau_k) / (1.0 - tau_k).max(0.15)).clamp(-1.0, 1.0)
            };
            let opt_adv_dir = if a_k >= 0.0 {
                channel_u[2] - channel_u[0]
            } else {
                channel_u[3] - channel_u[1]
            };
            a_heads[k] = a_k;
            a_mean += w_k * a_k;
            z_mean += w_k * z_k;
            delta_mean += w_k * tau_k;
            v_mean += w_k
                * if option_allowed {
                    sigmoid(v_pre + (opt_adv_dir * 16.0).clamp(-1.1, 1.1))
                } else {
                    0.0
                };
            s_mean += w_k * (softplus(s_pre) * 0.036 + 0.011);
        }

        let mut a_var = 0.0_f32;
        for k in 0..SPECIALIST_K {
            let diff = a_heads[k] - a_mean;
            a_var += self.spec_weights[k] * diff * diff;
        }
        let a_std = a_var.max(0.0).sqrt();
        let action_mag = if !is_holding {
            (a_mean.abs() - 0.16 * a_std).max(0.0)
        } else {
            a_mean.abs()
        };
        let action = if action_mag <= 1e-6 {
            0.0
        } else {
            (a_mean.signum() * action_mag).clamp(-1.0, 1.0)
        };
        let active_ch = if action >= 0.0 {
            if v_mean >= 0.5 && option_allowed {
                2
            } else {
                0
            }
        } else if v_mean >= 0.5 && option_allowed {
            3
        } else {
            1
        };
        let mut quantiles = [0.0_f32; CRITIC_QUANTILES];
        let q_off = active_ch * CRITIC_QUANTILES;
        quantiles.copy_from_slice(&q_all[q_off..q_off + CRITIC_QUANTILES]);

        SequenceStepOutput {
            action,
            raw_z: z_mean,
            deadzone: delta_mean,
            vehicle_prob: v_mean.clamp(0.0, 1.0),
            tail_stop_dist: s_mean.clamp(0.010, 0.068),
            quantiles,
            critic_all: q_all,
            critic_channel_utils: channel_u,
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
        self.train_bptt_chunk_full(
            chunk,
            init_wm_z,
            world_model,
            feat_mean,
            feat_var,
            hparams,
            lr_scale,
            ewc_lambda,
            target_override,
            None,
            accumulate_fisher_into,
        )
    }

    pub fn new_grad_accum(&self) -> PolicyGradAccum {
        PolicyGradAccum {
            g_w_in: vec![0.0_f32; self.w_in.len()],
            g_b_in: vec![0.0_f32; self.b_in.len()],
            g_ln_in_g: vec![0.0_f32; self.ln_in_g.len()],
            g_ln_in_b: vec![0.0_f32; self.ln_in_b.len()],
            g_w_g: vec![0.0_f32; self.w_g.len()],
            g_u_g: vec![0.0_f32; self.u_g.len()],
            g_b_g: vec![0.0_f32; self.b_g.len()],
            g_w_r: vec![0.0_f32; self.w_r.len()],
            g_u_r: vec![0.0_f32; self.u_r.len()],
            g_b_r: vec![0.0_f32; self.b_r.len()],
            g_w_h: vec![0.0_f32; self.w_h.len()],
            g_u_h: vec![0.0_f32; self.u_h.len()],
            g_b_h: vec![0.0_f32; self.b_h.len()],
            g_ln_out_g: vec![0.0_f32; self.ln_out_g.len()],
            g_ln_out_b: vec![0.0_f32; self.ln_out_b.len()],
            g_w_wm_aux: vec![0.0_f32; self.w_wm_aux.len()],
            g_b_wm_aux: vec![0.0_f32; self.b_wm_aux.len()],
            g_w_a: vec![0.0_f32; self.w_a_heads.len()],
            g_b_a: vec![0.0_f32; self.b_a_heads.len()],
            g_w_d: vec![0.0_f32; self.w_d_heads.len()],
            g_b_d: vec![0.0_f32; self.b_d_heads.len()],
            g_w_v: vec![0.0_f32; self.w_v_heads.len()],
            g_b_v: vec![0.0_f32; self.b_v_heads.len()],
            g_w_s: vec![0.0_f32; self.w_s_heads.len()],
            g_b_s: vec![0.0_f32; self.b_s_heads.len()],
            g_w_c: vec![0.0_f32; self.w_c.len()],
            g_b_c: vec![0.0_f32; self.b_c.len()],
            g_log_cost_sens: 0.0,
            chunks_count: 0,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn accumulate_bptt_chunk_full(
        &self,
        accum: &mut PolicyGradAccum,
        chunk: &[CausalTrajectoryBar],
        init_wm_z: &[f32; WM_DIM],
        world_model: Option<&PhysicsWorldModel>,
        feat_mean: &[f32],
        feat_var: &[f32],
        hparams: &AgentHyperParams,
        target_override: Option<(&[Option<f32>], f32)>,
        vehicle_override: Option<(&[Option<f32>], f32)>,
        accumulate_fisher_into: Option<&mut [f32]>,
    ) -> (f64, f32, f32) {
        self.accumulate_bptt_chunk_with_held_context(
            accum,
            chunk,
            init_wm_z,
            world_model,
            feat_mean,
            feat_var,
            hparams,
            target_override,
            vehicle_override,
            None,
            accumulate_fisher_into,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn accumulate_bptt_chunk_with_held_context(
        &self,
        accum: &mut PolicyGradAccum,
        chunk: &[CausalTrajectoryBar],
        init_wm_z: &[f32; WM_DIM],
        world_model: Option<&PhysicsWorldModel>,
        feat_mean: &[f32],
        feat_var: &[f32],
        hparams: &AgentHyperParams,
        target_override: Option<(&[Option<f32>], f32)>,
        vehicle_override: Option<(&[Option<f32>], f32)>,
        forced_held: Option<ForcedHeldContext>,
        accumulate_fisher_into: Option<&mut [f32]>,
    ) -> (f64, f32, f32) {
        let t_len = chunk.len();
        if t_len < 3 {
            return (0.0, 0.0, 0.0);
        }
        let is_fisher_pass = accumulate_fisher_into.is_some();
        let prior_snapshot = if is_fisher_pass && accum.chunks_count > 0 {
            Some(accum.flatten())
        } else {
            None
        };
        let d = HIDDEN_DIM;
        let hd = HEAD_IN_DIM;

        #[derive(Clone)]
        struct StepCache {
            x_norm: [f32; RL_STATE_DIM],
            prev_wm_z: [f32; WM_DIM],
            has_ext_wm: bool,
            x_cat: [f32; TOTAL_INPUT_DIM],
            u_norm: [f32; HIDDEN_DIM],
            u_inv_std: f32,
            u_lin_pre: [f32; HIDDEN_DIM],
            u_t: [f32; HIDDEN_DIM],
            prev_h: [f32; HIDDEN_DIM],
            g_t: [f32; HIDDEN_DIM],
            r_t: [f32; HIDDEN_DIM],
            h_cand: [f32; HIDDEN_DIM],
            h_norm: [f32; HIDDEN_DIM],
            h_ln: [f32; HIDDEN_DIM],
            h_inv_std: f32,
            wm_aux_pred: [f32; WM_OUT_DIM],
            head_in: [f32; HEAD_IN_DIM],
            z_k: [f32; SPECIALIST_K],
            tau_k: [f32; SPECIALIST_K],
            dtau_ddpre: [f32; SPECIALIST_K],
            dtau_dlogk: [f32; SPECIALIST_K],
            a_k: [f32; SPECIALIST_K],
            s_k: [f32; SPECIALIST_K],
            a_t: f32,
            v_t: f32,
            q_all: [f32; CRITIC_OUT_DIM],
            r_liq: f32,
            r_eff: f32,
            r_shares: f32,
            r_option_dir: f32,
            opt_theta: f32,
            theta_cost: f32,
            c_eff: f32,
            eq_curve: f32,
            dd_peak: f32,
            ema_turn: f32,
            force_flat: bool,
            entry_allowed: bool,
            option_allowed: bool,
            prev_a: f32,
        }

        let mut caches = Vec::with_capacity(t_len);
        let mut prev_h = [0.0_f32; HIDDEN_DIM];
        let mut prev_wm_z = *init_wm_z;
        let mut prev_a = 0.0_f32;
        let mut prev_v = 0.0_f32;
        let mut prev_liq_res = 0.0_f32;
        let mut ema_turn = 0.0_f32;
        let mut eq_cur = 0.0_f32;
        let mut bars_held = 0usize;
        let mut cum_unit_gross = 0.0_f32;
        let mut cum_ret = 0.0_f32;
        let mut peak_ret = 0.0_f32;

        let mut fh_bars_held = 0usize;
        let mut fh_cum_gross = 0.0_f32;
        let mut fh_cum_ret = 0.0_f32;
        let mut fh_peak_ret = 0.0_f32;

        for (t_idx, bar) in chunk.iter().enumerate() {
            let step_prev_wm_z = prev_wm_z;
            let x_norm = normalize_212_with_stats(&bar.x_212, feat_mean, feat_var);
            let (z_wm, wm_pred) = if let Some(wm) = world_model {
                let out = wm.forward_step(&x_norm, &step_prev_wm_z);
                prev_wm_z = out.0;
                out
            } else {
                ([0.0_f32; WM_DIM], [0.0_f32; WM_OUT_DIM])
            };

            let (eff_prev_a, eff_cum_ret, eff_peak_ret, dd_peak, is_holding, active_one_way_bps, eff_log_bars) =
                if let Some(fh) = forced_held {
                    if t_idx >= fh.entry_offset && !bar.force_flat {
                        let fh_dd = (fh_peak_ret - fh_cum_ret).max(0.0);
                        let c_fh = if fh.is_option_routed {
                            (bar.one_way_cost_option * 10_000.0).max(1.0)
                        } else {
                            (bar.one_way_cost_shares * 10_000.0).max(1.0)
                        };
                        (
                            fh.direction_sign * 0.72,
                            fh_cum_ret,
                            fh_peak_ret,
                            fh_dd,
                            true,
                            c_fh,
                            (1.0 + fh_bars_held.max(1) as f32).ln(),
                        )
                    } else {
                        let ddp = (peak_ret - cum_ret).max(0.0);
                        let ih = prev_a.abs() > 1e-4;
                        let c_bps = if ih {
                            let eff_c_pos = (1.0 - prev_v) * bar.one_way_cost_shares
                                + prev_v * bar.one_way_cost_option;
                            (eff_c_pos * 10_000.0).max(1.0)
                        } else {
                            bar.one_way_cost_bps
                        };
                        (
                            prev_a,
                            cum_ret,
                            peak_ret,
                            ddp,
                            ih,
                            c_bps,
                            if ih { (1.0 + bars_held as f32).ln() } else { 0.0 },
                        )
                    }
                } else {
                    let ddp = (peak_ret - cum_ret).max(0.0);
                    let ih = prev_a.abs() > 1e-4;
                    let c_bps = if ih {
                        let eff_c_pos = (1.0 - prev_v) * bar.one_way_cost_shares
                            + prev_v * bar.one_way_cost_option;
                        (eff_c_pos * 10_000.0).max(1.0)
                    } else {
                        bar.one_way_cost_bps
                    };
                    (
                        prev_a,
                        cum_ret,
                        peak_ret,
                        ddp,
                        ih,
                        c_bps,
                        if ih { (1.0 + bars_held as f32).ln() } else { 0.0 },
                    )
                };

            let self_s = SelfAwarenessState {
                current_pos: eff_prev_a,
                unrealized_net_ret: eff_cum_ret,
                peak_excursion: if is_holding { eff_peak_ret.max(0.0) } else { 0.0 },
                drawdown_from_peak: if is_holding { dd_peak } else { 0.0 },
                log_bars_held: eff_log_bars,
                one_way_cost_bps: active_one_way_bps,
                session_time_left: bar.session_time_left,
                ema_turnover: ema_turn,
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
            let mut u_lin_pre = [0.0_f32; HIDDEN_DIM];
            let mut u_t = [0.0_f32; HIDDEN_DIM];
            for i in 0..d {
                u_norm[i] = (u_raw[i] - u_mean) * u_inv_std;
                let pre = u_norm[i] * self.ln_in_g[i] + self.ln_in_b[i];
                u_lin_pre[i] = pre;
                u_t[i] = nonsat_input_act(pre).0;
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
            let mut h_ln = [0.0_f32; HIDDEN_DIM];
            for i in 0..d {
                h_norm[i] = (h_next[i] - h_mean) * h_inv_std;
                h_ln[i] = h_norm[i] * self.ln_out_g[i] + self.ln_out_b[i];
            }
            let wm_aux_pred = self.eval_aux_wm(&h_ln);
            let has_ext_wm =
                z_wm.iter().any(|v| v.abs() > 1e-8) || wm_pred.iter().any(|v| v.abs() > 1e-8);
            let mut wm_fused = [0.0_f32; WM_OUT_DIM];
            for k in 0..WM_OUT_DIM {
                wm_fused[k] = if has_ext_wm {
                    0.55 * wm_pred[k] + 0.45 * wm_aux_pred[k]
                } else {
                    wm_aux_pred[k]
                };
            }
            let head_in = fill_head_input(&h_ln, &wm_fused, &self_s);

            let (q_all, channel_u, channel_lcb) =
                self.eval_critic_lcb_channels(&head_in, hparams.downside_aversion);
            let best_long_lcb = if bar.option_allowed {
                channel_lcb[0].max(channel_lcb[2])
            } else {
                channel_lcb[0]
            };
            let best_short_lcb = if bar.option_allowed {
                channel_lcb[1].max(channel_lcb[3])
            } else {
                channel_lcb[1]
            };
            let rt_cost_bps = (2.0 * active_one_way_bps).max(4.0);
            let wm_lcb_long =
                wm_directional_lcb_bps(&wm_fused, 1.0, rt_cost_bps, hparams.downside_aversion);
            let wm_lcb_short =
                wm_directional_lcb_bps(&wm_fused, -1.0, rt_cost_bps, hparams.downside_aversion);

            let mut z_k = [0.0_f32; SPECIALIST_K];
            let mut tau_k = [0.0_f32; SPECIALIST_K];
            let mut dtau_ddpre = [0.0_f32; SPECIALIST_K];
            let mut dtau_dlogk = [0.0_f32; SPECIALIST_K];
            let mut a_k = [0.0_f32; SPECIALIST_K];
            let mut s_k = [0.0_f32; SPECIALIST_K];
            let mut a_mean = 0.0_f32;
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
                let (wm_dir_lcb_bps, dir_critic_lcb_bps) = if is_holding {
                    if eff_prev_a >= 0.0 {
                        (
                            wm_lcb_long,
                            best_long_lcb * 10_000.0 - self_s.drawdown_from_peak * 1_250.0,
                        )
                    } else {
                        (
                            wm_lcb_short,
                            best_short_lcb * 10_000.0 - self_s.drawdown_from_peak * 1_250.0,
                        )
                    }
                } else if z_k[k] >= 0.0 {
                    (
                        wm_lcb_long - hparams.conformal_q_bps,
                        best_long_lcb * 10_000.0,
                    )
                } else {
                    (
                        wm_lcb_short - hparams.conformal_q_bps,
                        best_short_lcb * 10_000.0,
                    )
                };
                let (tk, dtk, dtk_logk) = compute_cost_scaled_deadzone(
                    d_pre,
                    active_one_way_bps,
                    hparams.cost_sensitivity,
                    wm_dir_lcb_bps,
                    dir_critic_lcb_bps,
                    is_holding,
                    self_s.session_time_left,
                    self_s.ema_turnover,
                    self_s.peak_excursion,
                    self_s.drawdown_from_peak,
                    self_s.log_bars_held,
                );
                tau_k[k] = tk;
                dtau_ddpre[k] = dtk;
                dtau_dlogk[k] = dtk_logk;

                // Hard-enforce `a_{T_eod} = 0.0` on `force_flat`, and exact `0.0` inside deadzone
                a_k[k] = if bar.force_flat || (!bar.entry_allowed && !is_holding) || z_k[k].abs() <= tk {
                    0.0
                } else {
                    let raw_ak =
                        (z_k[k].signum() * (z_k[k].abs() - tk) / (1.0 - tk).max(0.15)).clamp(-1.0, 1.0);
                    if !bar.entry_allowed {
                        if raw_ak * eff_prev_a <= 0.0 {
                            0.0
                        } else {
                            raw_ak.signum() * raw_ak.abs().min(eff_prev_a.abs())
                        }
                    } else {
                        raw_ak
                    }
                };
                let opt_adv_dir = if a_k[k] >= 0.0 {
                    channel_u[2] - channel_u[0]
                } else {
                    channel_u[3] - channel_u[1]
                };
                let vk = if bar.option_allowed {
                    sigmoid(v_pre + (opt_adv_dir * 16.0).clamp(-1.1, 1.1))
                } else {
                    0.0
                };
                s_k[k] = (softplus(s_pre) * 0.036 + 0.011).clamp(0.010, 0.068);
                a_mean += w_k * a_k[k];
                v_mean += w_k * vk;
            }
            if !bar.force_flat && !is_holding {
                let mut a_var = 0.0_f32;
                for k in 0..SPECIALIST_K {
                    let diff = a_k[k] - a_mean;
                    a_var += self.spec_weights[k] * diff * diff;
                }
                let a_std = a_var.max(0.0).sqrt();
                let action_mag = (a_mean.abs() - 0.16 * a_std).max(0.0);
                a_mean = if action_mag <= 1e-6 {
                    0.0
                } else {
                    (a_mean.signum() * action_mag).clamp(-1.0, 1.0)
                };
            }

            // Physical Mark-to-Liquidation step reward (`V_liq(t) - V_liq(t-1)`):
            // Books round-trip entry + liquidation friction immediately at initiation so exiting
            // `a_t -> 0` creates zero artificial step-cost barrier inside BPTT.
            let r_shares = bar.ret_oo_shares;
            let (r_option_dir, opt_theta) = if bar.option_gamma_boost > 1e-6 {
                (
                    bar.ret_oo_shares * bar.option_gamma_boost,
                    bar.option_theta_per_bar.max(0.0),
                )
            } else {
                (bar.ret_oo_option, 0.0)
            };
            let r_eff = (1.0 - v_mean) * r_shares + v_mean * r_option_dir;
            let theta_cost = v_mean * opt_theta;
            let c_eff =
                (1.0 - v_mean) * bar.one_way_cost_shares + v_mean * bar.one_way_cost_option;
            let r_gross = a_mean * r_eff - a_mean.abs() * theta_cost;
            let r_mtm = r_gross - (a_mean - prev_a).abs() * c_eff;
            let cur_liq_res = if bar.force_flat {
                0.0
            } else {
                a_mean.abs() * c_eff
            };
            let r_liq = r_mtm - (cur_liq_res - prev_liq_res);
            eq_cur += r_liq;

            let cur_prev_a = eff_prev_a;
            if a_mean.abs() < 1e-4 {
                bars_held = 0;
                cum_unit_gross = 0.0;
                cum_ret = 0.0;
                peak_ret = 0.0;
            } else if a_mean * prev_a > 0.0 {
                bars_held += 1;
                let step_unit = a_mean.signum() * r_eff - theta_cost;
                cum_unit_gross += step_unit;
                cum_ret = cum_unit_gross - 2.0 * c_eff;
                peak_ret = peak_ret.max(cum_ret);
            } else {
                bars_held = 1;
                let step_unit = a_mean.signum() * r_eff - theta_cost;
                cum_unit_gross = step_unit;
                cum_ret = cum_unit_gross - 2.0 * c_eff;
                peak_ret = (-2.0 * c_eff).max(cum_ret);
            }

            if let Some(fh) = forced_held
                && t_idx >= fh.entry_offset
                && !bar.force_flat
            {
                let prev_fh_bars = fh_bars_held;
                fh_bars_held += 1;
                let (fh_step_r, fh_c) = if fh.is_option_routed {
                    (
                        fh.direction_sign * r_option_dir - opt_theta,
                        bar.one_way_cost_option,
                    )
                } else {
                    (fh.direction_sign * r_shares, bar.one_way_cost_shares)
                };
                fh_cum_gross += fh_step_r;
                fh_cum_ret = fh_cum_gross - 2.0 * fh_c;
                if prev_fh_bars == 0 {
                    fh_peak_ret = (-2.0 * fh_c).max(fh_cum_ret);
                } else {
                    fh_peak_ret = fh_peak_ret.max(fh_cum_ret);
                }
            }

            let step_ema_turn = ema_turn;
            ema_turn = (0.90 * ema_turn + 0.10 * (a_mean - prev_a).abs()).clamp(0.0, 2.0);

            caches.push(StepCache {
                x_norm,
                prev_wm_z: step_prev_wm_z,
                has_ext_wm,
                x_cat,
                u_norm,
                u_inv_std,
                u_lin_pre,
                u_t,
                prev_h,
                g_t,
                r_t,
                h_cand,
                h_norm,
                h_ln,
                h_inv_std,
                wm_aux_pred,
                head_in,
                z_k,
                tau_k,
                dtau_ddpre,
                dtau_dlogk,
                a_k,
                s_k,
                a_t: a_mean,
                v_t: v_mean,
                q_all,
                r_liq,
                r_eff,
                r_shares,
                r_option_dir,
                opt_theta,
                theta_cost,
                c_eff,
                eq_curve: eq_cur,
                dd_peak,
                ema_turn: step_ema_turn,
                force_flat: bar.force_flat,
                entry_allowed: bar.entry_allowed,
                option_allowed: bar.option_allowed,
                prev_a: cur_prev_a,
            });
            prev_h = h_next;
            prev_a = a_mean;
            prev_v = v_mean;
            prev_liq_res = cur_liq_res;
        }

        // Separate discounted causal long/short continuation trajectories (theta strictly negative on BOTH long & short!)
        // combined with strictly episode-bounded `[1, 3, 8, 16]`-bar net-of-cost counterfactual targets.
        let gamma = hparams.horizon_gamma.clamp(0.70, 0.96);
        let ep_cf = compute_episode_counterfactual_targets(chunk, gamma);
        let mut cont_long_net = vec![0.0_f32; t_len];
        let mut cont_short_net = vec![0.0_f32; t_len];
        let mut run_long = 0.0_f32;
        let mut run_short = 0.0_f32;
        for t in (0..t_len).rev() {
            if caches[t].force_flat {
                run_long = 0.0;
                run_short = 0.0;
                cont_long_net[t] = 0.0;
                cont_short_net[t] = 0.0;
            } else {
                let step_l = caches[t].r_eff - caches[t].theta_cost;
                let step_s = -caches[t].r_eff - caches[t].theta_cost;
                run_long = step_l + gamma * run_long;
                run_short = step_s + gamma * run_short;
                cont_long_net[t] = run_long;
                cont_short_net[t] = run_short;
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
        let sum_r_liq: f32 = caches.iter().map(|c| c.r_liq).sum();
        let sum_down: f32 = caches
            .iter()
            .map(|c| {
                let d_neg = (-c.r_liq).max(0.0);
                d_neg * d_neg
            })
            .sum();
        let sum_l1: f32 = caches.iter().map(|c| c.a_t.abs()).sum();
        let sum_turn_drag: f32 = caches
            .iter()
            .map(|c| {
                if c.force_flat {
                    0.0
                } else {
                    c.ema_turn * (c.a_t - c.prev_a).abs() * (2.0 * c.c_eff)
                }
            })
            .sum();
        let utility = inv_t * sum_r_liq
            - 0.45 * inv_t * sum_turn_drag
            - hparams.downside_aversion * inv_t * sum_down
            - hparams.dd_aversion * max_dd
            - hparams.l1_sparsity * inv_t * sum_l1;

        let g_w_in = &mut accum.g_w_in;
        let g_b_in = &mut accum.g_b_in;
        let g_ln_in_g = &mut accum.g_ln_in_g;
        let g_ln_in_b = &mut accum.g_ln_in_b;
        let g_w_g = &mut accum.g_w_g;
        let g_u_g = &mut accum.g_u_g;
        let g_b_g = &mut accum.g_b_g;
        let g_w_r = &mut accum.g_w_r;
        let g_u_r = &mut accum.g_u_r;
        let g_b_r = &mut accum.g_b_r;
        let g_w_h = &mut accum.g_w_h;
        let g_u_h = &mut accum.g_u_h;
        let g_b_h = &mut accum.g_b_h;
        let g_ln_out_g = &mut accum.g_ln_out_g;
        let g_ln_out_b = &mut accum.g_ln_out_b;
        let g_w_wm_aux = &mut accum.g_w_wm_aux;
        let g_b_wm_aux = &mut accum.g_b_wm_aux;
        let g_w_a = &mut accum.g_w_a;
        let g_b_a = &mut accum.g_b_a;
        let g_w_d = &mut accum.g_w_d;
        let g_b_d = &mut accum.g_b_d;
        let g_w_v = &mut accum.g_w_v;
        let g_b_v = &mut accum.g_b_v;
        let g_w_s = &mut accum.g_w_s;
        let g_b_s = &mut accum.g_b_s;
        let g_w_c = &mut accum.g_w_c;
        let g_b_c = &mut accum.g_b_c;
        let g_log_cost_sens = &mut accum.g_log_cost_sens;

        let util_scale = 180.0_f32;
        let mut dh_next = [0.0_f32; HIDDEN_DIM];
        let mut dz_wm_next = [0.0_f32; WM_DIM];
        let mut total_sal_dx = 0.0_f32;
        let mut total_sal_dh = 0.0_f32;
        let cs_unbounded = hparams.cost_sensitivity.max(0.02);
        let has_eod_flatten = chunk.iter().any(|b| b.force_flat);

        for t in (0..t_len).rev() {
            let c = &caches[t];
            let mut grad_dr = inv_t * util_scale;
            if c.r_liq < 0.0 {
                grad_dr += -hparams.downside_aversion * inv_t * (2.0 * c.r_liq) * util_scale;
            }
            if max_dd > 1e-6 && t > max_dd_peak_t && t <= max_dd_t {
                let denom = (max_dd_t - max_dd_peak_t).max(1) as f32;
                grad_dr += (hparams.dd_aversion / denom) * util_scale;
            }

            let rt_cost = 2.0 * c.c_eff;
            let cf_tgt = ep_cf.critic_targets[t];
            let cf_hld = ep_cf.hold_cont_channels[t];
            let cf_long_entry = cf_tgt[0].max(if c.option_allowed {
                cf_tgt[2]
            } else {
                f32::NEG_INFINITY
            });
            let cf_short_entry = cf_tgt[1].max(if c.option_allowed {
                cf_tgt[3]
            } else {
                f32::NEG_INFINITY
            });
            let cf_long_blend = (1.0 - c.v_t) * cf_tgt[0]
                + if c.option_allowed {
                    c.v_t * cf_tgt[2]
                } else {
                    0.0
                };
            let cf_short_blend = (1.0 - c.v_t) * cf_tgt[1]
                + if c.option_allowed {
                    c.v_t * cf_tgt[3]
                } else {
                    0.0
                };
            let net_long_entry = if chunk[t].cf_valid {
                0.72 * cf_long_blend + 0.28 * (cont_long_net[t] - rt_cost)
            } else {
                0.72 * cf_long_entry + 0.28 * (cont_long_net[t] - rt_cost)
            };
            let net_short_entry = if chunk[t].cf_valid {
                0.72 * cf_short_blend + 0.28 * (cont_short_net[t] - rt_cost)
            } else {
                0.72 * cf_short_entry + 0.28 * (cont_short_net[t] - rt_cost)
            };
            let eff_hold_long = 0.58 * ((1.0 - c.v_t) * cf_hld[0] + c.v_t * cf_hld[2])
                + 0.24 * ((1.0 - c.v_t) * ep_cf.hold_to_eod[t][0] + c.v_t * ep_cf.hold_to_eod[t][2])
                + 0.18 * cont_long_net[t];
            let eff_hold_short = 0.58 * ((1.0 - c.v_t) * cf_hld[1] + c.v_t * cf_hld[3])
                + 0.24 * ((1.0 - c.v_t) * ep_cf.hold_to_eod[t][1] + c.v_t * ep_cf.hold_to_eod[t][3])
                + 0.18 * cont_short_net[t];
            let best_cf_adv = if c.force_flat {
                0.0
            } else {
                ep_cf.best_entry_net[t].max(net_long_entry.max(net_short_entry))
            };
            let turn_clamped = c.ema_turn.clamp(0.0, 1.5);
            let extra_cushion =
                rt_cost * (0.12 + 0.10 * cs_unbounded.min(4.0) + 0.35 * turn_clamped);

            let fh_decay_regret = if let Some(fh) = forced_held {
                if t >= fh.optimal_exit_offset && !c.force_flat {
                    (fh.peak_giveback_ret.max(0.002) + c.dd_peak).clamp(0.002, 0.08)
                } else {
                    0.0
                }
            } else {
                0.0
            };

            // `du_da` = d(Utility) / d(a_t) under Mark-to-Liquidation equity `V_liq(t)`
            let du_da = if c.force_flat {
                0.0
            } else if let Some((targets, weight)) = target_override {
                if let Some(Some(tgt_a)) = targets.get(t).copied() {
                    let decay_boost = 1.0 + 18.0 * fh_decay_regret;
                    -weight * inv_t * 25.0 * decay_boost * (c.a_t - tgt_a)
                } else {
                    0.0
                }
            } else {
                let hurdle_long = extra_cushion + 0.08 * ep_cf.path_mae[t][0];
                let hurdle_short = extra_cushion + 0.08 * ep_cf.path_mae[t][1];
                let long_excess = net_long_entry - hurdle_long;
                let short_excess = net_short_entry - hurdle_short;
                let dr_da = if c.prev_a.abs() <= 1e-4 {
                    if !c.entry_allowed {
                        -c.a_t * rt_cost * 4.0
                    } else if long_excess > 0.0 && long_excess >= short_excess {
                        long_excess
                    } else if short_excess > 0.0 && short_excess > long_excess {
                        -short_excess
                    } else if c.a_t.abs() > 1e-5 {
                        let shortfall = (-long_excess.max(short_excess)).max(0.0);
                        let tx_scale = (1.5 + 0.10 * (t_len as f32)).min(4.5);
                        -c.a_t * shortfall * tx_scale
                    } else {
                        0.0
                    }
                } else {
                    let pos_sgn = c.prev_a.signum();
                    let same_cont = if pos_sgn > 0.0 {
                        eff_hold_long
                    } else {
                        eff_hold_short
                    };
                    let (opp_entry, opp_hurdle) = if pos_sgn > 0.0 {
                        (net_short_entry, hurdle_short)
                    } else {
                        (net_long_entry, hurdle_long)
                    };
                    let sess_rem = chunk[t].session_time_left.clamp(0.0, 1.0);
                    let sess_elapsed = (1.0 - sess_rem).clamp(0.0, 1.0);
                    let eod_press = if has_eod_flatten {
                        ((0.34 - sess_rem).max(0.0) / 0.34).powi(2) * (0.85 * rt_cost + 0.0014)
                    } else {
                        0.0
                    };
                    let dd_mult = if has_eod_flatten {
                        0.62 + 0.68 * sess_elapsed
                    } else {
                        0.42
                    };
                    let carry_hurdle = (0.14 + 0.48 * sess_elapsed * sess_elapsed)
                        * c.c_eff
                        * (1.0 + 0.25 * hparams.downside_aversion)
                        + dd_mult * c.dd_peak
                        + eod_press;
                    // Opportunity-Adjusted `Q_hold(x_t, s_t) - Q_flat(x_t, s_t)`:
                    // Going flat forfeits the already-paid-for position and incurs expected future re-entry friction
                    // ONLY when new entries remain allowed and sufficient session remains.
                    let reentry_prob = if !c.entry_allowed {
                        0.0
                    } else {
                        ((0.22 + 0.45 * turn_clamped.min(1.0)) * (0.20 + 0.80 * sess_rem))
                            .clamp(0.05, 0.75)
                    };
                    let q_flat_val = -0.35 * reentry_prob * rt_cost.min(0.0026);
                    let q_hold_minus_flat = (same_cont - carry_hurdle) - q_flat_val;
                    if c.a_t * pos_sgn < -1e-5 {
                        let opp_excess = opp_entry - opp_hurdle;
                        if c.entry_allowed && opp_excess > 0.0 {
                            -pos_sgn * opp_excess
                        } else {
                            let shortfall = (-opp_excess).max(0.0);
                            -c.a_t * shortfall * 3.0
                        }
                    } else if c.a_t.abs() <= 1e-5 {
                        if q_hold_minus_flat > 0.0 {
                            pos_sgn * q_hold_minus_flat
                        } else if c.entry_allowed && opp_entry > opp_hurdle {
                            -pos_sgn * (opp_entry - opp_hurdle)
                        } else {
                            0.0
                        }
                    } else {
                        pos_sgn * q_hold_minus_flat
                    }
                };
                let turn_step_pen = if !c.force_flat && (c.a_t - c.prev_a).abs() > 1e-5 {
                    0.18 * inv_t * turn_clamped * (c.a_t - c.prev_a).signum() * rt_cost
                } else {
                    0.0
                };
                let g_l1 = if c.a_t.abs() > 1e-5 {
                    hparams.l1_sparsity * inv_t * c.a_t.signum()
                } else {
                    0.0
                };
                grad_dr * dr_da - util_scale * (g_l1 + turn_step_pen)
            };

            let du_dv = if c.option_allowed {
                if let Some((v_tgts, v_weight)) = vehicle_override {
                    if let Some(Some(tgt_v)) = v_tgts.get(t).copied() {
                        -v_weight * inv_t * 36.0 * (c.v_t - tgt_v) * c.v_t * (1.0 - c.v_t)
                    } else {
                        0.0
                    }
                } else if !is_fisher_pass {
                    let opt_ret_diff = (c.r_option_dir - c.r_shares) * c.a_t
                        - c.opt_theta * c.a_t.abs();
                    let opt_cost_diff = 2.0
                        * (chunk[t].one_way_cost_option - chunk[t].one_way_cost_shares)
                        * (c.a_t - c.prev_a).abs().max(0.15 * c.a_t.abs());
                    let choose_long = if c.a_t.abs() > 1e-4 {
                        c.a_t >= 0.0
                    } else {
                        cf_tgt[0].max(cf_tgt[2]) >= cf_tgt[1].max(cf_tgt[3])
                    };
                    let cf_v_adv = if choose_long {
                        (cf_tgt[2] - cf_tgt[0]) * c.a_t.abs().max(0.20)
                    } else {
                        (cf_tgt[3] - cf_tgt[1]) * c.a_t.abs().max(0.20)
                    };
                    let opt_var_pen = 2.5
                        * hparams.downside_aversion
                        * (c.r_option_dir * c.r_option_dir - c.r_shares * c.r_shares).max(0.0)
                        * c.a_t.abs();
                    let opt_adv =
                        0.65 * cf_v_adv + 0.35 * (opt_ret_diff - opt_cost_diff) - opt_var_pen;
                    let v_scale = if target_override.is_some() { 0.35 } else { 1.0 };
                    v_scale * grad_dr * opt_adv * c.v_t * (1.0 - c.v_t)
                } else {
                    0.0
                }
            } else {
                0.0
            };

            let mut d_head_in = [0.0_f32; HEAD_IN_DIM];
            if !is_fisher_pass {
                let critic_scale = if target_override.is_some() || vehicle_override.is_some() {
                    0.35_f32
                } else {
                    0.45_f32
                };
                for ch in 0..CRITIC_CHANNELS {
                    let target_r = cf_tgt[ch];
                    let base = ch * CRITIC_QUANTILES;
                    for q in 0..CRITIC_QUANTILES {
                        let idx = base + q;
                        let err = target_r - c.q_all[idx];
                        let tau = QUANTILE_TAUS[q];
                        let huber_step = (err / 0.0025).clamp(-1.0, 1.0);
                        let asym_w = if err >= 0.0 { tau } else { 1.0 - tau };
                        let dl_dq = -asym_w * huber_step * inv_t * critic_scale;
                        let dl_ds = dl_dq * RET_LOGIT_UNIT_SCALE;
                        g_b_c[idx] += dl_ds;
                        for i in 0..hd {
                            g_w_c[idx * hd + i] += dl_ds * c.head_in[i];
                            d_head_in[i] += dl_ds * self.w_c[idx * hd + i];
                        }
                    }
                }
            }

            let ov_weight_clamped = target_override
                .map(|(_, w)| w.clamp(0.5, 8.0))
                .unwrap_or(1.0);

            for k in 0..SPECIALIST_K {
                let w_k = self.spec_weights[k];
                let off_k = k * hd;
                let g_disagree = if target_override.is_none() {
                    -2.0 * hparams.disagree_aversion * inv_t * util_scale * (c.a_k[k] - c.a_t)
                } else {
                    0.0
                };
                let du_dak = w_k * (du_da + g_disagree);

                let tgt_val =
                    target_override.and_then(|(tgts, _)| tgts.get(t).copied().flatten());
                let is_zero_override = tgt_val.is_some_and(|v| v.abs() <= 1e-4);
                let has_nonzero_override = tgt_val.is_some_and(|v| v.abs() > 1e-4);

                let (du_dzk, du_dtauk) = if c.force_flat {
                    (0.0, 0.0)
                } else {
                    let denom = (1.0 - c.tau_k[k]).max(0.15);
                    let boundary_temp = if has_nonzero_override { 0.55 } else { 0.28 };
                    let boundary_sech = sech2((c.z_k[k].abs() - c.tau_k[k]) / boundary_temp);

                    if c.z_k[k].abs() > c.tau_k[k] && c.a_k[k].abs() > 1e-6 {
                        let da_dz = 1.0 / denom;
                        let sgn = c.z_k[k].signum();
                        let da_dtau = -sgn * (1.0 - c.z_k[k].abs()) / (denom * denom);
                        let extra_widen = if is_zero_override {
                            0.12 * inv_t * ov_weight_clamped * (1.0 + 12.0 * fh_decay_regret)
                        } else {
                            0.0
                        };
                        (du_dak * da_dz, du_dak * da_dtau + extra_widen)
                    } else {
                        let is_holding_step = c.prev_a.abs() > 1e-4;
                        let flat_hurdle = (extra_cushion + 0.12 * rt_cost).max(0.25 * rt_cost);
                        if is_zero_override {
                            // Localized boundary + deadzone widening on FrictionInversion / HoldingEodDecay / FlatCounterfactual:
                            // `du_dtauk > 0` increases `d_pre` -> widens `tau_k` without injecting any constant leaky floor!
                            let decay_mult = 1.0 + 14.0 * fh_decay_regret;
                            let dz_damp =
                                -c.z_k[k] * (0.20 + 0.45 * boundary_sech) * inv_t * ov_weight_clamped * decay_mult;
                            let dtau_widen =
                                (0.08 + 0.25 * boundary_sech) * inv_t * ov_weight_clamped * decay_mult;
                            (dz_damp, dtau_widen)
                        } else {
                            let hold_wants_resume =
                                is_holding_step && du_dak * c.prev_a.signum() > 0.0;
                            let viable_opportunity = has_nonzero_override
                                || hold_wants_resume
                                || (!c.force_flat && c.entry_allowed && best_cf_adv > flat_hurdle);
                            if viable_opportunity && du_dak.abs() > 1e-6 {
                                // Pure `sech^2((|z| - tau) / temp)` boundary surrogate (ZERO constant leaky floor!).
                                let surr_dz = (boundary_sech
                                    * (1.0 + 0.25 * hparams.explore_pressure))
                                    .min(1.15);
                                let dtau_narrow = -du_dak.abs() * 0.42 * boundary_sech;
                                (du_dak * surr_dz, dtau_narrow)
                            } else if is_holding_step
                                && du_dak * c.prev_a.signum() < 0.0
                                && c.z_k[k] * c.prev_a > 0.0
                            {
                                (
                                    -c.z_k[k] * 0.18 * boundary_sech * inv_t,
                                    du_dak.abs() * 0.45 * boundary_sech,
                                )
                            } else if !c.entry_allowed || best_cf_adv < 0.0 {
                                // Negative-expectancy noise inside deadzone: damp non-zero `z_k` near boundary
                                // (`0.0` at `z_k == 0`) and gently widen `tau_k`.
                                (
                                    -c.z_k[k] * boundary_sech * 0.22 * inv_t,
                                    0.06 * boundary_sech * inv_t,
                                )
                            } else {
                                (0.0, 0.0)
                            }
                        }
                    }
                };

                let du_dzpre = du_dzk * (1.0 - c.z_k[k] * c.z_k[k]);
                let du_ddpre = du_dtauk * c.dtau_ddpre[k];
                let du_dvpre = w_k * du_dv;
                if !is_fisher_pass {
                    *g_log_cost_sens += (-du_dtauk) * c.dtau_dlogk[k];
                }

                let dir_mae = if c.a_t >= 0.0 {
                    ep_cf.path_mae[t][0]
                } else {
                    ep_cf.path_mae[t][1]
                };
                let target_stop =
                    (1.95 * dir_mae + 3.0 * chunk[t].one_way_cost_shares).clamp(0.012, 0.062);
                let dl_ds = if !is_fisher_pass {
                    (c.s_k[k] - target_stop) * inv_t * 0.16
                } else {
                    0.0
                };
                let sp_val = ((c.s_k[k] - 0.011) / 0.036).max(1e-4);
                let d_sp = (1.0 - (-sp_val).exp()).clamp(0.01, 1.0);
                let dl_dspre = dl_ds * 0.036 * d_sp;

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

            let mut d_h_ln_wm = [0.0_f32; HIDDEN_DIM];
            let mut dwm_pred_in = [0.0_f32; WM_OUT_DIM];
            if self.w_wm_aux.len() >= WM_OUT_DIM * d && self.b_wm_aux.len() >= WM_OUT_DIM {
                // 1. Chain-rule gradient from policy/critic heads through `head_in[d .. d + WM_OUT_DIM]` (`wm_fused`)
                // Stop-gradient on `w_wm_aux`/`b_wm_aux` from actor/critic heads (prevent wireheading);
                // propagate only damped recurrent saliency back through `d_h_ln_wm`.
                for m in 0..WM_OUT_DIM {
                    let scale_in = if m % 4 < 3 { 40.0 } else { 0.50 };
                    let dfused = d_head_in[d + m] * scale_in;
                    let w_aux_share = if c.has_ext_wm { 0.45 } else { 1.0 };
                    let w_ext_share = if c.has_ext_wm { 0.55 } else { 0.0 };
                    dwm_pred_in[m] += dfused * w_ext_share;
                    let unit_scale = if m % 4 < 3 { RET_LOGIT_UNIT_SCALE } else { 1.0 };
                    let dl_ds_skip = dfused * w_aux_share * unit_scale;
                    let row_w = &self.w_wm_aux[m * d..(m + 1) * d];
                    for i in 0..d {
                        d_h_ln_wm[i] += 0.25 * dl_ds_skip * row_w[i];
                    }
                }
                // 2. Joint Self-Supervised Auxiliary World-Model Pinball Loss -> `h_ln` -> `w_in` -> `x_t`
                if !is_fisher_pass {
                    let aux_w = if target_override.is_some() || vehicle_override.is_some() {
                        0.25_f32
                    } else {
                        0.35_f32
                    };
                    for m in 0..WM_OUT_DIM {
                        if !chunk[t].wm_valid[m] {
                            continue;
                        }
                        let err = chunk[t].wm_targets[m] - c.wm_aux_pred[m];
                        let kappa = if m % 4 < 3 { 0.0025 } else { 0.25 };
                        let huber_step = (err / kappa).clamp(-1.0, 1.0);
                        let q_tau = WM_TAUS[m];
                        let asym_w = if err >= 0.0 { q_tau } else { 1.0 - q_tau };
                        let unit_scale = if m % 4 < 3 { RET_LOGIT_UNIT_SCALE } else { 1.0 };
                        let dl_ds = -asym_w * huber_step * inv_t * aux_w * unit_scale;
                        g_b_wm_aux[m] += dl_ds;
                        let row_w = &self.w_wm_aux[m * d..(m + 1) * d];
                        let row_g = &mut g_w_wm_aux[m * d..(m + 1) * d];
                        for i in 0..d {
                            row_g[i] += dl_ds * c.h_ln[i];
                            d_h_ln_wm[i] += dl_ds * row_w[i];
                        }
                    }
                }
            }

            let mut d_h_norm = [0.0_f32; HIDDEN_DIM];
            for i in 0..d {
                let dhl = d_head_in[i] + d_h_ln_wm[i];
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
                let dpre = du_t[i] * nonsat_input_act(c.u_lin_pre[i]).1;
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
            let mut dz_wm_in = dz_wm_next;
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
                    } else if j < RL_STATE_DIM + WM_DIM {
                        dz_wm_in[j - RL_STATE_DIM] += du_raw * row_w[j];
                    } else if j < RL_STATE_DIM + WM_DIM + WM_OUT_DIM {
                        dwm_pred_in[j - (RL_STATE_DIM + WM_DIM)] += du_raw * row_w[j];
                    }
                }
            }

            // Analytical chain-rule backpropagation through Stage-A `PhysicsWorldModel` (`z_wm`, `wm_pred` -> `x_norm`)
            if let Some(wm) = world_model {
                let (dx_wm, dz_wm_prev) =
                    wm.backprop_step_to_x(&c.x_norm, &c.prev_wm_z, &dz_wm_in, &dwm_pred_in);
                for j in 0..RL_STATE_DIM {
                    dx_sal[j] += dx_wm[j];
                }
                dz_wm_next = dz_wm_prev;
            } else {
                dz_wm_next = [0.0_f32; WM_DIM];
            }

            let step_sal_dx: f32 =
                dx_sal.iter().map(|v| v.abs()).sum::<f32>() / (RL_STATE_DIM as f32);
            total_sal_dx += step_sal_dx;

            dh_next = dh_prev;
        }

        // Each step's loss contribution already carries `inv_t = 1 / T`, so summing `|dL/dx_t|` across
        // `t = 0..T` yields exact time-averaged per-bar Jacobian `E_t[|dl_t/dx_t|]` without O(1/T^2) decay.
        let mean_sal_dx = total_sal_dx;
        let mean_sal_dh = total_sal_dh;
        accum.chunks_count += 1;

        if let Some(fisher_acc) = accumulate_fisher_into {
            let mut idx = 0usize;
            for sl in accum.slices() {
                for &val in sl {
                    let g_chunk = if let Some(ref prior) = prior_snapshot {
                        val - prior[idx]
                    } else {
                        val
                    };
                    if idx < fisher_acc.len() {
                        fisher_acc[idx] += g_chunk * g_chunk;
                    }
                    idx += 1;
                }
            }
        }

        (utility as f64, mean_sal_dx, mean_sal_dh)
    }

    pub fn apply_accumulated_adam(
        &mut self,
        accum: &mut PolicyGradAccum,
        hparams: &AgentHyperParams,
        lr_scale: f32,
        ewc_lambda: f32,
    ) {
        if accum.chunks_count == 0 || lr_scale <= 0.0 {
            accum.zero();
            return;
        }
        self.ensure_adam_buffers();
        let inv_n = 1.0 / (accum.chunks_count as f32);
        let mut sq_norm = 0.0_f32;
        for sl in accum.slices() {
            for &g in sl {
                let gs = g * inv_n;
                sq_norm += gs * gs;
            }
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
                let g = (g_raw * inv_n) * clip_scale + ewc_g;
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

        apply_slice(&mut self.w_in, &accum.g_w_in, wd);
        apply_slice(&mut self.b_in, &accum.g_b_in, 0.0);
        apply_slice(&mut self.ln_in_g, &accum.g_ln_in_g, 0.0);
        apply_slice(&mut self.ln_in_b, &accum.g_ln_in_b, 0.0);
        apply_slice(&mut self.w_g, &accum.g_w_g, wd);
        apply_slice(&mut self.u_g, &accum.g_u_g, wd);
        apply_slice(&mut self.b_g, &accum.g_b_g, 0.0);
        apply_slice(&mut self.w_r, &accum.g_w_r, wd);
        apply_slice(&mut self.u_r, &accum.g_u_r, wd);
        apply_slice(&mut self.b_r, &accum.g_b_r, 0.0);
        apply_slice(&mut self.w_h, &accum.g_w_h, wd);
        apply_slice(&mut self.u_h, &accum.g_u_h, wd);
        apply_slice(&mut self.b_h, &accum.g_b_h, 0.0);
        apply_slice(&mut self.ln_out_g, &accum.g_ln_out_g, 0.0);
        apply_slice(&mut self.ln_out_b, &accum.g_ln_out_b, 0.0);
        apply_slice(&mut self.w_wm_aux, &accum.g_w_wm_aux, wd);
        apply_slice(&mut self.b_wm_aux, &accum.g_b_wm_aux, 0.0);
        apply_slice(&mut self.w_a_heads, &accum.g_w_a, wd);
        apply_slice(&mut self.b_a_heads, &accum.g_b_a, 0.0);
        apply_slice(&mut self.w_d_heads, &accum.g_w_d, wd);
        apply_slice(&mut self.b_d_heads, &accum.g_b_d, 0.0);
        apply_slice(&mut self.w_v_heads, &accum.g_w_v, wd);
        apply_slice(&mut self.b_v_heads, &accum.g_b_v, 0.0);
        apply_slice(&mut self.w_s_heads, &accum.g_w_s, wd);
        apply_slice(&mut self.b_s_heads, &accum.g_b_s, 0.0);
        apply_slice(&mut self.w_c, &accum.g_w_c, wd);
        apply_slice(&mut self.b_c, &accum.g_b_c, 0.0);

        // Zero-rule conditional symmetry regularization: prevent scalar intercepts & unsigned magnitude
        // columns (`session_time_left`, `one_way_cost_bps`, `rc_log`) from memorizing unconditional warmup drift
        // as a permanent `LONG`-only or `SHORT`-only DC offset.
        for b in &mut self.b_a_heads {
            *b = (*b * 0.88).clamp(-0.022, 0.022);
        }
        let d = HIDDEN_DIM;
        let hd = HEAD_IN_DIM;
        let s_off = d + WM_OUT_DIM;
        for k in 0..SPECIALIST_K {
            let off_k = k * hd;
            for h_idx in 0..WM_HORIZONS.len() {
                let rc_col = off_k + d + h_idx * 4 + 3;
                if rc_col < self.w_a_heads.len() {
                    self.w_a_heads[rc_col] = (self.w_a_heads[rc_col] * 0.85).clamp(-0.02, 0.02);
                }
            }
            for u_col in (s_off + 2)..(s_off + SELF_STATE_DIM) {
                let c_idx = off_k + u_col;
                if c_idx < self.w_a_heads.len() {
                    self.w_a_heads[c_idx] = (self.w_a_heads[c_idx] * 0.85).clamp(-0.02, 0.02);
                }
            }
        }
        for h_idx in 0..WM_HORIZONS.len() {
            let base = h_idx * 4;
            if base + 2 < self.b_wm_aux.len() {
                let q50 = (self.b_wm_aux[base + 1] * 0.90).clamp(-0.025, 0.025);
                let half_iqr =
                    (0.5 * (self.b_wm_aux[base + 2] - self.b_wm_aux[base])).clamp(0.008, 0.45);
                self.b_wm_aux[base] = q50 - half_iqr;
                self.b_wm_aux[base + 1] = q50;
                self.b_wm_aux[base + 2] = q50 + half_iqr;
            }
        }
        if self.b_c.len() >= CRITIC_OUT_DIM {
            for q in 0..CRITIC_QUANTILES {
                let sym_sh = 0.5 * (self.b_c[q] + self.b_c[CRITIC_QUANTILES + q]);
                self.b_c[q] = sym_sh;
                self.b_c[CRITIC_QUANTILES + q] = sym_sh;
                let sym_opt = 0.5 * (self.b_c[2 * CRITIC_QUANTILES + q] + self.b_c[3 * CRITIC_QUANTILES + q]);
                self.b_c[2 * CRITIC_QUANTILES + q] = sym_opt;
                self.b_c[3 * CRITIC_QUANTILES + q] = sym_opt;
            }
        }

        accum.zero();
    }

    #[allow(clippy::too_many_arguments)]
    pub fn train_bptt_chunk_full(
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
        vehicle_override: Option<(&[Option<f32>], f32)>,
        accumulate_fisher_into: Option<&mut [f32]>,
    ) -> (f64, f32, f32) {
        self.train_bptt_chunk_with_held_context(
            chunk,
            init_wm_z,
            world_model,
            feat_mean,
            feat_var,
            hparams,
            lr_scale,
            ewc_lambda,
            target_override,
            vehicle_override,
            None,
            accumulate_fisher_into,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn train_bptt_chunk_with_held_context(
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
        vehicle_override: Option<(&[Option<f32>], f32)>,
        forced_held: Option<ForcedHeldContext>,
        accumulate_fisher_into: Option<&mut [f32]>,
    ) -> (f64, f32, f32) {
        let mut accum = self.new_grad_accum();
        let res = self.accumulate_bptt_chunk_with_held_context(
            &mut accum,
            chunk,
            init_wm_z,
            world_model,
            feat_mean,
            feat_var,
            hparams,
            target_override,
            vehicle_override,
            forced_held,
            accumulate_fisher_into,
        );
        self.apply_accumulated_adam(&mut accum, hparams, lr_scale, ewc_lambda);
        res
    }

    #[allow(clippy::too_many_arguments)]
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
    #[serde(default)]
    pub conformal_q_bps: f32,
    #[serde(default)]
    pub autopsy_telemetry: BucketAutopsyTelemetry,
    #[serde(default = "default_autopsy_grad_ema")]
    pub autopsy_grad_ema: [f32; 4],
    pub seed_base: u64,
    #[serde(skip)]
    pub red_replay_buffer: Vec<AuditedRedTradeTrajectory>,
    #[serde(skip)]
    pub winner_fisher_buffer: Vec<SettledWinnerTrajectory>,
}

fn default_autopsy_grad_ema() -> [f32; 4] {
    [1.0_f32; 4]
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
            let hparams = AgentHyperParams::diverse_prior_for_horizon(horizon_label, id, &mut seed);
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
            conformal_q_bps: 0.0,
            autopsy_telemetry: BucketAutopsyTelemetry {
                brain_label: horizon_label.to_string(),
                ..Default::default()
            },
            autopsy_grad_ema: [1.0_f32; 4],
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
        if self.autopsy_telemetry.brain_label.is_empty() {
            self.autopsy_telemetry.brain_label = self.horizon_label.clone();
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

    /// Trust-region Polyak parameter interpolation `self <- (1 - beta) * self + beta * other`.
    pub fn polyak_blend_from(&mut self, other: &Self, beta: f32) {
        let b = beta.clamp(0.0, 1.0);
        if b <= 1e-6 || self.population.len() != other.population.len() {
            return;
        }
        let inv = 1.0 - b;
        for (dst, src) in self.population.iter_mut().zip(other.population.iter()) {
            dst.net.polyak_blend_from(&src.net, b);
            let blended_cs =
                (inv * dst.hparams.cost_sensitivity + b * src.hparams.cost_sensitivity).max(0.05);
            dst.hparams.set_log_cost_sens(blended_cs.ln());
            dst.hparams.l1_sparsity = inv * dst.hparams.l1_sparsity + b * src.hparams.l1_sparsity;
            dst.hparams.downside_aversion =
                inv * dst.hparams.downside_aversion + b * src.hparams.downside_aversion;
            dst.hparams.explore_pressure =
                inv * dst.hparams.explore_pressure + b * src.hparams.explore_pressure;
            dst.hparams.conformal_q_bps =
                inv * dst.hparams.conformal_q_bps + b * src.hparams.conformal_q_bps;
            dst.hedge_weight = (inv as f64) * dst.hedge_weight + (b as f64) * src.hedge_weight;
        }
        self.conformal_q_bps = inv * self.conformal_q_bps + b * other.conformal_q_bps;
        let sum_w: f64 = self.population.iter().map(|m| m.hedge_weight).sum();
        if sum_w > 1e-12 {
            for m in &mut self.population {
                m.hedge_weight /= sum_w;
            }
        }
        self.autopsy_telemetry.cost_sensitivity_after = self
            .population
            .iter()
            .map(|m| m.hparams.cost_sensitivity)
            .sum::<f32>()
            / (self.population.len().max(1) as f32);
    }

    pub fn snapshot_ewc_anchors(&mut self) {
        for m in &mut self.population {
            m.net.snapshot_ewc_anchor();
        }
    }

    /// Computes Diagonal Fisher Information `F_i = E_{winners}[(d L / d theta_i)^2]` across all
    /// settled causal winning trajectories so subsequent Red-Trade RL updates lock profitable patterns!
    pub fn compute_fisher_ewc_from_winners(&mut self) {
        self.compute_fisher_ewc_from_winners_le(i64::MAX);
    }

    /// Chronologically bounded variant (`w_ep.exit_ts_sec <= max_train_ts_sec`) for leak-free split RL.
    pub fn compute_fisher_ewc_from_winners_le(&mut self, max_train_ts_sec: i64) {
        self.snapshot_ewc_anchors();
        let winners: Vec<SettledWinnerTrajectory> = self
            .winner_fisher_buffer
            .iter()
            .filter(|w| w.exit_ts_sec <= max_train_ts_sec)
            .cloned()
            .collect();
        if winners.is_empty() {
            return;
        }
        let feat_mean = self.feat_mean.clone();
        let feat_var = self.feat_var.clone();
        let wm = &self.world_model;

        self.population.par_iter_mut().for_each(|member| {
            let n_params = member.net.param_count();
            let mut fisher_acc = vec![0.0_f32; n_params];
            let mut count = 0usize;
            for w_ep in winners.iter().rev().take(12) {
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
        self.audit_and_insert_loss_event(
            symbol,
            exit_ts_sec,
            net_pnl_usd,
            net_pnl_usd,
            net_r,
            direction_sign,
            is_option,
            entry_offset,
            prior_regime_z,
            bars,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn audit_and_insert_loss_event(
        &mut self,
        symbol: String,
        exit_ts_sec: i64,
        net_pnl_usd: f64,
        gross_pnl_usd: f64,
        net_r: f32,
        direction_sign: f32,
        is_option: bool,
        entry_offset: usize,
        prior_regime_z: [f32; WM_DIM],
        bars: Vec<CausalTrajectoryBar>,
    ) -> AuditedRedTradeTrajectory {
        let t_len = bars.len();
        let entry_idx = entry_offset.min(t_len.saturating_sub(2));
        let unit_cost_chosen = if is_option {
            bars[entry_idx].one_way_cost_option
        } else {
            bars[entry_idx].one_way_cost_shares
        };
        let rt_cost_chosen = 2.0 * unit_cost_chosen;
        let rt_cost_alt = if is_option {
            2.0 * bars[entry_idx].one_way_cost_shares
        } else {
            2.0 * bars[entry_idx].one_way_cost_option
        };

        let mut cum_chosen = 0.0_f32;
        let mut best_cf_net = -rt_cost_chosen;
        let mut best_exit_off = entry_idx;
        let mut cum_alt = 0.0_f32;
        let mut best_cf_alt = -rt_cost_alt;
        let mut best_alt_exit_off = entry_idx;

        for (idx, b) in bars.iter().enumerate().skip(entry_idx) {
            let (r_opt_dir, opt_theta) = if b.option_gamma_boost > 1e-6 {
                (
                    b.ret_oo_shares * b.option_gamma_boost,
                    b.option_theta_per_bar.max(0.0),
                )
            } else {
                (b.ret_oo_option, 0.0)
            };
            let (step_chosen, step_alt) = if is_option {
                (
                    direction_sign * r_opt_dir - opt_theta,
                    direction_sign * b.ret_oo_shares,
                )
            } else {
                (
                    direction_sign * b.ret_oo_shares,
                    direction_sign * r_opt_dir - opt_theta,
                )
            };
            cum_chosen += step_chosen;
            cum_alt += step_alt;
            if idx + 1 < t_len {
                let net_c = cum_chosen - rt_cost_chosen;
                if net_c > best_cf_net {
                    best_cf_net = net_c;
                    best_exit_off = idx + 1;
                }
                let net_a = cum_alt - rt_cost_alt;
                if net_a > best_cf_alt {
                    best_cf_alt = net_a;
                    best_alt_exit_off = idx + 1;
                }
            }
        }
        let approx_r_unit = (rt_cost_chosen * 2.5).max(0.008);
        let best_cf_r = best_cf_net / approx_r_unit;
        let opt_allowed = bars[entry_idx].option_allowed;
        let vehicle_regret_delta = if is_option || opt_allowed {
            (best_cf_alt - best_cf_net).max(0.0)
        } else {
            0.0
        };

        let is_vehicle_regret = (is_option && net_pnl_usd < 0.0 && best_cf_alt > best_cf_net + 0.0015)
            || (!is_option && opt_allowed && vehicle_regret_delta > rt_cost_chosen.max(0.0025));

        let (fault_kind, optimal_exit_offset) = if is_vehicle_regret {
            (
                LossFaultKind::VehicleSelectionRegret,
                best_alt_exit_off.max(entry_idx),
            )
        } else if best_cf_r > 0.12
            && best_exit_off > entry_idx
            && (gross_pnl_usd > 0.0 || best_cf_net > 0.0)
        {
            (LossFaultKind::HoldingEodDecay, best_exit_off)
        } else {
            (LossFaultKind::FrictionInversion, entry_idx)
        };

        let cost_inversion_boost = if fault_kind == LossFaultKind::FrictionInversion
            && gross_pnl_usd > 0.0
            && net_pnl_usd < 0.0
        {
            1.35_f32
        } else {
            1.0_f32
        };
        let priority_weight =
            ((1.0 + 4.0 * net_r.abs() * net_r.abs()) * cost_inversion_boost).clamp(1.0, 35.0);

        if self.autopsy_telemetry.cost_sensitivity_before <= 0.0 {
            self.autopsy_telemetry.cost_sensitivity_before = self
                .population
                .iter()
                .map(|m| m.hparams.cost_sensitivity)
                .sum::<f32>()
                / (self.population.len().max(1) as f32);
        }
        let is_new_event = !self
            .red_replay_buffer
            .iter()
            .any(|r| r.symbol == symbol && r.exit_ts_sec == exit_ts_sec);
        if is_new_event {
            match fault_kind {
                LossFaultKind::FrictionInversion => {
                    self.autopsy_telemetry.friction_inversion_events += 1;
                }
                LossFaultKind::HoldingEodDecay => {
                    self.autopsy_telemetry.holding_eod_decay_events += 1;
                }
                LossFaultKind::VehicleSelectionRegret => {
                    self.autopsy_telemetry.vehicle_regret_events += 1;
                }
                LossFaultKind::DeadzoneParalysisRegret => {
                    self.autopsy_telemetry.deadzone_paralysis_events += 1;
                }
            }
        }
        self.autopsy_telemetry.cost_sensitivity_after = self
            .population
            .iter()
            .map(|m| m.hparams.cost_sensitivity)
            .sum::<f32>()
            / (self.population.len().max(1) as f32);

        let draft = AuditedRedTradeTrajectory {
            symbol,
            exit_ts_sec,
            net_pnl_usd,
            gross_pnl_usd,
            net_r,
            direction_sign,
            is_option_routed: is_option,
            priority_weight,
            fault_kind,
            entry_offset: entry_idx,
            optimal_exit_offset,
            best_counterfactual_net_r: best_cf_r,
            vehicle_regret_delta,
            saliency_dx: 0.0,
            saliency_dh: 0.0,
            prior_regime_z,
            bars,
        };
        let (targets, v_tgts) = Self::build_autopsy_targets(&draft);
        let forced_held = if fault_kind == LossFaultKind::HoldingEodDecay {
            let realized_ret = (net_r * approx_r_unit).clamp(-0.15, 0.15);
            Some(ForcedHeldContext {
                entry_offset: entry_idx,
                optimal_exit_offset,
                direction_sign,
                is_option_routed: is_option,
                peak_giveback_ret: (best_cf_net - realized_ret).max(0.004),
            })
        } else {
            None
        };

        // Compute end-to-end BPTT saliency `|d(Loss)/dx_t|` and `|d(Loss)/dh_t|` on champion member (read-only `lr_scale = 0.0`)
        let champ_idx = self
            .population
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.causal_score.total_cmp(&b.1.causal_score))
            .map(|(i, _)| i)
            .unwrap_or(0);
        let mut probe_net = self.population[champ_idx].net.clone();
        let hp = self.population[champ_idx].hparams.clone();
        let (_, sal_dx, sal_dh) = probe_net.train_bptt_chunk_with_held_context(
            &draft.bars,
            &draft.prior_regime_z,
            Some(&self.world_model),
            &self.feat_mean,
            &self.feat_var,
            &hp,
            0.0,
            0.0,
            Some((&targets, priority_weight)),
            v_tgts.as_deref().map(|v| (v, priority_weight)),
            forced_held,
            None,
        );

        let record = AuditedRedTradeTrajectory {
            saliency_dx: sal_dx,
            saliency_dh: sal_dh,
            ..draft
        };

        self.insert_dedup_autopsy_record(record.clone());
        record
    }

    fn insert_dedup_autopsy_record(&mut self, record: AuditedRedTradeTrajectory) {
        if let Some(pos) = self.red_replay_buffer.iter().position(|r| {
            r.symbol == record.symbol
                && r.exit_ts_sec == record.exit_ts_sec
                && r.fault_kind == record.fault_kind
        }) {
            if record.priority_weight >= self.red_replay_buffer[pos].priority_weight {
                self.red_replay_buffer[pos] = record;
            }
        } else {
            self.red_replay_buffer.push(record);
        }
        if self.red_replay_buffer.len() > MAX_RED_REPLAY_CAP {
            self.red_replay_buffer
                .sort_by(|a, b| b.priority_weight.total_cmp(&a.priority_weight));
            self.red_replay_buffer.truncate(MAX_RED_REPLAY_CAP);
        }
    }

    fn build_autopsy_targets(
        chosen: &AuditedRedTradeTrajectory,
    ) -> (Vec<Option<f32>>, Option<Vec<Option<f32>>>) {
        let t_len = chosen.bars.len();
        let mut targets = vec![None; t_len];
        let mut v_tgts: Option<Vec<Option<f32>>> = None;
        match chosen.fault_kind {
            LossFaultKind::FrictionInversion => {
                for slot in targets
                    .iter_mut()
                    .skip(chosen.entry_offset.saturating_sub(1))
                {
                    *slot = Some(0.0);
                }
            }
            LossFaultKind::HoldingEodDecay => {
                for slot in targets.iter_mut().skip(chosen.optimal_exit_offset) {
                    *slot = Some(0.0);
                }
            }
            LossFaultKind::VehicleSelectionRegret => {
                let target_v = if chosen.is_option_routed { 0.05 } else { 0.92 };
                let mut vt = vec![None; t_len];
                let end_hold = chosen.optimal_exit_offset.max(chosen.entry_offset + 1).min(t_len);
                for slot in vt.iter_mut().take(end_hold).skip(chosen.entry_offset) {
                    *slot = Some(target_v);
                }
                v_tgts = Some(vt);
                if chosen.is_option_routed && chosen.net_pnl_usd < 0.0 {
                    // Option blew up; prefer alternate or smaller sizing after entry if net-positive
                    if let Some(last) = targets.last_mut() {
                        *last = Some(0.0);
                    }
                }
            }
            LossFaultKind::DeadzoneParalysisRegret => {
                let sgn = if chosen.direction_sign.abs() > 1e-4 {
                    chosen.direction_sign.signum()
                } else {
                    1.0
                };
                let mut vt = vec![None; t_len];
                let mut any_opt = false;
                let end_hold = chosen
                    .optimal_exit_offset
                    .max(chosen.entry_offset + 1)
                    .min(t_len.saturating_sub(1));
                for idx in chosen.entry_offset.min(end_hold)..end_hold {
                    targets[idx] = Some(sgn * 0.55);
                    if chosen.bars[idx].option_allowed {
                        vt[idx] = Some(if chosen.is_option_routed { 0.88 } else { 0.08 });
                        any_opt = true;
                    }
                }
                for slot in targets.iter_mut().skip(end_hold) {
                    *slot = Some(0.0);
                }
                if any_opt {
                    v_tgts = Some(vt);
                }
            }
        }
        (targets, v_tgts)
    }

    /// Unconstrained log-space `log_cost_sens` adaptation driven by causal `T_train` friction-inversion
    /// prevalence (`cost_sensitivity = exp(log_cost_sens)` with zero `2.20` / `2.60` / `3.15` ceiling).
    pub fn adapt_cost_sensitivity_from_autopsy_cycle(
        &mut self,
        friction_inv_count: usize,
        hold_decay_count: usize,
        settled_trades: usize,
    ) {
        if self.autopsy_telemetry.cost_sensitivity_before <= 0.0 {
            self.autopsy_telemetry.cost_sensitivity_before = self
                .population
                .iter()
                .map(|m| m.hparams.cost_sensitivity)
                .sum::<f32>()
                / (self.population.len().max(1) as f32);
        }
        if settled_trades == 0 {
            return;
        }
        let inv_rate = (friction_inv_count as f32) / (settled_trades.max(1) as f32);
        let decay_rate = (hold_decay_count as f32) / (settled_trades.max(1) as f32);
        // Smooth log-space update proportional to excess friction-inversion rate above healthy equilibrium (~0.25)
        let delta_log = ((inv_rate - 0.25) * 0.28 + decay_rate * 0.05).clamp(-0.12, 0.22);
        if delta_log.abs() > 1e-4 {
            for m in &mut self.population {
                m.hparams.sync_log_cost_sens();
                m.hparams.set_log_cost_sens(m.hparams.log_cost_sens + delta_log);
                if delta_log > 0.0 {
                    m.hparams.downside_aversion =
                        (m.hparams.downside_aversion + 0.06 * delta_log).clamp(0.40, 8.0);
                }
            }
        }
        self.autopsy_telemetry.cost_sensitivity_after = self
            .population
            .iter()
            .map(|m| m.hparams.cost_sensitivity)
            .sum::<f32>()
            / (self.population.len().max(1) as f32);
    }

    fn apply_paralysis_meta_adaptation(&mut self) {
        if self.autopsy_telemetry.cost_sensitivity_before <= 0.0 {
            self.autopsy_telemetry.cost_sensitivity_before = self
                .population
                .iter()
                .map(|m| m.hparams.cost_sensitivity)
                .sum::<f32>()
                / (self.population.len().max(1) as f32);
        }
        let allow_cost_relax = self.autopsy_telemetry.friction_inversion_events
            <= self.autopsy_telemetry.deadzone_paralysis_events;
        for m in &mut self.population {
            if allow_cost_relax {
                m.hparams.sync_log_cost_sens();
                m.hparams.set_log_cost_sens(m.hparams.log_cost_sens - 0.14);
                m.hparams.l1_sparsity = (m.hparams.l1_sparsity * 0.85).max(1e-6);
                m.hparams.explore_pressure = (m.hparams.explore_pressure + 0.10).min(0.95);
                for b in &mut m.net.b_d_heads {
                    *b = (*b - 0.08).clamp(-3.2, 0.8);
                }
            } else {
                // Friction inversions dominate paralysis: never blind cost sensitivity;
                // apply only a gentle exploration nudge.
                m.hparams.explore_pressure = (m.hparams.explore_pressure + 0.05).min(0.95);
            }
        }
        self.autopsy_telemetry.cost_sensitivity_after = self
            .population
            .iter()
            .map(|m| m.hparams.cost_sensitivity)
            .sum::<f32>()
            / (self.population.len().max(1) as f32);
    }

    /// Diagnoses option-vs-shares counterfactual regret across causal episodes and trains
    /// both the 4-channel critic and specialist vehicle heads in parallel.
    pub fn train_vehicle_regret_episodes(
        &mut self,
        episodes: &[CausalSessionEpisode],
        steps: usize,
        lr_scale: f32,
    ) -> usize {
        self.train_vehicle_regret_episodes_ext(episodes, steps, lr_scale, true)
    }

    /// Internal vehicle-regret updater allowing speculative shadow candidates to run
    /// without polluting champion `autopsy_telemetry`.
    pub fn train_vehicle_regret_episodes_ext(
        &mut self,
        episodes: &[CausalSessionEpisode],
        steps: usize,
        lr_scale: f32,
        record_telemetry: bool,
    ) -> usize {
        if episodes.is_empty() || steps == 0 {
            return 0;
        }
        let mut regret_candidates: Vec<(usize, Vec<Option<f32>>)> = Vec::new();
        for (ep_i, ep) in episodes.iter().enumerate() {
            if ep.bars.len() < 4 {
                continue;
            }
            let ep_cf = compute_episode_counterfactual_targets(&ep.bars, 0.86);
            let mut v_tgts = vec![None; ep.bars.len()];
            let mut ep_events = 0usize;
            for (t, b) in ep.bars.iter().enumerate() {
                if !b.option_allowed || b.force_flat {
                    continue;
                }
                let tgt = if b.cf_valid {
                    b.cf_net_returns
                } else {
                    ep_cf.critic_targets[t]
                };
                let sh_best = tgt[0].max(tgt[1]);
                let opt_best = tgt[2].max(tgt[3]);
                if (opt_best - sh_best) > 0.0045 && opt_best > 0.0018 {
                    v_tgts[t] = Some(0.90);
                    ep_events += 1;
                } else if (sh_best - opt_best) > 0.0020 && sh_best > 0.0010 {
                    v_tgts[t] = Some(0.08);
                    ep_events += 1;
                }
            }
            if ep_events > 0 {
                if record_telemetry {
                    self.autopsy_telemetry.vehicle_regret_events += 1;
                }
                regret_candidates.push((ep_i, v_tgts));
            }
        }
        if regret_candidates.is_empty() {
            return 0;
        }
        let detected = regret_candidates.len();
        let feat_mean = self.feat_mean.clone();
        let feat_var = self.feat_var.clone();
        let wm = &self.world_model;
        let seed_base = self.seed_base ^ self.online_updates_count.wrapping_mul(0x7F4A_7C15_9E37_79B9);
        self.population.par_iter_mut().for_each(|member| {
            let mut seed = seed_base ^ (member.id as u64 + 1).wrapping_mul(0x94D0_49BB_1331_11EB);
            for _ in 0..steps {
                let pick = (xorshift64(&mut seed) as usize) % regret_candidates.len();
                let (ep_i, ref v_tgts) = regret_candidates[pick];
                let ep = &episodes[ep_i];
                let _ = member.net.train_bptt_chunk_full(
                    &ep.bars,
                    &ep.prior_regime_z,
                    Some(wm),
                    &feat_mean,
                    &feat_var,
                    &member.hparams,
                    lr_scale,
                    0.02,
                    None,
                    Some((v_tgts, 2.5)),
                    None,
                );
            }
        });
        detected
    }

    /// Causal Adaptive Conformal Inference (`q_conformal`) computed strictly on past/inner-validation
    /// bars (`b.ts_sec <= max_ts_sec`): measures the empirical `(1 - alpha)` optimism residual between
    /// Stage A World-Model LCB excursion predictions and realized net-of-cost liquidation trajectories (`V_liq`),
    /// adapting `alpha` online via causal empirical Profit Factor (Gibbs & Candes ACI) without any `> t` lookahead.
    pub fn calibrate_causal_conformal_residual(
        &mut self,
        cal_episodes: &[CausalSessionEpisode],
        max_ts_sec: i64,
        empirical_pf: f64,
        empirical_trades: usize,
    ) -> f32 {
        let mut residuals_bps: Vec<f32> = Vec::new();
        let mut fallback_residuals_bps: Vec<f32> = Vec::new();
        let mut rt_cost_sum = 0.0_f64;
        let mut abs_target_sum = 0.0_f64;
        let mut rt_cost_n = 0usize;

        for ep in cal_episodes {
            let valid_len = ep.bars.partition_point(|b| b.ts_sec <= max_ts_sec);
            if valid_len < 8 {
                continue;
            }
            let slice = &ep.bars[..valid_len];
            let ep_cf = compute_episode_counterfactual_targets(slice, 0.85);
            let mut wm_z = ep.prior_regime_z;
            for (t, b) in slice.iter().enumerate() {
                let x_norm = normalize_212_with_stats(&b.x_212, &self.feat_mean, &self.feat_var);
                let (next_z, wm_pred) = self.world_model.forward_step(&x_norm, &wm_z);
                wm_z = next_z;
                if t + 6 >= valid_len || b.force_flat || !b.entry_allowed {
                    continue;
                }
                let rt_cost_bps = (2.0 * b.one_way_cost_bps).max(4.0);
                rt_cost_sum += rt_cost_bps as f64;
                rt_cost_n += 1;

                let pred_l = wm_directional_lcb_bps(&wm_pred, 1.0, rt_cost_bps, 1.35);
                let pred_s = wm_directional_lcb_bps(&wm_pred, -1.0, rt_cost_bps, 1.35);
                let (pred_best, real_best) = if pred_l >= pred_s {
                    let rl = ep_cf.critic_targets[t][0].max(if b.option_allowed {
                        ep_cf.critic_targets[t][2]
                    } else {
                        f32::NEG_INFINITY
                    }) * 10_000.0;
                    (pred_l, rl)
                } else {
                    let rs = ep_cf.critic_targets[t][1].max(if b.option_allowed {
                        ep_cf.critic_targets[t][3]
                    } else {
                        f32::NEG_INFINITY
                    }) * 10_000.0;
                    (pred_s, rs)
                };
                if real_best.is_finite() {
                    abs_target_sum += real_best.abs() as f64;
                }
                let resid = pred_best - real_best;
                if resid.is_finite() {
                    if pred_best > -0.30 * rt_cost_bps {
                        residuals_bps.push(resid);
                    } else if pred_best > -0.90 * rt_cost_bps && fallback_residuals_bps.len() < 512 {
                        fallback_residuals_bps.push(resid);
                    }
                }
            }
        }

        if residuals_bps.len() < 16 {
            residuals_bps.extend(fallback_residuals_bps);
        }
        if residuals_bps.is_empty() {
            return self.conformal_q_bps;
        }
        residuals_bps.sort_by(|a, b| a.total_cmp(b));

        // Adaptive Conformal Inference miscoverage level `alpha` + horizon-dispersion bound:
        // - Dormant (`0` trades): `alpha = 0.62` (`38th` percentile) with gentle cap so calibration never freezes wakeup.
        // - Profitable (`PF >= 1.25`): `alpha = 0.50` (`50th` percentile gentle epistemic trim).
        // - Over-optimistic / friction-bleeding (`PF < 1.0`): `alpha -> 0.28` (`72nd` percentile epistemic LCB).
        let alpha = if empirical_trades == 0 {
            0.62_f32
        } else {
            let pf_f = (empirical_pf as f32).clamp(0.20, 2.50);
            (0.42 + 0.18 * (pf_f - 1.10)).clamp(0.28, 0.54)
        };
        let n = residuals_bps.len();
        let rank = (((1.0 - alpha) * ((n + 1) as f32)).ceil() as usize)
            .saturating_sub(1)
            .min(n - 1);
        let mean_rt_bps = if rt_cost_n > 0 {
            (rt_cost_sum / (rt_cost_n as f64)) as f32
        } else {
            16.0
        };
        let mean_abs_target_bps = if rt_cost_n > 0 {
            (abs_target_sum / (rt_cost_n as f64)) as f32
        } else {
            30.0
        };
        let disp_cap = (0.45 * mean_rt_bps)
            .max(0.30 * mean_abs_target_bps)
            .clamp(6.0, 22.0);
        let eff_cap = if empirical_trades == 0 {
            (disp_cap * 0.60).max(4.0)
        } else {
            disp_cap
        };
        let q_bps = residuals_bps[rank].clamp(0.0, eff_cap);
        self.conformal_q_bps = q_bps;
        for m in &mut self.population {
            m.hparams.conformal_q_bps = q_bps;
        }
        q_bps
    }

    /// Intraday Held-Trajectory Exit Calibration (`ForcedHeldContext` on causal warmup sessions):
    /// Exposes `s_t = (drawdown_from_peak, session_time_left, unrealized_net_ret)` to dense post-peak
    /// `V_liq` exit gradients (`a_t -> 0.0`) during warmup without inflating flat entry probabilities.
    pub fn train_intraday_exit_timing_episodes(
        &mut self,
        episodes: &[CausalSessionEpisode],
        steps: usize,
        lr_scale: f32,
    ) -> usize {
        if episodes.is_empty() || steps == 0 || lr_scale <= 0.0 {
            return 0;
        }
        let mut exit_candidates: Vec<(usize, ForcedHeldContext, Vec<Option<f32>>)> = Vec::new();
        for (ep_i, ep) in episodes.iter().enumerate() {
            if !ep.is_intraday_session || ep.bars.len() < 8 {
                continue;
            }
            let t_len = ep.bars.len();
            let ep_cf = compute_episode_counterfactual_targets(&ep.bars, 0.86);
            let search_end = (t_len * 3 / 5).max(2).min(t_len - 4);
            let mut best_entry = None;
            let mut best_score = 0.0_f32;
            for t in 0..search_end {
                let b = &ep.bars[t];
                if !b.entry_allowed || b.force_flat {
                    continue;
                }
                let rt_c = 2.0 * b.one_way_cost_shares;
                let net_adv = ep_cf.best_entry_net[t];
                if net_adv > (0.55 * rt_c).max(0.0008) && net_adv > best_score {
                    best_score = net_adv;
                    let dir = if ep_cf.critic_targets[t][0] >= ep_cf.critic_targets[t][1] {
                        1.0_f32
                    } else {
                        -1.0_f32
                    };
                    best_entry = Some((t, dir, rt_c));
                }
            }
            let Some((t_entry, dir, rt_c)) = best_entry else {
                continue;
            };
            let mut cum_gross = 0.0_f32;
            let mut peak_net = -rt_c;
            let mut t_peak = t_entry + 1;
            for (idx, b) in ep.bars.iter().enumerate().take(t_len - 1).skip(t_entry) {
                cum_gross += dir * b.ret_oo_shares;
                let net_here = cum_gross - rt_c;
                if net_here > peak_net {
                    peak_net = net_here;
                    t_peak = (idx + 1).min(t_len - 1);
                }
            }
            let eod_net = cum_gross - rt_c;
            let giveback = peak_net - eod_net;
            let min_hold_bars = if t_len >= 140 { 12usize } else { 4usize };
            if peak_net > (1.35 * rt_c).max(0.0020)
                && t_peak >= t_entry + min_hold_bars
                && t_peak + 1 < t_len
                && (eod_net < 0.40 * peak_net || giveback > (1.05 * rt_c).max(0.0020))
            {
                let mut targets = vec![None; t_len];
                for slot in targets.iter_mut().skip(t_peak) {
                    *slot = Some(0.0);
                }
                let fh = ForcedHeldContext {
                    entry_offset: t_entry,
                    optimal_exit_offset: t_peak,
                    direction_sign: dir,
                    is_option_routed: false,
                    peak_giveback_ret: giveback.clamp(0.002, 0.06),
                };
                exit_candidates.push((ep_i, fh, targets));
                if exit_candidates.len() >= 36 {
                    break;
                }
            }
        }
        if exit_candidates.is_empty() {
            return 0;
        }
        let detected = exit_candidates.len();
        let feat_mean = self.feat_mean.clone();
        let feat_var = self.feat_var.clone();
        let wm = &self.world_model;
        let seed_base =
            self.seed_base ^ self.generation_steps.wrapping_mul(0x94D0_49BB_1331_11EB);
        self.population.par_iter_mut().for_each(|member| {
            let mut seed =
                seed_base ^ (member.id as u64 + 1).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            for _ in 0..steps {
                let pick = (xorshift64(&mut seed) as usize) % exit_candidates.len();
                let (ep_i, fh, ref tgts) = exit_candidates[pick];
                let ep = &episodes[ep_i];
                let _ = member.net.train_bptt_chunk_with_held_context(
                    &ep.bars,
                    &ep.prior_regime_z,
                    Some(wm),
                    &feat_mean,
                    &feat_var,
                    &member.hparams,
                    lr_scale,
                    0.02,
                    Some((tgts, 1.35)),
                    None,
                    Some(fh),
                    None,
                );
            }
        });
        detected
    }

    /// Active Counterfactual `Paralysis Regret` Audit & BPTT (`<= max_ts_sec`):
    /// Detects missed causal dislocations where the live policy output `a_t == 0.0` despite strictly
    /// positive net-of-cost excursions, computes exact analytical BPTT saliency `(|dL/dx|, |dL/dh|)`
    /// even when `0` realized trades occurred, and (when `steps > 0`) runs balanced wake + post-peak/chop
    /// `0.0` counterfactual BPTT across all `M = 24` specialists.
    pub fn audit_and_train_paralysis_regret(
        &mut self,
        episodes: &[CausalSessionEpisode],
        max_ts_sec: i64,
        steps: usize,
        lr_scale: f32,
    ) -> (usize, f32, f32) {
        if episodes.is_empty() {
            return (0, 0.0, 0.0);
        }
        let champ_idx = self
            .population
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.causal_score.total_cmp(&b.1.causal_score))
            .map(|(i, _)| i)
            .unwrap_or(0);

        struct ParalysisWakeItem {
            ep_i: usize,
            valid_len: usize,
            targets: Vec<Option<f32>>,
            v_tgts: Option<Vec<Option<f32>>>,
        }

        let mut wake_eps: Vec<ParalysisWakeItem> = Vec::new();
        let mut paralysis_windows = 0usize;
        let mut chop_counter_windows = 0usize;

        for (ep_i, ep) in episodes.iter().enumerate() {
            let valid_len = ep.bars.partition_point(|b| b.ts_sec <= max_ts_sec);
            if valid_len < 4 {
                continue;
            }
            let slice = &ep.bars[..valid_len];
            let ep_cf = compute_episode_counterfactual_targets(slice, 0.85);
            let mut targets = vec![None; valid_len];
            let mut v_tgts = vec![None; valid_len];
            let mut any_opt = false;
            let mut missed_in_ep = 0usize;
            let mut chop_in_ep = 0usize;

            let lead = &self.population[champ_idx];
            let mut wm_z = ep.prior_regime_z;
            let mut prev_h = [0.0_f32; HIDDEN_DIM];
            let mut hold_until = 0usize;

            for (t, b) in slice.iter().enumerate() {
                let x_norm = normalize_212_with_stats(&b.x_212, &self.feat_mean, &self.feat_var);
                let (next_z, wm_pred) = self.world_model.forward_step(&x_norm, &wm_z);
                wm_z = next_z;
                let self_s = SelfAwarenessState {
                    current_pos: 0.0,
                    unrealized_net_ret: 0.0,
                    peak_excursion: 0.0,
                    drawdown_from_peak: 0.0,
                    log_bars_held: 0.0,
                    one_way_cost_bps: b.one_way_cost_bps,
                    session_time_left: b.session_time_left,
                    ema_turnover: 0.0,
                };
                let step = lead.net.forward_step_with_wm_cal(
                    &x_norm,
                    &wm_z,
                    &wm_pred,
                    self_s,
                    &prev_h,
                    wm_z,
                    lead.hparams.cost_sensitivity,
                    lead.hparams.downside_aversion,
                    lead.hparams.conformal_q_bps,
                    b.option_allowed,
                );
                prev_h = step.next_hidden;

                if b.force_flat {
                    targets[t] = Some(0.0);
                    continue;
                }
                if t < hold_until {
                    continue;
                }
                if !b.entry_allowed {
                    continue;
                }
                let tgt = ep_cf.critic_targets[t];
                let long_best =
                    tgt[0].max(if b.option_allowed { tgt[2] } else { f32::NEG_INFINITY });
                let short_best =
                    tgt[1].max(if b.option_allowed { tgt[3] } else { f32::NEG_INFINITY });
                let best_net = long_best.max(short_best);
                let rt_c = 2.0 * b.one_way_cost_shares;
                let wake_hurdle = (0.70 * rt_c).max(0.00045);

                if step.action.abs() <= 1e-5 && best_net > wake_hurdle {
                    let dir = if long_best >= short_best { 1.0_f32 } else { -1.0_f32 };
                    let opt_better = b.option_allowed
                        && (if dir > 0.0 {
                            tgt[2] > tgt[0] + 0.0015
                        } else {
                            tgt[3] > tgt[1] + 0.0015
                        });
                    // Find causal excursion peak over next up to 8 bars so wake training also teaches crisp post-peak exit!
                    let look_end = (t + 8).min(valid_len - 1);
                    let mut cum_g = 0.0_f32;
                    let mut pk_g = 0.0_f32;
                    let mut pk_off = (t + 1).min(look_end);
                    for (k_idx, kb) in slice.iter().enumerate().take(look_end).skip(t) {
                        cum_g += dir * kb.ret_oo_shares;
                        if cum_g > pk_g {
                            pk_g = cum_g;
                            pk_off = (k_idx + 1).min(look_end);
                        }
                    }
                    for h_idx in t..pk_off.max(t + 1) {
                        if !slice[h_idx].force_flat {
                            targets[h_idx] = Some(dir * 0.58);
                            if slice[h_idx].option_allowed {
                                v_tgts[h_idx] = Some(if opt_better { 0.88 } else { 0.10 });
                                any_opt = true;
                            }
                        }
                    }
                    if pk_off < valid_len {
                        targets[pk_off] = Some(0.0);
                    }
                    hold_until = (pk_off + 1).min(valid_len);
                    missed_in_ep += 1;
                } else if best_net < -0.15 * rt_c {
                    targets[t] = Some(0.0);
                    chop_in_ep += 1;
                }
            }

            if missed_in_ep > 0 {
                paralysis_windows += missed_in_ep;
                chop_counter_windows += chop_in_ep;
                wake_eps.push(ParalysisWakeItem {
                    ep_i,
                    valid_len,
                    targets,
                    v_tgts: if any_opt { Some(v_tgts) } else { None },
                });
                if wake_eps.len() >= 32 {
                    break;
                }
            }
        }

        if wake_eps.is_empty() {
            return (0, 0.0, 0.0);
        }

        let events_found = wake_eps.len();
        self.autopsy_telemetry.deadzone_paralysis_events += events_found;
        self.autopsy_telemetry.flat_regret_windows_replayed +=
            paralysis_windows + (chop_counter_windows > 0) as usize;

        // Compute exact analytical BPTT saliency `(|dL/dx|, |dL/dh|)` on champion member (`lr_scale = 0.0`)
        let mut probe_net = self.population[champ_idx].net.clone();
        let hp = self.population[champ_idx].hparams.clone();
        let probe_n = wake_eps.len().min(4);
        let mut dx_sum = 0.0_f32;
        let mut dh_sum = 0.0_f32;
        for item in wake_eps.iter().take(probe_n) {
            let ep = &episodes[item.ep_i];
            let slice = &ep.bars[..item.valid_len];
            let cap = slice.len().min(96);
            let (_, sdx, sdh) = probe_net.train_bptt_chunk_with_held_context(
                &slice[..cap],
                &ep.prior_regime_z,
                Some(&self.world_model),
                &self.feat_mean,
                &self.feat_var,
                &hp,
                0.0,
                0.0,
                Some((&item.targets[..cap], 3.5)),
                item.v_tgts.as_deref().map(|v| (&v[..cap], 2.5)),
                None,
                None,
            );
            dx_sum += sdx;
            dh_sum += sdh;
        }
        let mean_dx = dx_sum / (probe_n.max(1) as f32);
        let mean_dh = dh_sum / (probe_n.max(1) as f32);

        if steps == 0 || lr_scale <= 0.0 {
            return (events_found, mean_dx, mean_dh);
        }

        self.apply_paralysis_meta_adaptation();

        let feat_mean = self.feat_mean.clone();
        let feat_var = self.feat_var.clone();
        let wm = &self.world_model;
        let seed_base =
            self.seed_base ^ self.generation_steps.wrapping_mul(0xD6E8_FEB8_6659_FD93);
        self.population.par_iter_mut().for_each(|member| {
            let mut seed =
                seed_base ^ (member.id as u64 + 1).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            for _ in 0..steps {
                let pick = (xorshift64(&mut seed) as usize) % wake_eps.len();
                let item = &wake_eps[pick];
                let ep = &episodes[item.ep_i];
                let full_slice = &ep.bars[..item.valid_len];
                let max_c = if ep.is_intraday_session {
                    member.hparams.chunk_len.max(84)
                } else {
                    member.hparams.chunk_len.max(32)
                };
                let (b_slice, t_slice, v_slice) = if full_slice.len() <= max_c {
                    (full_slice, &item.targets[..], item.v_tgts.as_deref())
                } else {
                    let st = (xorshift64(&mut seed) as usize) % (full_slice.len() - max_c + 1);
                    (
                        &full_slice[st..st + max_c],
                        &item.targets[st..st + max_c],
                        item.v_tgts.as_deref().map(|v| &v[st..st + max_c]),
                    )
                };
                let _ = member.net.train_bptt_chunk_full(
                    b_slice,
                    &ep.prior_regime_z,
                    Some(wm),
                    &feat_mean,
                    &feat_var,
                    &member.hparams,
                    lr_scale,
                    0.0,
                    Some((t_slice, 3.4)),
                    v_slice.map(|v| (v, 2.2)),
                    None,
                );
            }
        });
        (events_found, mean_dx, mean_dh)
    }

    /// Diagnoses `DeadzoneParalysisRegret` when active policies freeze at 0 trades despite
    /// strictly net-positive counterfactual opportunities, and wakes `z_k` / `tau_k`.
    pub fn train_deadzone_paralysis_wake(
        &mut self,
        episodes: &[CausalSessionEpisode],
        steps: usize,
        lr_scale: f32,
    ) -> usize {
        self.audit_and_train_paralysis_regret(episodes, i64::MAX, steps, lr_scale)
            .0
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
        self.train_red_autopsy_and_causal_batch_parallel_le(
            causal_episodes,
            i64::MAX,
            steps,
            lr_scale,
            ewc_lambda,
        );
    }

    /// Chronologically bounded 4-Vector GradNorm Loss-Autopsy + Active `FLAT` Noise Replay (`<= max_train_ts_sec`):
    /// Separates gradients into 4 orthogonal vectors:
    /// - `0: L_friction_inv` (`FrictionInversion` overtrading/cost-inversion entries)
    /// - `1: L_hold_decay` (`HoldingEodDecay` replayed with `ForcedHeldContext` peak-giveback/EOD regret)
    /// - `2: L_vehicle_regret` (`VehicleSelectionRegret` counterfactual equity-vs-option routing)
    /// - `3: L_flat_counterfactual` (active replay of negative-expectancy `FLAT` noise windows + paralysis wake)
    ///
    /// Normalizes each vector by its running causal EMA L2 norm before summing with causal episode BPTT!
    pub fn train_red_autopsy_and_causal_batch_parallel_le(
        &mut self,
        causal_episodes: &[CausalSessionEpisode],
        max_train_ts_sec: i64,
        steps: usize,
        lr_scale: f32,
        ewc_lambda: f32,
    ) {
        let reds_f: Vec<AuditedRedTradeTrajectory> = self
            .red_replay_buffer
            .iter()
            .filter(|r| {
                r.exit_ts_sec <= max_train_ts_sec && r.fault_kind == LossFaultKind::FrictionInversion
            })
            .cloned()
            .collect();
        let reds_h: Vec<AuditedRedTradeTrajectory> = self
            .red_replay_buffer
            .iter()
            .filter(|r| {
                r.exit_ts_sec <= max_train_ts_sec && r.fault_kind == LossFaultKind::HoldingEodDecay
            })
            .cloned()
            .collect();
        let reds_v: Vec<AuditedRedTradeTrajectory> = self
            .red_replay_buffer
            .iter()
            .filter(|r| {
                r.exit_ts_sec <= max_train_ts_sec
                    && r.fault_kind == LossFaultKind::VehicleSelectionRegret
            })
            .cloned()
            .collect();

        // Build Vector 3 (`L_flat_counterfactual`): negative-expectancy chop & whipsaw windows where `FLAT` (`0.0`)
        // strictly dominates risk-adjusted post-cost trajectories (`<= max_train_ts_sec`) across `1m`, `5m`, AND `1h`.
        let mut flat_noise_eps: Vec<(usize, usize, usize, Vec<Option<f32>>)> = Vec::new();
        let mut flat_windows_found = 0usize;
        for (ep_i, ep) in causal_episodes.iter().enumerate() {
            let valid_len = ep.bars.partition_point(|b| b.ts_sec <= max_train_ts_sec);
            if valid_len < 4 {
                continue;
            }
            let win_cap = if ep.is_intraday_session {
                valid_len.min(84)
            } else {
                32usize.min(valid_len)
            };
            let stride = if ep.is_intraday_session {
                win_cap.max(1)
            } else {
                24usize
            };
            let mut w_st = 0usize;
            while w_st + 4 <= valid_len && flat_noise_eps.len() < 32 {
                let w_en = (w_st + win_cap).min(valid_len);
                let win_bars = &ep.bars[w_st..w_en];
                let ep_cf = compute_episode_counterfactual_targets(win_bars, 0.85);
                let mut tgts = vec![None; win_bars.len()];
                let mut noise_bars = 0usize;
                for (t, b) in win_bars.iter().enumerate() {
                    if b.force_flat {
                        tgts[t] = Some(0.0);
                        continue;
                    }
                    if !b.entry_allowed {
                        continue;
                    }
                    let rt_c = 2.0 * b.one_way_cost_shares;
                    let long_risk_adj = ep_cf.critic_targets[t][0] - 0.55 * ep_cf.path_mae[t][0];
                    let short_risk_adj = ep_cf.critic_targets[t][1] - 0.55 * ep_cf.path_mae[t][1];
                    let opt_risk_adj = if b.option_allowed {
                        ep_cf.critic_targets[t][2].max(ep_cf.critic_targets[t][3])
                            - 0.35 * ep_cf.path_mae[t][0].min(ep_cf.path_mae[t][1])
                    } else {
                        f32::NEG_INFINITY
                    };
                    let best_risk_adj = long_risk_adj.max(short_risk_adj).max(opt_risk_adj);
                    if ep_cf.best_entry_net[t] < -0.10 * rt_c || best_risk_adj <= 0.0 {
                        tgts[t] = Some(0.0);
                        noise_bars += 1;
                    }
                }
                if noise_bars >= 3 && noise_bars * 5 >= win_bars.len() * 2 {
                    flat_windows_found += 1;
                    flat_noise_eps.push((ep_i, w_st, w_en, tgts));
                }
                if w_en == valid_len {
                    break;
                }
                w_st += stride;
            }
            if flat_noise_eps.len() >= 32 {
                break;
            }
        }
        if flat_windows_found > 0 {
            self.autopsy_telemetry.flat_regret_windows_replayed += flat_windows_found;
        }

        if reds_f.is_empty()
            && reds_h.is_empty()
            && reds_v.is_empty()
            && causal_episodes.is_empty()
        {
            return;
        }

        let feat_mean = self.feat_mean.clone();
        let feat_var = self.feat_var.clone();
        let wm = &self.world_model;
        let seed_base =
            self.seed_base ^ self.online_updates_count.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let init_ema = self.autopsy_grad_ema;

        let sample_red = |bucket: &[AuditedRedTradeTrajectory], seed: &mut u64| -> Option<usize> {
            if bucket.is_empty() {
                return None;
            }
            let tot_w: f32 = bucket.iter().map(|r| r.priority_weight).sum();
            if tot_w <= 0.0 {
                return Some((xorshift64(seed) as usize) % bucket.len());
            }
            let u = rand_uniform(seed, 0.0, tot_w);
            let mut acc = 0.0_f32;
            for (i, r) in bucket.iter().enumerate() {
                acc += r.priority_weight;
                if acc >= u {
                    return Some(i);
                }
            }
            Some(bucket.len() - 1)
        };

        let member_emas: Vec<[f32; 4]> = self
            .population
            .par_iter_mut()
            .map(|member| {
                let mut seed =
                    seed_base ^ (member.id as u64 + 1).wrapping_mul(0x517C_C1B7_2722_0A95);
                let mut local_ema = init_ema;
                let mut combined = member.net.new_grad_accum();
                let mut v_acc = [
                    member.net.new_grad_accum(),
                    member.net.new_grad_accum(),
                    member.net.new_grad_accum(),
                    member.net.new_grad_accum(),
                ];

                for _ in 0..steps {
                    combined.zero();
                    for va in &mut v_acc {
                        va.zero();
                    }

                    // Vector 0: FrictionInversion
                    if let Some(ri) = sample_red(&reds_f, &mut seed) {
                        let chosen = &reds_f[ri];
                        if chosen.bars.len() >= 3 {
                            let (targets, v_tgts) = Self::build_autopsy_targets(chosen);
                            let _ = member.net.accumulate_bptt_chunk_full(
                                &mut v_acc[0],
                                &chosen.bars,
                                &chosen.prior_regime_z,
                                Some(wm),
                                &feat_mean,
                                &feat_var,
                                &member.hparams,
                                Some((&targets, chosen.priority_weight.min(12.0))),
                                v_tgts
                                    .as_deref()
                                    .map(|v| (v, chosen.priority_weight.min(12.0))),
                                None,
                            );
                        }
                    }

                    // Vector 1: HoldingEodDecay (conditioned on active held position trajectory!)
                    if let Some(ri) = sample_red(&reds_h, &mut seed) {
                        let chosen = &reds_h[ri];
                        if chosen.bars.len() >= 3 {
                            let (targets, v_tgts) = Self::build_autopsy_targets(chosen);
                            let fh = ForcedHeldContext {
                                entry_offset: chosen.entry_offset,
                                optimal_exit_offset: chosen.optimal_exit_offset,
                                direction_sign: chosen.direction_sign,
                                is_option_routed: chosen.is_option_routed,
                                peak_giveback_ret: (chosen.best_counterfactual_net_r.max(0.1) * 0.012)
                                    .clamp(0.003, 0.08),
                            };
                            let _ = member.net.accumulate_bptt_chunk_with_held_context(
                                &mut v_acc[1],
                                &chosen.bars,
                                &chosen.prior_regime_z,
                                Some(wm),
                                &feat_mean,
                                &feat_var,
                                &member.hparams,
                                Some((&targets, chosen.priority_weight.min(12.0))),
                                v_tgts
                                    .as_deref()
                                    .map(|v| (v, chosen.priority_weight.min(12.0))),
                                Some(fh),
                                None,
                            );
                        }
                    }

                    // Vector 2: VehicleSelectionRegret
                    if let Some(ri) = sample_red(&reds_v, &mut seed) {
                        let chosen = &reds_v[ri];
                        if chosen.bars.len() >= 3 {
                            let (targets, v_tgts) = Self::build_autopsy_targets(chosen);
                            let _ = member.net.accumulate_bptt_chunk_full(
                                &mut v_acc[2],
                                &chosen.bars,
                                &chosen.prior_regime_z,
                                Some(wm),
                                &feat_mean,
                                &feat_var,
                                &member.hparams,
                                Some((&targets, chosen.priority_weight.min(10.0))),
                                v_tgts
                                    .as_deref()
                                    .map(|v| (v, chosen.priority_weight.min(10.0))),
                                None,
                            );
                        }
                    }

                    // Vector 3: Active Counterfactual FLAT Noise Replay (`L_flat_counterfactual`)
                    if !flat_noise_eps.is_empty() {
                        let pick = (xorshift64(&mut seed) as usize) % flat_noise_eps.len();
                        let (ep_i, w_st, w_en, ref tgts) = flat_noise_eps[pick];
                        let ep = &causal_episodes[ep_i];
                        let win_bars = &ep.bars[w_st..w_en];
                        let max_c = member.hparams.chunk_len.max(24);
                        let (slice, tgt_slice) = if win_bars.len() <= max_c {
                            (win_bars, &tgts[..])
                        } else {
                            let st = (xorshift64(&mut seed) as usize) % (win_bars.len() - max_c + 1);
                            (&win_bars[st..st + max_c], &tgts[st..st + max_c])
                        };
                        let _ = member.net.accumulate_bptt_chunk_full(
                            &mut v_acc[3],
                            slice,
                            &ep.prior_regime_z,
                            Some(wm),
                            &feat_mean,
                            &feat_var,
                            &member.hparams,
                            Some((tgt_slice, 2.0)),
                            None,
                            None,
                        );
                    }

                    // 4-Vector GradNorm normalization by running causal EMA gradient norm
                    let mut active_vecs = 0usize;
                    for (vi, va) in v_acc.iter().enumerate() {
                        if va.chunks_count > 0 {
                            let g_norm = va.mean_l2_norm();
                            if g_norm > 1e-8 && g_norm.is_finite() {
                                local_ema[vi] =
                                    (0.85 * local_ema[vi] + 0.15 * g_norm).clamp(1e-5, 100.0);
                                active_vecs += 1;
                            }
                        }
                    }
                    let mean_ema = if active_vecs > 0 {
                        (local_ema.iter().sum::<f32>() / 4.0).clamp(1e-4, 10.0)
                    } else {
                        1.0
                    };
                    for (vi, va) in v_acc.iter().enumerate() {
                        if va.chunks_count > 0 {
                            combined.add_normalized_from(va, local_ema[vi], 0.30 * mean_ema);
                        }
                    }

                    // Anchor with 2 unfiltered causal episode BPTT trajectories so winners stay sharp
                    if !causal_episodes.is_empty() {
                        for _ in 0..2 {
                            let ep_idx = (xorshift64(&mut seed) as usize) % causal_episodes.len();
                            let ep = &causal_episodes[ep_idx];
                            if ep.bars.len() >= 4 {
                                let max_c = member.hparams.chunk_len.max(24);
                                let cap_c = if ep.is_intraday_session {
                                    max_c.max(78)
                                } else {
                                    max_c
                                };
                                let slice = if ep.bars.len() <= cap_c {
                                    &ep.bars[..]
                                } else {
                                    let max_start = ep.bars.len() - cap_c;
                                    let st = (xorshift64(&mut seed) as usize) % (max_start + 1);
                                    &ep.bars[st..st + cap_c]
                                };
                                let _ = member.net.accumulate_bptt_chunk_full(
                                    &mut combined,
                                    slice,
                                    &ep.prior_regime_z,
                                    Some(wm),
                                    &feat_mean,
                                    &feat_var,
                                    &member.hparams,
                                    None,
                                    None,
                                    None,
                                );
                            }
                        }
                    }

                    combined.step_hyperparams(&mut member.hparams, lr_scale);
                    member.net.apply_accumulated_adam(
                        &mut combined,
                        &member.hparams,
                        lr_scale,
                        ewc_lambda,
                    );
                }
                local_ema
            })
            .collect();

        if !member_emas.is_empty() {
            let mut avg_ema = [0.0_f32; 4];
            for e in &member_emas {
                for k in 0..4 {
                    avg_ema[k] += e[k];
                }
            }
            let inv_m = 1.0 / (member_emas.len() as f32);
            for k in 0..4 {
                self.autopsy_grad_ema[k] = (avg_ema[k] * inv_m).clamp(1e-5, 100.0);
            }
        }
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
        normalize_212_with_stats(x, &self.feat_mean, &self.feat_var)
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
        if episodes.is_empty() || epochs == 0 || !episodes.iter().any(|ep| ep.bars.len() >= 4) {
            return;
        }
        const MINI_BATCH_CHUNKS: usize = 12;
        let feat_mean = self.feat_mean.clone();
        let feat_var = self.feat_var.clone();
        let wm = &self.world_model;

        self.population.par_iter_mut().for_each(|member| {
            let chunk_len = member.hparams.chunk_len.max(24);
            let mut chunk_descs: Vec<(usize, usize, usize, [f32; WM_DIM])> = Vec::new();
            for (ep_i, ep) in episodes.iter().enumerate() {
                if ep.bars.len() < 4 {
                    continue;
                }
                if ep.bars.len() <= 96 {
                    chunk_descs.push((ep_i, 0, ep.bars.len(), ep.prior_regime_z));
                } else {
                    let mut z_cur = ep.prior_regime_z;
                    let mut start = 0usize;
                    while start + 6 <= ep.bars.len() {
                        let end = (start + chunk_len).min(ep.bars.len());
                        chunk_descs.push((ep_i, start, end, z_cur));
                        for b in &ep.bars[start..end] {
                            let x_norm =
                                normalize_212_with_stats(&b.x_212, &feat_mean, &feat_var);
                            z_cur = wm.forward_step(&x_norm, &z_cur).0;
                        }
                        start += chunk_len;
                    }
                }
            }
            let n_chunks = chunk_descs.len();
            if n_chunks == 0 {
                return;
            }
            let mut accum = member.net.new_grad_accum();
            for epoch in 0..epochs {
                accum.zero();
                let mut stride = ((n_chunks / 3).max(1)) | 1;
                while gcd_usize(stride, n_chunks) != 1 {
                    stride += 1;
                }
                let start_idx = (epoch * 7 + member.id * 13) % n_chunks;
                for step in 0..n_chunks {
                    let desc_idx = (start_idx + step * stride) % n_chunks;
                    let (ep_i, st, en, init_z) = chunk_descs[desc_idx];
                    let ep = &episodes[ep_i];
                    let _ = member.net.accumulate_bptt_chunk_full(
                        &mut accum,
                        &ep.bars[st..en],
                        &init_z,
                        Some(wm),
                        &feat_mean,
                        &feat_var,
                        &member.hparams,
                        None,
                        None,
                        None,
                    );
                    if accum.chunks_count >= MINI_BATCH_CHUNKS {
                        accum.step_hyperparams(&mut member.hparams, lr_scale);
                        member.net.apply_accumulated_adam(
                            &mut accum,
                            &member.hparams,
                            lr_scale * 1.25,
                            ewc_lambda,
                        );
                    }
                }
                if accum.chunks_count > 0 {
                    accum.step_hyperparams(&mut member.hparams, lr_scale);
                    member.net.apply_accumulated_adam(
                        &mut accum,
                        &member.hparams,
                        lr_scale * 1.25,
                        ewc_lambda,
                    );
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

        let min_q_trades =
            if self.population.iter().filter(|m| m.causal_trades >= 2).count() >= 2 {
                2usize
            } else {
                1usize
            };
        let qualified: Vec<&PopulationMember> = self
            .population
            .iter()
            .filter(|m| m.causal_trades >= min_q_trades)
            .collect();
        let has_eval_history = !qualified.is_empty();
        let positive_members: Vec<&PopulationMember> = qualified
            .iter()
            .copied()
            .filter(|m| m.causal_net_pnl > 0.0 && m.causal_pf >= 1.05)
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
        let top_k_idx = (self.population.len() / 6).max(3) - 1;
        let top_cutoff = ranked_scores
            .get(top_k_idx)
            .copied()
            .unwrap_or(f64::NEG_INFINITY);

        let mut weighted_action = 0.0_f64;
        let mut weighted_action_sq = 0.0_f64;
        let mut pos_action_sum = 0.0_f64;
        let mut neg_action_sum = 0.0_f64;
        let mut pos_sign_weight = 0.0_f64;
        let mut neg_sign_weight = 0.0_f64;
        let mut weighted_vehicle = 0.0_f64;
        let mut weighted_stop = 0.0_f64;
        let mut weighted_dz = 0.0_f64;
        let mut weighted_long_mean_bps = 0.0_f64;
        let mut weighted_long_mean_sq = 0.0_f64;
        let mut weighted_long_spread_bps = 0.0_f64;
        let mut weighted_short_mean_bps = 0.0_f64;
        let mut weighted_short_mean_sq = 0.0_f64;
        let mut weighted_short_spread_bps = 0.0_f64;
        let mut weighted_downside_av = 0.0_f64;
        let mut weighted_cost_sens = 0.0_f64;
        let mut weighted_causal_pf = 0.0_f64;
        let mut positive_weight_sum = 0.0_f64;
        let mut active_specialists = 0usize;

        for (i, member) in self.population.iter().enumerate() {
            let w = if !has_eval_history {
                member.hedge_weight
            } else if positive_members.len() >= 2 {
                if member.causal_trades >= min_q_trades
                    && member.causal_net_pnl > 0.0
                    && member.causal_pf >= 1.08
                    && member.causal_score >= top_cutoff
                {
                    member.hedge_weight
                } else if member.causal_trades >= min_q_trades
                    && member.causal_net_pnl > 0.0
                    && member.causal_pf >= 1.15
                {
                    member.hedge_weight * 0.30
                } else {
                    0.0
                }
            } else if positive_members.len() == 1 {
                if member.causal_trades >= min_q_trades
                    && member.causal_net_pnl > 0.0
                    && member.causal_pf >= 1.05
                {
                    member.hedge_weight
                } else if member.causal_trades >= min_q_trades && member.causal_score >= top_cutoff {
                    member.hedge_weight * 0.15
                } else {
                    0.0
                }
            } else if member.causal_trades >= min_q_trades && member.causal_score >= top_cutoff {
                member.hedge_weight
            } else if !qualified.is_empty() {
                0.0
            } else if member.causal_score >= top_cutoff {
                member.hedge_weight
            } else {
                0.0
            };

            if w <= 1e-8 {
                // Skip inactive routed member forward pass (`w == 0` contributes zero to ensemble!)
                next_hiddens.push([0.0_f32; HIDDEN_DIM]);
                continue;
            }

            let prev_h = prev_hiddens
                .get(i)
                .copied()
                .unwrap_or([0.0_f32; HIDDEN_DIM]);
            let step = member.net.forward_step_with_wm_cal(
                x_norm,
                z_wm,
                wm_pred,
                self_state,
                &prev_h,
                *z_wm,
                member.hparams.cost_sensitivity,
                member.hparams.downside_aversion,
                member.hparams.conformal_q_bps,
                option_allowed,
            );
            next_hiddens.push(step.next_hidden);

            if step.action.abs() > 1e-6 {
                active_specialists += 1;
                if step.action > 0.0 {
                    pos_sign_weight += w;
                    pos_action_sum += w * (step.action as f64);
                } else {
                    neg_sign_weight += w;
                    neg_action_sum += w * (step.action as f64);
                }
            }

            let (l_mean_sh, l_spr_sh, l_lcb_sh) =
                compute_critic_channel_lcb(&step.critic_all[0..5], member.hparams.downside_aversion);
            let (s_mean_sh, s_spr_sh, s_lcb_sh) =
                compute_critic_channel_lcb(&step.critic_all[5..10], member.hparams.downside_aversion);
            let (l_mean_best, l_spr_best) = if option_allowed {
                let (lm_o, ls_o, ll_o) = compute_critic_channel_lcb(
                    &step.critic_all[10..15],
                    member.hparams.downside_aversion,
                );
                if ll_o > l_lcb_sh {
                    (lm_o, ls_o)
                } else {
                    (l_mean_sh, l_spr_sh)
                }
            } else {
                (l_mean_sh, l_spr_sh)
            };
            let (s_mean_best, s_spr_best) = if option_allowed {
                let (sm_o, ss_o, sl_o) = compute_critic_channel_lcb(
                    &step.critic_all[15..20],
                    member.hparams.downside_aversion,
                );
                if sl_o > s_lcb_sh {
                    (sm_o, ss_o)
                } else {
                    (s_mean_sh, s_spr_sh)
                }
            } else {
                (s_mean_sh, s_spr_sh)
            };

            let lm_bps = (l_mean_best * 10_000.0) as f64;
            let sm_bps = (s_mean_best * 10_000.0) as f64;
            weighted_long_mean_bps += w * lm_bps;
            weighted_long_mean_sq += w * lm_bps * lm_bps;
            weighted_long_spread_bps += w * ((l_spr_best * 10_000.0) as f64);
            weighted_short_mean_bps += w * sm_bps;
            weighted_short_mean_sq += w * sm_bps * sm_bps;
            weighted_short_spread_bps += w * ((s_spr_best * 10_000.0) as f64);
            weighted_downside_av += w * (member.hparams.downside_aversion as f64);
            weighted_cost_sens += w * (member.hparams.cost_sensitivity.max(0.02) as f64);
            weighted_causal_pf += w * member.causal_pf.clamp(0.0, 4.0);

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
        let turn_clamped = self_state.ema_turnover.clamp(0.0, 1.5);
        if positive_weight_sum > 1e-9 {
            let rt_cost_bps = (2.0 * self_state.one_way_cost_bps).max(4.0);
            let avg_down_av = (weighted_downside_av / positive_weight_sum) as f32;
            let avg_cs = (weighted_cost_sens / positive_weight_sum) as f32;
            let avg_pf = if has_eval_history {
                (weighted_causal_pf / positive_weight_sum) as f32
            } else {
                1.10
            };
            let cal_shortfall = (1.05 - avg_pf).max(0.0);
            let lambda_aleatoric =
                0.075 + 0.02 * avg_down_av.clamp(0.3, 6.0) + 0.06 * cal_shortfall;
            let lambda_epistemic = 0.20 + 0.45 * cal_shortfall;
            let has_wm_signal = wm_pred.iter().any(|v| v.abs() > 1e-8);

            if !is_holding {
                let active_w = pos_sign_weight + neg_sign_weight;
                if active_w <= 1e-9 || agg_action.abs() <= 1e-6 {
                    agg_action = 0.0;
                } else {
                    let dir_consensus = (((pos_sign_weight - neg_sign_weight).abs() / active_w)
                        as f32)
                        .clamp(0.0, 1.0);
                    let coverage = ((active_w / positive_weight_sum) as f32).clamp(0.0, 1.0);
                    let active_mean = (weighted_action / active_w) as f32;
                    let var_active =
                        ((weighted_action_sq / active_w) - (active_mean as f64).powi(2)).max(0.0);
                    let std_active = var_active.sqrt() as f32;

                    // Distributional + Epistemic Ensemble LCB coupled with unconstrained `k_theta` & `ema_turnover`:
                    let (dir_mean_bps, dir_spread_bps, dir_epistemic_std_bps) =
                        if agg_action >= 0.0 {
                            let m = weighted_long_mean_bps / positive_weight_sum;
                            let s = weighted_long_spread_bps / positive_weight_sum;
                            let v =
                                ((weighted_long_mean_sq / positive_weight_sum) - m * m).max(0.0);
                            (m as f32, s as f32, v.sqrt() as f32)
                        } else {
                            let m = weighted_short_mean_bps / positive_weight_sum;
                            let s = weighted_short_spread_bps / positive_weight_sum;
                            let v =
                                ((weighted_short_mean_sq / positive_weight_sum) - m * m).max(0.0);
                            (m as f32, s as f32, v.sqrt() as f32)
                        };
                    let wm_cal_lcb_bps = wm_directional_lcb_bps(
                        wm_pred,
                        agg_action.signum(),
                        rt_cost_bps,
                        avg_down_av,
                    ) - self.conformal_q_bps;
                    let ens_critic_lcb_bps = dir_mean_bps
                        - lambda_aleatoric * dir_spread_bps
                        - lambda_epistemic * dir_epistemic_std_bps;
                    let lcb_net_bps = if has_wm_signal {
                        0.58 * ens_critic_lcb_bps + 0.42 * wm_cal_lcb_bps
                    } else {
                        ens_critic_lcb_bps - 0.42 * self.conformal_q_bps
                    } - 0.45 * cal_shortfall * rt_cost_bps;

                    let hurdle_turn_bps = rt_cost_bps
                        * (0.14 * (avg_cs - 0.75).max(0.0) + 0.72 * turn_clamped);
                    let q_entry_minus_flat_bps = lcb_net_bps - hurdle_turn_bps;

                    if coverage < 0.08 || dir_consensus < 0.25 {
                        agg_action = 0.0;
                    } else {
                        let scale_bps = rt_cost_bps.clamp(8.0, 28.0);
                        let net_ratio =
                            (q_entry_minus_flat_bps / scale_bps).clamp(-4.0, 4.0);
                        let lcb_taper = ((net_ratio + 0.72) / 0.72).clamp(0.0, 1.0);
                        let soft_conf = sigmoid(1.95 * (net_ratio + 0.42)) * lcb_taper;
                        let support = 0.50 + 0.50 * coverage.powf(0.42);
                        let blended = active_mean * support;
                        let lcb_mag = (blended.abs() - 0.09 * std_active).max(0.0)
                            * dir_consensus
                            * soft_conf;
                        agg_action = if lcb_mag <= 1e-5 {
                            0.0
                        } else {
                            blended.signum() * lcb_mag
                        };
                    }
                }
            } else {
                // Smooth active same-sign aggregation when holding: trained specialists and
                // differentiable `tau_hold` drive exits without hard LCB zeroing overrides.
                let pos_sgn = self_state.current_pos.signum();
                let (same_w, same_act_sum) = if pos_sgn >= 0.0 {
                    (pos_sign_weight, pos_action_sum)
                } else {
                    (neg_sign_weight, neg_action_sum)
                };
                if same_w > 1e-9 && agg_action * pos_sgn >= 0.0 {
                    let same_cov = ((same_w / positive_weight_sum) as f32).clamp(0.0, 1.0);
                    let same_active_mean = (same_act_sum / same_w) as f32;
                    agg_action = 0.52 * agg_action + 0.48 * same_active_mean * same_cov.sqrt();
                }
            }
        }

        // Responsive Neural-Exit snap when holding without prematurely whipsawing healthy trends!
        let flat_snap = if !is_holding {
            0.070
        } else {
            (0.22 * self_state.current_pos.abs()).clamp(0.028, 0.075)
        };
        if agg_action.abs() < flat_snap {
            agg_action = 0.0;
        }

        EnsembleInferenceOutput {
            action: agg_action.clamp(-1.0, 1.0),
            vehicle_prob: agg_vehicle.clamp(0.0, 1.0),
            tail_stop_dist: agg_stop.clamp(0.015, 0.16),
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
        let opt_theta = if opt_ok { 0.0005_f32 } else { 0.0_f32 };
        let opt_gamma = if opt_ok { 1.30_f32 } else { 1.0_f32 };
        let opt_c = cost_s * 3.0;
        let cf_l_sh = ret_s - 2.0 * cost_s;
        let cf_s_sh = -ret_s - 2.0 * cost_s;
        let cf_l_opt = if opt_ok {
            ret_s * opt_gamma - opt_theta - 2.0 * opt_c
        } else {
            cf_l_sh
        };
        let cf_s_opt = if opt_ok {
            -ret_s * opt_gamma - opt_theta - 2.0 * opt_c
        } else {
            cf_s_sh
        };
        CausalTrajectoryBar {
            ts_sec: 1_700_000_000 + t * 300,
            x_212: x,
            ret_oo_shares: ret_s,
            ret_oo_option: ret_s * opt_gamma - opt_theta,
            one_way_cost_shares: cost_s,
            one_way_cost_option: opt_c,
            one_way_cost_bps: cost_bps,
            session_time_left: (1.0 - (t as f32 / 40.0)).clamp(0.0, 1.0),
            option_allowed: opt_ok,
            entry_allowed: !force_flat,
            force_flat,
            option_gamma_boost: opt_gamma,
            option_theta_per_bar: opt_theta,
            cf_net_returns: [cf_l_sh, cf_s_sh, cf_l_opt, cf_s_opt],
            cf_valid: true,
            intrabar_mae: 0.0,
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
        assert_eq!(audit.fault_kind, LossFaultKind::FrictionInversion);
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
            log_cost_sens: 1.5_f32.ln(),
            ret_scale: 0.38,
            timescale_shift: 0.0,
            horizon_gamma: 0.85,
            explore_pressure: 0.25,
            conformal_q_bps: 0.0,
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

    #[test]
    fn test_v_liq_telescoping_identity_and_put_theta_sign() {
        // Verify Mark-to-Liquidation step reward telescopes identically to Gross - EntryCost - ExitCost,
        // induces zero exit shock on a_t -> 0, and charges positive theta decay on short options too.
        let actions = [0.8_f32, 0.8, 0.0, -0.6, 0.0];
        let rets = [0.004_f32, 0.002, 0.001, -0.005, 0.002];
        let c_eff = 0.001_f32;
        let gamma = 1.25_f32;
        let theta = 0.0004_f32;
        let mut prev_a = 0.0_f32;
        let mut prev_res = 0.0_f32;
        let mut sum_mtm = 0.0_f32;
        let mut sum_liq = 0.0_f32;
        let mut exit_liq_long = 0.0_f32;
        for t in 0..actions.len() {
            let a = actions[t];
            let force_flat = t + 1 == actions.len();
            let r_gross = a * (rets[t] * gamma) - a.abs() * theta;
            let r_mtm = r_gross - (a - prev_a).abs() * c_eff;
            let cur_res = if force_flat { 0.0 } else { a.abs() * c_eff };
            let r_liq = r_mtm - (cur_res - prev_res);
            if t == 2 {
                exit_liq_long = r_liq;
            }
            // Verify Put option bar t=3 still incurs negative `-|a|*theta` rather than flipping sign!
            if t == 3 {
                assert!(r_gross < a * (rets[t] * gamma));
            }
            sum_mtm += r_mtm;
            sum_liq += r_liq;
            prev_a = a;
            prev_res = cur_res;
        }
        assert!(
            exit_liq_long.abs() < 1e-7,
            "Exiting a_t -> 0 under V_liq must produce zero exit-step penalty (got {exit_liq_long})"
        );
        assert!(
            (sum_mtm - sum_liq).abs() < 1e-6,
            "Telescoping V_liq sum ({sum_liq}) must equal MTM net PnL ({sum_mtm})"
        );
    }

    #[test]
    fn test_4_channel_critic_and_selective_leaky_deadzone() {
        let mut brain = PureAutonomousRlBrain::new_for_horizon("TEST_SELECTIVE_LEAKY", 77);
        for m in &mut brain.population {
            m.net.w_a_heads.fill(0.0);
            m.net.b_a_heads.fill(0.0);
        }
        // Negative-expectancy noise sequence: all 4 counterfactual channels strictly negative -> ZERO wake drift!
        let mut neg_seq = Vec::new();
        for t in 0..16 {
            neg_seq.push(make_bar(t, 0.2, 0.0001, 0.0035, 35.0, false, t == 15));
        }
        let mut net_neg = brain.population[0].net.clone();
        let hp = brain.population[0].hparams.clone();
        let mean = vec![0.0_f32; RL_STATE_DIM];
        let var = vec![1.0_f32; RL_STATE_DIM];
        let w_before = net_neg.w_a_heads.clone();
        for _ in 0..5 {
            net_neg.train_bptt_chunk(&neg_seq, None, &mean, &var, &hp, 1.0, 0.0);
        }
        let drift_neg: f32 = w_before
            .iter()
            .zip(net_neg.w_a_heads.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            drift_neg < 1e-5,
            "Inside deadzone when FLAT is optimal, selective leaky surrogate must not wake zero weights (drift={drift_neg})"
        );
        // Critic must have learned across all 4x5=20 outputs
        let step_out = net_neg.forward_step(
            &neg_seq[0].x_212,
            SelfAwarenessState::default(),
            &[0.0; HIDDEN_DIM],
            hp.cost_sensitivity,
            false,
        );
        assert_eq!(step_out.critic_all.len(), CRITIC_OUT_DIM);
        assert_eq!(step_out.critic_channel_utils.len(), CRITIC_CHANNELS);
    }

    #[test]
    fn test_4_vector_autopsy_attribution_and_paralysis_wake() {
        let mut brain = PureAutonomousRlBrain::new_for_horizon("TEST_4VEC_AUTOPSY", 333);
        // 1. HoldingEodDecay: rises strongly then reverses red before EOD
        let mut decay_bars = Vec::new();
        for t in 0..12 {
            let ret = if t < 5 { 0.006 } else { -0.007 };
            decay_bars.push(make_bar(t, 0.8, ret, 0.0005, 5.0, false, t == 11));
        }
        let audit_decay = brain.audit_and_insert_loss_event(
            "SPY".to_string(),
            decay_bars.last().unwrap().ts_sec,
            -120.0,
            -90.0,
            -0.8,
            1.0,
            false,
            1,
            [0.0; WM_DIM],
            decay_bars,
        );
        assert_eq!(audit_decay.fault_kind, LossFaultKind::HoldingEodDecay);

        // 2. DeadzoneParalysisRegret: zeroed heads + positive opportunity episode must wake & log telemetry
        for m in &mut brain.population {
            m.net.w_a_heads.fill(0.0);
            m.net.b_a_heads.fill(0.0);
        }
        let mut opp_bars = Vec::new();
        for t in 0..20 {
            opp_bars.push(make_bar(t, 1.4, 0.005, 0.0005, 5.0, true, t == 19));
        }
        let cs_before = brain.population[0].hparams.cost_sensitivity;
        let ep = CausalSessionEpisode {
            symbol: "QQQ".to_string(),
            day_key: 19700,
            start_ts_sec: opp_bars[0].ts_sec,
            end_ts_sec: opp_bars[19].ts_sec,
            is_intraday_session: true,
            prior_regime_z: [0.0; WM_DIM],
            bars: opp_bars,
        };
        let woken = brain.train_deadzone_paralysis_wake(std::slice::from_ref(&ep), 4, 1.0);
        assert!(woken > 0);
        assert!(brain.autopsy_telemetry.deadzone_paralysis_events >= 1);
        assert!(brain.population[0].hparams.cost_sensitivity < cs_before);
    }

    #[test]
    fn test_nonzero_dl_dx_across_all_six_brains_and_unbounded_cost_sens() {
        assert_eq!(
            HIDDEN_DIM, 64,
            "Multi-timescale GRU latent capacity D must be 64"
        );
        let mut brains = PureAutonomousRlBrain::default_horizon_suite();
        assert_eq!(brains.len(), 6);

        let mut f_bars = Vec::new();
        let mut h_bars = Vec::new();
        let mut v_bars = Vec::new();
        for t in 0..16 {
            f_bars.push(make_bar(t, 0.5, -0.0025, 0.0035, 35.0, false, t == 15));
            let h_ret = if t < 6 { 0.0065 } else { -0.0075 };
            h_bars.push(make_bar(t, 0.9, h_ret, 0.0005, 5.0, false, t == 15));
            v_bars.push(make_bar(t, 1.2, 0.006, 0.0005, 5.0, true, t == 15));
        }

        for brain in &mut brains {
            let xs: Vec<[f32; RL_STATE_DIM]> = f_bars.iter().map(|b| b.x_212).collect();
            brain.update_welford_batch(&xs);

            // 1. FrictionInversion saliency must have |dL/dx| > 0 and |dL/dh| > 0
            let af = brain.audit_and_insert_loss_event(
                "SOUN".to_string(),
                f_bars.last().unwrap().ts_sec,
                -180.0,
                -110.0,
                -1.1,
                1.0,
                false,
                2,
                [0.0; WM_DIM],
                f_bars.clone(),
            );
            assert!(
                af.saliency_dx > 1e-6 && af.saliency_dh > 1e-6,
                "{} FrictionInversion must produce |dL/dx|>0 ({}) and |dL/dh|>0 ({})",
                brain.horizon_label,
                af.saliency_dx,
                af.saliency_dh
            );

            // 2. HoldingEodDecay saliency must have |dL/dx| > 0 and |dL/dh| > 0
            let ah = brain.audit_and_insert_loss_event(
                "SPY".to_string(),
                h_bars.last().unwrap().ts_sec + 60,
                -95.0,
                -70.0,
                -0.7,
                1.0,
                false,
                1,
                [0.0; WM_DIM],
                h_bars.clone(),
            );
            assert_eq!(ah.fault_kind, LossFaultKind::HoldingEodDecay);
            assert!(
                ah.saliency_dx > 1e-6 && ah.saliency_dh > 1e-6,
                "{} HoldingEodDecay must produce |dL/dx|>0 ({}) and |dL/dh|>0 ({})",
                brain.horizon_label,
                ah.saliency_dx,
                ah.saliency_dh
            );

            // 3. VehicleSelectionRegret saliency must have |dL/dx| > 0 and |dL/dh| > 0
            let av = brain.audit_and_insert_loss_event(
                "NVDA".to_string(),
                v_bars.last().unwrap().ts_sec + 120,
                -140.0,
                -60.0,
                -0.9,
                1.0,
                true,
                1,
                [0.0; WM_DIM],
                v_bars.clone(),
            );
            assert!(
                av.saliency_dx > 1e-6 && av.saliency_dh > 1e-6,
                "{} VehicleSelectionRegret must produce |dL/dx|>0 ({}) and |dL/dh|>0 ({})",
                brain.horizon_label,
                av.saliency_dx,
                av.saliency_dh
            );

            // 4. Unbounded log-space cost_sensitivity must cross past legacy 2.60 ceiling freely
            for _ in 0..14 {
                brain.adapt_cost_sensitivity_from_autopsy_cycle(10, 2, 10);
            }
            assert!(
                brain.autopsy_telemetry.cost_sensitivity_after > 3.0,
                "{} cost_sensitivity must adapt freely past 2.60 ceiling in log-space (got {})",
                brain.horizon_label,
                brain.autopsy_telemetry.cost_sensitivity_after
            );
        }
    }

    #[test]
    fn test_turnover_amortized_q_hold_vs_q_flat_and_hypergrad() {
        // 1. Holding with mild positive continuation vs negative Q_flat (re-entry friction loss) keeps tau_hold low
        let (tau_hold_pos, _, dtau_dlogk_hold) = compute_cost_scaled_deadzone(
            -0.75, 10.0, 1.5, 4.0, 3.0, true, 0.70, 0.35, 0.004, 0.0, 1.2,
        );
        // Exiting into flat when continuation goes negative and giveback rises expands tau_hold to trigger clean exit
        let (tau_hold_neg, _, _) = compute_cost_scaled_deadzone(
            -0.75, 10.0, 1.5, -25.0, -22.0, true, 0.70, 0.35, 0.008, 0.006, 2.0,
        );
        assert!(
            tau_hold_neg > tau_hold_pos + 0.08,
            "Negative Q_hold - Q_flat must expand exit deadzone ({tau_hold_neg} vs {tau_hold_pos})"
        );
        assert!(dtau_dlogk_hold.abs() < 1e-8);

        // 2. Flat entry with high recent ema_turnover faces elevated epistemic LCB hurdle & positive d(tau)/d(log_k)
        let (tau_flat_calm, _, dtau_dlogk_calm) = compute_cost_scaled_deadzone(
            -0.75, 10.0, 1.8, 12.0, 10.0, false, 0.75, 0.0, 0.0, 0.0, 0.0,
        );
        let (tau_flat_churn, _, dtau_dlogk_churn) = compute_cost_scaled_deadzone(
            -0.75, 10.0, 1.8, 12.0, 10.0, false, 0.75, 0.85, 0.0, 0.0, 0.0,
        );
        assert!(
            tau_flat_churn > tau_flat_calm + 0.05,
            "High ema_turnover must elevate flat entry hurdle ({tau_flat_churn} vs {tau_flat_calm})"
        );
        assert!(
            dtau_dlogk_calm > 0.0 && dtau_dlogk_churn > 0.0,
            "Analytical hypergradient d(tau)/d(log_k_theta) must be strictly positive when flat ({dtau_dlogk_calm}, {dtau_dlogk_churn})"
        );
    }

    #[test]
    fn test_paralysis_regret_nonzero_saliency_at_zero_realized_trades() {
        let mut brain = PureAutonomousRlBrain::new_for_horizon("SCALP_0DTE_1M", 0x9911);
        // Paralyze all specialist action heads (`a_t == 0.0` everywhere -> 0 realized trades)
        for m in &mut brain.population {
            m.net.w_a_heads.fill(0.0);
            m.net.b_a_heads.fill(0.0);
        }
        let mut bars = Vec::new();
        for t in 0..18 {
            let ret = if (2..10).contains(&t) {
                0.0045
            } else {
                -0.0003
            };
            bars.push(make_bar(t, 1.35, ret, 0.0005, 5.0, true, t == 17));
        }
        let xs: Vec<[f32; RL_STATE_DIM]> = bars.iter().map(|b| b.x_212).collect();
        brain.update_welford_batch(&xs);

        // Confirm live ensemble emits exact `0.0` action (`0` realized trades) across the entire episode
        let mut wm_z = [0.0_f32; WM_DIM];
        let mut hids = vec![[0.0_f32; HIDDEN_DIM]; brain.population.len()];
        let mut realized_entries = 0usize;
        for b in &bars {
            let x_norm = brain.normalize_x(&b.x_212);
            let (nz, wm_pred) = brain.world_model.forward_step(&x_norm, &wm_z);
            wm_z = nz;
            let out = brain.infer_ensemble_step_with_precomputed_wm(
                &x_norm,
                &wm_z,
                &wm_pred,
                SelfAwarenessState {
                    one_way_cost_bps: b.one_way_cost_bps,
                    session_time_left: b.session_time_left,
                    ..Default::default()
                },
                &hids,
                b.option_allowed,
            );
            hids = out.next_hiddens;
            if out.action.abs() > 1e-6 {
                realized_entries += 1;
            }
        }
        assert_eq!(realized_entries, 0, "Dormant policy must start at 0 realized trades");

        let ep = CausalSessionEpisode {
            symbol: "SPY".to_string(),
            day_key: 19800,
            start_ts_sec: bars[0].ts_sec,
            end_ts_sec: bars.last().unwrap().ts_sec,
            is_intraday_session: true,
            prior_regime_z: [0.0; WM_DIM],
            bars,
        };

        let (events, sal_dx, sal_dh) =
            brain.audit_and_train_paralysis_regret(std::slice::from_ref(&ep), i64::MAX, 3, 0.65);
        assert!(events >= 1, "Must detect Paralysis Regret episode at 0 trades");
        assert!(
            brain.autopsy_telemetry.deadzone_paralysis_events >= 1,
            "Must log Paralysis Regret telemetry event"
        );
        assert!(
            sal_dx > 1e-6 && sal_dh > 1e-6,
            "Paralysis Regret at 0 realized trades must produce strictly positive |dL/dx| ({sal_dx}) and |dL/dh| ({sal_dh})"
        );
    }
}
