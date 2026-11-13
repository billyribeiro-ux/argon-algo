//! Phase 35 Deliverable 1 & 2:
//! 1. Cross-Position Eigen-Covariance & Absorption-Ratio Crowding Gate (`PortfolioEigenCrowdingEngine`)
//! 2. Regime-Contextual Normal-Inverse-Gamma (NIG) Cross-Brain Capital Meta-Allocator (`ContextualNigMetaAllocator`)

use crate::autopsy_evolver::RegimeBucket;
use crate::ledger::{ClosedTradeRow, OpenPositionRow};
use crate::regime_bocpd::{
    compute_adaptive_forgetting_gamma, AdamsMacKayBocpdEngine, BocpdStepTelemetry,
};
use raw_fingerprint_core::BrainHorizonId;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// General exact KL Bregman projection onto the capped $K$-simplex
/// $\{w \in [\text{floor\_w}, \text{ceil\_w}]^K : \sum_{i=0}^{K-1} w_i = \text{total\_mass}\}$
/// via deterministic bisection on the multiplicative dual multiplier $\lambda > 0$.
pub fn project_capped_simplex_kl(
    unnormalized: &[f64],
    floor_w: f64,
    ceil_w: f64,
    total_mass: f64,
) -> Vec<f64> {
    let k = unnormalized.len();
    if k == 0 {
        return Vec::new();
    }
    let safe_u: Vec<f64> = unnormalized.iter().map(|&v| v.max(1e-12)).collect();

    let mut log_lo = -35.0_f64;
    let mut log_hi = 35.0_f64;
    for _ in 0..80 {
        let log_mid = 0.5 * (log_lo + log_hi);
        let scale = log_mid.exp();
        let s: f64 = safe_u
            .iter()
            .map(|&u| (u * scale).clamp(floor_w, ceil_w))
            .sum();
        if s < total_mass {
            log_lo = log_mid;
        } else {
            log_hi = log_mid;
        }
    }

    let opt_scale = (0.5 * (log_lo + log_hi)).exp();
    safe_u
        .into_iter()
        .map(|u| (u * opt_scale).clamp(floor_w, ceil_w))
        .collect()
}

/// Directional leg series passed into `PortfolioEigenCrowdingEngine`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EigenPositionSeries {
    pub symbol: String,
    /// `+1.0` for Long, `-1.0` for Short.
    pub direction_sign: f64,
    /// Risk capital weight $w_i > 0$ (e.g. `risk_budget_usd` or `1.0`).
    pub risk_weight: f64,
    /// Causal rolling bar close series ordered chronologically ($P_0, P_1, \dots, P_T$).
    pub close_prices: Vec<f64>,
}

impl EigenPositionSeries {
    pub fn from_open_row(row: &OpenPositionRow) -> Self {
        let close_prices = if row.entry_bars.len() >= 4 {
            row.entry_bars.iter().map(|b| b.close.to_f64()).collect()
        } else {
            row.price_sparkline.clone()
        };
        Self {
            symbol: row.symbol.clone(),
            direction_sign: row.direction.sign(),
            risk_weight: row.risk_budget_usd.max(1.0),
            close_prices,
        }
    }
}

/// Book-level and candidate-level Eigen-Correlation & Crowding telemetry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PortfolioEigenRiskReport {
    pub total_legs_k: usize,
    pub open_positions_count: usize,
    pub shared_return_bars_t: usize,
    pub ledoit_wolf_shrinkage_delta: f64,
    pub top_eigenvalue_lambda_max: f64,
    pub absorption_ratio: f64,
    pub effective_independent_bets_neff: f64,
    pub marginal_correlation_penalty: f64,
    pub eigen_kelly_multiplier: f64,
    pub gross_risk_heat_fraction: f64,
    pub vetoed: bool,
    pub veto_reason: String,
}

impl Default for PortfolioEigenRiskReport {
    fn default() -> Self {
        Self {
            total_legs_k: 0,
            open_positions_count: 0,
            shared_return_bars_t: 0,
            ledoit_wolf_shrinkage_delta: 0.0,
            top_eigenvalue_lambda_max: 1.0,
            absorption_ratio: 1.0,
            effective_independent_bets_neff: 1.0,
            marginal_correlation_penalty: 0.0,
            eigen_kelly_multiplier: 1.0,
            gross_risk_heat_fraction: 0.0,
            vetoed: false,
            veto_reason: "PASS".to_string(),
        }
    }
}

/// Cross-Position Eigen-Covariance & Absorption-Ratio Crowding Engine.
pub struct PortfolioEigenCrowdingEngine;

impl PortfolioEigenCrowdingEngine {
    /// Extracts trailing causal log-returns $r_t = \ln(P_t / P_{t-1})$ from a close price series.
    fn causal_log_returns(closes: &[f64]) -> Vec<f64> {
        if closes.len() < 2 {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(closes.len() - 1);
        for w in closes.windows(2) {
            let p0 = w[0].max(1e-8);
            let p1 = w[1].max(1e-8);
            out.push((p1 / p0).ln());
        }
        out
    }

    /// Computes causal rolling signed log-return correlation matrix $C \in \mathbb{R}^{K \times K}$
    /// with Ledoit-Wolf shrinkage toward $I_K$ when shared return history $T < 30$:
    /// $$C = (1 - \delta_{\text{LW}}) R + \delta_{\text{LW}} I_K$$
    #[allow(clippy::needless_range_loop)]
    pub fn build_shrunk_signed_correlation_matrix(
        legs: &[EigenPositionSeries],
    ) -> (Vec<Vec<f64>>, usize, f64) {
        let k = legs.len();
        if k == 0 {
            return (Vec::new(), 0, 0.0);
        }
        if k == 1 {
            return (vec![vec![1.0]], legs[0].close_prices.len().saturating_sub(1), 0.0);
        }

        let raw_rets: Vec<Vec<f64>> = legs
            .iter()
            .map(|leg| Self::causal_log_returns(&leg.close_prices))
            .collect();

        let shared_t = raw_rets.iter().map(Vec::len).min().unwrap_or(0);
        let delta_lw = if shared_t >= 30 {
            0.0
        } else {
            (((30usize.saturating_sub(shared_t)) as f64) / 30.0).clamp(0.05, 0.85)
        };

        let mut c_mat = vec![vec![0.0_f64; k]; k];
        for i in 0..k {
            c_mat[i][i] = 1.0;
        }

        if shared_t < 2 {
            for i in 0..k {
                for j in (i + 1)..k {
                    if legs[i].symbol.eq_ignore_ascii_case(&legs[j].symbol) {
                        let signed_same =
                            (1.0 - delta_lw) * legs[i].direction_sign * legs[j].direction_sign;
                        c_mat[i][j] = signed_same;
                        c_mat[j][i] = signed_same;
                    }
                }
            }
            return (c_mat, shared_t, delta_lw);
        }

        let mut centered: Vec<Vec<f64>> = Vec::with_capacity(k);
        let mut norms: Vec<f64> = Vec::with_capacity(k);

        for i in 0..k {
            let r_slice = &raw_rets[i][raw_rets[i].len() - shared_t..];
            let dir = if legs[i].direction_sign >= 0.0 {
                1.0
            } else {
                -1.0
            };
            let mean: f64 = r_slice.iter().map(|&v| dir * v).sum::<f64>() / (shared_t as f64);
            let mut c_vec = Vec::with_capacity(shared_t);
            let mut sq = 0.0_f64;
            for &v in r_slice {
                let dv = dir * v - mean;
                sq += dv * dv;
                c_vec.push(dv);
            }
            centered.push(c_vec);
            norms.push(sq.sqrt());
        }

        for i in 0..k {
            for j in (i + 1)..k {
                let raw_corr = if legs[i].symbol.eq_ignore_ascii_case(&legs[j].symbol) {
                    legs[i].direction_sign.signum() * legs[j].direction_sign.signum()
                } else if norms[i] > 1e-12 && norms[j] > 1e-12 {
                    let dot: f64 = centered[i]
                        .iter()
                        .zip(centered[j].iter())
                        .map(|(&a, &b)| a * b)
                        .sum();
                    (dot / (norms[i] * norms[j])).clamp(-1.0, 1.0)
                } else {
                    0.0
                };
                let shrunk = ((1.0 - delta_lw) * raw_corr).clamp(-0.999, 0.999);
                c_mat[i][j] = shrunk;
                c_mat[j][i] = shrunk;
            }
        }

        (c_mat, shared_t, delta_lw)
    }

    /// Deterministic shifted power iteration computing $\lambda_{\max}(C) \in [1.0, K]$
    /// on $M = C + I_K \succeq 0$.
    #[allow(clippy::needless_range_loop)]
    pub fn top_eigenvalue_power_iteration(c_mat: &[Vec<f64>]) -> f64 {
        let k = c_mat.len();
        if k <= 1 {
            return 1.0;
        }
        let inv_sqrt_k = 1.0 / (k as f64).sqrt();
        let mut v = vec![inv_sqrt_k; k];

        // Deterministic symmetry breaker if initial vector lies near a balanced hedge nullspace
        let mut init_norm_sq = 0.0_f64;
        for i in 0..k {
            let mut s = v[i];
            for j in 0..k {
                s += c_mat[i][j] * v[j];
            }
            init_norm_sq += s * s;
        }
        if init_norm_sq < 1e-9 {
            v.fill(0.0);
            v[0] = 1.0;
        }

        let mut next_v = vec![0.0_f64; k];
        for _ in 0..64 {
            let mut norm_sq = 0.0_f64;
            for i in 0..k {
                let mut mv_i = v[i]; // +1.0 * I_K shift ensures strict non-negativity of spectrum
                for j in 0..k {
                    mv_i += c_mat[i][j] * v[j];
                }
                next_v[i] = mv_i;
                norm_sq += mv_i * mv_i;
            }
            let norm = norm_sq.sqrt().max(1e-12);
            for i in 0..k {
                v[i] = next_v[i] / norm;
            }
        }

        let mut rayleigh = 0.0_f64;
        for i in 0..k {
            let mut cv_i = 0.0_f64;
            for j in 0..k {
                cv_i += c_mat[i][j] * v[j];
            }
            rayleigh += v[i] * cv_i;
        }
        rayleigh.clamp(1.0, k as f64)
    }

    /// Evaluates candidate setup crowding against `open_legs` and `gross_risk_heat_fraction`.
    /// When `candidate` is `None`, computes pure book-level Eigen-Risk telemetry over `open_legs`.
    #[allow(clippy::needless_range_loop)]
    pub fn evaluate_portfolio_eigen_crowding(
        open_legs: &[EigenPositionSeries],
        candidate: Option<&EigenPositionSeries>,
        gross_risk_heat_fraction: f64,
    ) -> PortfolioEigenRiskReport {
        let k_open = open_legs.len();
        let mut all_legs: Vec<EigenPositionSeries> = open_legs.to_vec();
        if let Some(cand) = candidate {
            all_legs.push(cand.clone());
        }
        let k_total = all_legs.len();
        if k_total == 0 {
            return PortfolioEigenRiskReport {
                gross_risk_heat_fraction,
                ..PortfolioEigenRiskReport::default()
            };
        }

        let (c_mat, shared_t, delta_lw) = Self::build_shrunk_signed_correlation_matrix(&all_legs);
        let lambda_max = Self::top_eigenvalue_power_iteration(&c_mat);
        let ar_ratio = (lambda_max / (k_total as f64)).clamp(1.0 / (k_total as f64), 1.0);

        let weights: Vec<f64> = all_legs.iter().map(|l| l.risk_weight.max(1e-6)).collect();
        let sum_w: f64 = weights.iter().sum();
        let sum_w_sq: f64 = weights.iter().map(|&w| w * w).sum();
        let mut quad_form = 0.0_f64;
        for i in 0..k_total {
            for j in 0..k_total {
                quad_form += weights[i] * c_mat[i][j] * weights[j];
            }
        }
        // Floor quadratic denominator by minimum positive variance so N_eff stays well-conditioned
        let safe_quad = quad_form.max(0.10 * sum_w_sq).max(1e-12);
        let neff = ((sum_w * sum_w) / safe_quad).clamp(1.0, (k_total as f64) * 2.0);

        let marginal_corr = if candidate.is_some() && k_open > 0 {
            let cand_idx = k_total - 1;
            (0..k_open)
                .map(|j| c_mat[cand_idx][j])
                .fold(f64::NEG_INFINITY, f64::max)
                .clamp(-1.0, 1.0)
        } else if k_open >= 2 {
            let mut max_pair = f64::NEG_INFINITY;
            for i in 0..k_open {
                for j in (i + 1)..k_open {
                    if c_mat[i][j] > max_pair {
                        max_pair = c_mat[i][j];
                    }
                }
            }
            max_pair.clamp(-1.0, 1.0)
        } else {
            0.0
        };

        let excess_corr = (marginal_corr - 0.35).max(0.0);
        let k_ref = if candidate.is_some() {
            k_open as f64
        } else {
            k_open.saturating_sub(1) as f64
        };
        let raw_kappa = 1.0 / (1.0 + 2.0 * excess_corr * excess_corr * k_ref).sqrt();
        let eigen_kelly_multiplier = raw_kappa.clamp(0.25, 1.0);

        let vetoed = candidate.is_some()
            && k_open >= 3
            && gross_risk_heat_fraction > 0.65
            && (ar_ratio > 0.78 || marginal_corr > 0.85);

        let veto_reason = if vetoed {
            "PORTFOLIO_VETOED_EIGEN_CORRELATION_CROWDING".to_string()
        } else {
            "PASS".to_string()
        };

        PortfolioEigenRiskReport {
            total_legs_k: k_total,
            open_positions_count: k_open,
            shared_return_bars_t: shared_t,
            ledoit_wolf_shrinkage_delta: delta_lw,
            top_eigenvalue_lambda_max: lambda_max,
            absorption_ratio: ar_ratio,
            effective_independent_bets_neff: neff,
            marginal_correlation_penalty: marginal_corr,
            eigen_kelly_multiplier,
            gross_risk_heat_fraction,
            vetoed,
            veto_reason,
        }
    }
}

/// Canonical 6-slot horizon universe for the Regime-Contextual NIG Capital Meta-Allocator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum MetaAllocatorHorizonSlot {
    #[serde(rename = "SCALP_0DTE")]
    Scalp0Dte,
    #[serde(rename = "BRAIN-01")]
    Brain01PennyAnomaly,
    #[serde(rename = "BRAIN-02")]
    Brain02IntradayDayTrade,
    #[serde(rename = "BRAIN-03")]
    Brain03ShortTermSwing,
    #[serde(rename = "BRAIN-04")]
    Brain04CoreSwing,
    #[serde(rename = "BRAIN-05")]
    Brain05LongTermWealth,
}

impl MetaAllocatorHorizonSlot {
    pub const ALL: [Self; 6] = [
        Self::Scalp0Dte,
        Self::Brain01PennyAnomaly,
        Self::Brain02IntradayDayTrade,
        Self::Brain03ShortTermSwing,
        Self::Brain04CoreSwing,
        Self::Brain05LongTermWealth,
    ];

    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Scalp0Dte => "SCALP_0DTE",
            Self::Brain01PennyAnomaly => "BRAIN-01",
            Self::Brain02IntradayDayTrade => "BRAIN-02",
            Self::Brain03ShortTermSwing => "BRAIN-03",
            Self::Brain04CoreSwing => "BRAIN-04",
            Self::Brain05LongTermWealth => "BRAIN-05",
        }
    }

    pub const fn from_brain_and_tier(brain_id: BrainHorizonId, execution_tier: &str) -> Self {
        if execution_tier.eq_ignore_ascii_case("SCALP_0DTE") {
            return Self::Scalp0Dte;
        }
        match brain_id {
            BrainHorizonId::Brain01PennyAnomaly => Self::Brain01PennyAnomaly,
            BrainHorizonId::Brain02IntradayDayTrade => Self::Brain02IntradayDayTrade,
            BrainHorizonId::Brain03ShortTermSwing => Self::Brain03ShortTermSwing,
            BrainHorizonId::Brain04CoreSwing => Self::Brain04CoreSwing,
            BrainHorizonId::Brain05LongTermWealth => Self::Brain05LongTermWealth,
        }
    }
}

/// Conjugate Normal-Inverse-Gamma $(\mu_n, \kappa_n, \alpha_n, \beta_n)$ posterior cell
/// with BOCPD-coupled adaptive forgetting $\gamma_t = \operatorname{clamp}(0.992 - 0.22 \cdot p_{\text{cp}, t},\; 0.76,\; 0.992)$.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NigPosteriorStats {
    pub mu_n: f64,
    pub kappa_n: f64,
    pub alpha_n: f64,
    pub beta_n: f64,
    pub last_forgetting_gamma: f64,
    pub observations_count: usize,
    pub false_veto_regrets_penalized: usize,
}

impl Default for NigPosteriorStats {
    fn default() -> Self {
        Self {
            mu_n: 0.25,
            kappa_n: 2.0,
            alpha_n: 2.5,
            beta_n: 1.25,
            last_forgetting_gamma: compute_adaptive_forgetting_gamma(0.06),
            observations_count: 0,
            false_veto_regrets_penalized: 0,
        }
    }
}

impl NigPosteriorStats {
    /// Assimilates a causal settled trade's risk-adjusted return
    /// $y_t = \text{realized\_r}_t / \max(0.35, \text{mae\_r}_t)$ using BOCPD structural
    /// changepoint probability $p_{\text{cp}, t} \in [0, 1]$ to set dynamic forgetting $\gamma_t$.
    pub fn observe_settled_trade_adaptive(&mut self, realized_r: f64, mae_r: f64, p_cp: f64) {
        let gamma_t = compute_adaptive_forgetting_gamma(p_cp);
        self.observe_settled_trade_with_gamma(realized_r, mae_r, gamma_t);
    }

    /// Assimilates a causal settled trade with an explicit forgetting factor $\gamma_t \in (0, 1)$.
    /// Contracts effective pre-observation pseudo-count $\tilde{\kappa}$ by the geometric memory
    /// horizon $\frac{\gamma_t}{1 - \gamma_t}$ so structural Gamma-Flip shocks ($\gamma_t \le 0.85$)
    /// shed stale pre-break inertia immediately while preserving variance scale $\beta_n / \alpha_n$.
    pub fn observe_settled_trade_with_gamma(
        &mut self,
        realized_r: f64,
        mae_r: f64,
        gamma_t: f64,
    ) {
        let g = gamma_t.clamp(0.50, 0.999);
        let y_t = (realized_r / mae_r.abs().max(0.35)).clamp(-5.0, 8.0);
        let asymptotic_horizon = (g / (1.0 - g)).max(1.5);
        let k_tilde = (g * self.kappa_n).min(asymptotic_horizon).max(1.5);
        let contraction_ratio = (k_tilde / self.kappa_n.max(1e-6)).clamp(0.05, 1.0);
        let a_tilde = (self.alpha_n * contraction_ratio).max(2.0);
        let b_tilde = (self.beta_n * contraction_ratio).max(0.25);

        let kappa_next = k_tilde + 1.0;
        let diff = y_t - self.mu_n;
        let mu_next = (k_tilde * self.mu_n + y_t) / kappa_next;
        let alpha_next = a_tilde + 0.5;
        let beta_next = b_tilde + (k_tilde * diff * diff) / (2.0 * kappa_next);

        self.mu_n = mu_next;
        self.kappa_n = kappa_next;
        self.alpha_n = alpha_next;
        self.beta_n = beta_next;
        self.last_forgetting_gamma = g;
        self.observations_count += 1;
    }

    /// Default causal observation entry point using baseline calm changepoint prior ($p_{\text{cp}} = 0.06$).
    pub fn observe_settled_trade(&mut self, realized_r: f64, mae_r: f64) {
        self.observe_settled_trade_adaptive(realized_r, mae_r, 0.06);
    }

    /// Penalizes posterior mean and dispersion when a hard-negative trap censors a counterfactual winner.
    pub fn penalize_false_veto_regret(&mut self, missed_winner_r: f64) {
        let r_pos = missed_winner_r.clamp(0.0, 6.0);
        if r_pos <= 0.0 {
            return;
        }
        self.mu_n -= (0.22 * r_pos) / (self.kappa_n + 1.0);
        self.beta_n += 0.15 * r_pos * r_pos;
        self.false_veto_regrets_penalized += 1;
    }

    /// Deterministic posterior predictive Student-$t$ Sharpe score:
    /// $$S_{b, r} = \frac{\mu_n}{\sqrt{\beta_n (\kappa_n + 1) / (\alpha_n \kappa_n)}}$$
    pub fn posterior_t_sharpe(&self) -> f64 {
        let pred_var =
            (self.beta_n * (self.kappa_n + 1.0)) / (self.alpha_n * self.kappa_n).max(1e-9);
        self.mu_n / pred_var.max(1e-9).sqrt()
    }
}

/// Wire-serializable per-slot allocation weight $\pi_b(r)$ and posterior Sharpe $S_{b, r}$.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BrainContextualAllocationRow {
    pub slot_id: String,
    pub regime_bucket: String,
    pub capital_weight_pi: f64,
    pub posterior_t_sharpe: f64,
    pub mu_n: f64,
    pub observations_count: usize,
    pub false_veto_regrets_penalized: usize,
}

/// Regime-Contextual Normal-Inverse-Gamma (NIG) Cross-Brain Capital Meta-Allocator
/// coupled with online Adams-MacKay BOCPD adaptive forgetting.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ContextualNigMetaAllocator {
    cells: BTreeMap<(MetaAllocatorHorizonSlot, String), NigPosteriorStats>,
    bocpd_engine: AdamsMacKayBocpdEngine,
}

impl ContextualNigMetaAllocator {
    pub fn new() -> Self {
        Self {
            cells: BTreeMap::new(),
            bocpd_engine: AdamsMacKayBocpdEngine::default(),
        }
    }

    pub fn bocpd_telemetry(&self) -> BocpdStepTelemetry {
        self.bocpd_engine.telemetry()
    }

    pub fn bocpd_engine(&self) -> &AdamsMacKayBocpdEngine {
        &self.bocpd_engine
    }

    pub fn bocpd_engine_mut(&mut self) -> &mut AdamsMacKayBocpdEngine {
        &mut self.bocpd_engine
    }

    /// Builds causal NIG posteriors from settled `ClosedTradeRow`s and settled counterfactual shadow regrets,
    /// stepping `AdamsMacKayBocpdEngine` chronologically to adapt $\gamma_t$ on every structural shift.
    pub fn from_causal_ledger(
        closed_trades: &[ClosedTradeRow],
        shadow_regrets: &[(MetaAllocatorHorizonSlot, RegimeBucket, f64)],
    ) -> Self {
        let mut alloc = Self::new();
        for row in closed_trades {
            let entry_vec = crate::autopsy_evolver::extract_causal_entry_array_32(row);
            let bocpd_step = alloc.bocpd_engine.observe_features_32(&entry_vec);
            let regime = crate::autopsy_evolver::classify_regime_bucket(&entry_vec);
            let is_0dte_scalp = matches!(row.symbol.as_str(), "SPY" | "QQQ" | "IWM")
                && row.hold_duration_ms <= 3_600_000
                && row.brain_id == BrainHorizonId::Brain02IntradayDayTrade;
            let slot = if is_0dte_scalp {
                MetaAllocatorHorizonSlot::Scalp0Dte
            } else {
                MetaAllocatorHorizonSlot::from_brain_and_tier(row.brain_id, "STANDARD")
            };
            let ep = row.entry_price.to_f64();
            let risk_pct = ((ep - row.initial_structural_stop.to_f64()).abs() / ep.max(1e-6)
                * 100.0)
                .max(0.10);
            let mae_r = (row.mae_pct / risk_pct).max(0.35);
            alloc.observe_adaptive(
                slot,
                regime,
                row.realized_r_multiple,
                mae_r,
                bocpd_step.bocpd_changepoint_prob,
            );
        }
        for &(slot, regime, missed_r) in shadow_regrets {
            alloc.penalize_regret(slot, regime, missed_r);
        }
        alloc
    }

    pub fn observe_settled_trade_adaptive(
        &mut self,
        slot: MetaAllocatorHorizonSlot,
        regime: RegimeBucket,
        realized_r: f64,
        mae_r: f64,
        bocpd_p_cp: f64,
    ) {
        self.observe_adaptive(slot, regime, realized_r, mae_r, bocpd_p_cp);
    }

    pub fn observe_adaptive(
        &mut self,
        slot: MetaAllocatorHorizonSlot,
        regime: RegimeBucket,
        realized_r: f64,
        mae_r: f64,
        p_cp: f64,
    ) {
        self.cells
            .entry((slot, regime.as_str().to_string()))
            .or_default()
            .observe_settled_trade_adaptive(realized_r, mae_r, p_cp);
    }

    pub fn observe_with_fixed_gamma(
        &mut self,
        slot: MetaAllocatorHorizonSlot,
        regime: RegimeBucket,
        realized_r: f64,
        mae_r: f64,
        fixed_gamma: f64,
    ) {
        self.cells
            .entry((slot, regime.as_str().to_string()))
            .or_default()
            .observe_settled_trade_with_gamma(realized_r, mae_r, fixed_gamma);
    }

    pub fn observe(
        &mut self,
        slot: MetaAllocatorHorizonSlot,
        regime: RegimeBucket,
        realized_r: f64,
        mae_r: f64,
    ) {
        let p_cp = self.bocpd_engine.telemetry().bocpd_changepoint_prob;
        self.observe_adaptive(slot, regime, realized_r, mae_r, p_cp);
    }

    pub fn penalize_regret(
        &mut self,
        slot: MetaAllocatorHorizonSlot,
        regime: RegimeBucket,
        missed_winner_r: f64,
    ) {
        self.cells
            .entry((slot, regime.as_str().to_string()))
            .or_default()
            .penalize_false_veto_regret(missed_winner_r);
    }

    /// Computes regime-conditioned capital budget weights $\sum_{b} \pi_b(r) = 1.0$, $\pi_b(r) \in [0.10, 0.42]$.
    pub fn allocate_for_regime(&self, regime: RegimeBucket) -> Vec<BrainContextualAllocationRow> {
        let regime_key = regime.as_str().to_string();
        let mut sharpes = [0.0_f64; 6];
        let mut unnorm = [0.0_f64; 6];
        let mut stats_snapshot: Vec<NigPosteriorStats> = Vec::with_capacity(6);

        for (idx, &slot) in MetaAllocatorHorizonSlot::ALL.iter().enumerate() {
            let st = self
                .cells
                .get(&(slot, regime_key.clone()))
                .cloned()
                .unwrap_or_default();
            let s_br = st.posterior_t_sharpe();
            sharpes[idx] = s_br;
            unnorm[idx] = (1.0 / 6.0) * (1.35 * s_br.clamp(-4.0, 4.0)).exp();
            stats_snapshot.push(st);
        }

        let projected = project_capped_simplex_kl(&unnorm, 0.10, 0.42, 1.0);
        MetaAllocatorHorizonSlot::ALL
            .iter()
            .enumerate()
            .map(|(idx, &slot)| BrainContextualAllocationRow {
                slot_id: slot.as_str().to_string(),
                regime_bucket: regime_key.clone(),
                capital_weight_pi: projected[idx],
                posterior_t_sharpe: sharpes[idx],
                mu_n: stats_snapshot[idx].mu_n,
                observations_count: stats_snapshot[idx].observations_count,
                false_veto_regrets_penalized: stats_snapshot[idx].false_veto_regrets_penalized,
            })
            .collect()
    }

    /// Looks up `(capital_weight_pi, posterior_t_sharpe, kelly_meta_multiplier)` for a `(slot, regime)`.
    pub fn weight_and_sharpe_for_slot(
        &self,
        slot: MetaAllocatorHorizonSlot,
        regime: RegimeBucket,
    ) -> (f64, f64, f64) {
        let rows = self.allocate_for_regime(regime);
        if let Some(found) = rows.iter().find(|r| r.slot_id == slot.as_str()) {
            let kelly_mult = (found.capital_weight_pi / (1.0 / 6.0)).clamp(0.60, 2.20);
            (found.capital_weight_pi, found.posterior_t_sharpe, kelly_mult)
        } else {
            (1.0 / 6.0, 0.0, 1.0)
        }
    }
}
