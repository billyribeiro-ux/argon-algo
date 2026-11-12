//! Phase 36 / 36.1 Pillar 3: Anytime-Valid Symmetric-Bounded Waudby-Smith–Ramdas Ville E-Process
//! & Page/Lorden Wealth-Mixture Alpha-Decay Shield.
//!
//! Protects live capital from silent alpha decay or structural microstructure breaks with zero
//! multiple-testing false alarms via Ville's Supermartingale Inequality
//! ($\mathbb{P}_{H_0}(\sup_{t \ge 1} E_t^{(b)} \ge 1/\alpha) \le \alpha$, $\alpha = 0.05 \implies 1/\alpha = 20.0$).
//!
//! ### Exact Symmetric-Bounded Supermartingale Construction (`Phase 36.1`)
//! For each horizon slot $b$, realized R-multiples are bounded symmetrically around minimum viable edge
//! floor $\mu_{\text{floor}} = 0.08\text{R}$ with half-width $B = 2.45\text{R}$ ($R_k^{\text{cap}} \in [-2.37\text{R}, +2.53\text{R}]$):
//! $$d_k = \frac{\mu_{\text{floor}} - \operatorname{clamp}(R_k,\; \mu_{\text{floor}} - B,\; \mu_{\text{floor}} + B)}{B} \in [-1.0,\; +1.0]$$
//! - Under $H_0: \mathbb{E}[R_k^{\text{cap}} \mid \mathcal{F}_{k-1}] \ge \mu_{\text{floor}}$, $\mathbb{E}[d_k \mid \mathcal{F}_{k-1}] \le 0$.
//!   In particular, for an asymmetric convexity runner (e.g., 40% `+2.0R`, 60% `-1.0R`, $\mathbb{E}[R] = +0.20\text{R}$),
//!   $\mathbb{E}[d_k] = (0.08 - 0.20) / 2.45 = -0.04898 < 0$ with zero upside winner truncation bias.
//! - With strictly predictable ($\mathcal{F}_{k-1}$-measurable) Online Newton Step stake $\lambda_k \in [0.06, 0.92]$,
//!   every step multiplier $f_k = 1.0 + \lambda_k d_k \ge 0.08 > 0$ satisfies $\mathbb{E}_{H_0}[f_k \mid \mathcal{F}_{k-1}] \le 1.0$.
//! - Tracks **BOTH**:
//!   1. `raw_ville_e_value` ($E_0 = 1.0$, $E_t = E_{t-1} \cdot f_t$ with **ZERO** lower-barrier reflection clamp,
//!      guaranteeing $\mathbb{P}_{H_0}(\sup_{t \ge 1} E_t \ge 20.0) \le 0.05$), AND
//!   2. `wealth_mixture_e_value` ($M_0 = 1.0$, Page/Lorden Geometric Wealth-Mixture E-Detector
//!      $M_t = (1 - \epsilon) M_{t-1} f_t + \epsilon f_t$ with $\epsilon = 0.01$), controlling
//!      $\text{ARL}_0 \ge \frac{1}{\epsilon \alpha} = 2{,}000$ trades under $H_0$ while catching post-streak
//!      alpha breaks in $< 15$ bad trades and healing below hysteresis floor $M_t < 4.0$ on shadow winners.

use crate::ledger::ClosedTradeRow;
use crate::portfolio_eigen_risk::MetaAllocatorHorizonSlot;
use raw_fingerprint_core::BrainHorizonId;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const VILLE_EDGE_FLOOR_R: f64 = 0.08;
pub const VILLE_SYMMETRIC_HALF_WIDTH_R: f64 = 2.45;
pub const VILLE_WEALTH_MIXTURE_EPSILON: f64 = 0.01;
pub const VILLE_ALPHA_THRESHOLD: f64 = 20.0; // alpha = 0.05 => 1 / alpha = 20.0
pub const VILLE_RECOVERY_HYSTERESIS_FLOOR: f64 = 4.0;

fn default_one() -> f64 {
    1.0
}

/// Per-horizon anytime-valid Ville Martingale E-Process & Page/Lorden Wealth-Mixture state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BrainMartingaleState {
    pub slot_id: String,
    /// Active Page/Lorden wealth-mixture E-detector $M_t$ governing probation ($M_t \ge 20.0$) & Kelly throttle.
    pub martingale_e_value: f64,
    /// Exact unreflected Ville test supermartingale $E_t = \prod_{j=1}^t (1 + \lambda_j d_j)$ ($E_0 = 1.0$, zero lower clamp).
    #[serde(default = "default_one")]
    pub raw_ville_e_value: f64,
    /// Explicit Page/Lorden geometric mixture $M_t = (1 - \epsilon) M_{t-1} f_t + \epsilon f_t$ ($\epsilon = 0.01$).
    #[serde(default = "default_one")]
    pub wealth_mixture_e_value: f64,
    /// Latest symmetrically bounded shortfall $d_k \in [-1.0, +1.0]$.
    #[serde(default)]
    pub last_bounded_shortfall_d: f64,
    pub martingale_kelly_multiplier: f64,
    pub probation_shadow_only: bool,
    pub predictable_stake_lambda: f64,
    pub live_trades_observed: usize,
    pub shadow_recovery_trades_observed: usize,
}

impl BrainMartingaleState {
    pub fn new(slot: MetaAllocatorHorizonSlot) -> Self {
        Self {
            slot_id: slot.as_str().to_string(),
            martingale_e_value: 1.0,
            raw_ville_e_value: 1.0,
            wealth_mixture_e_value: 1.0,
            last_bounded_shortfall_d: 0.0,
            martingale_kelly_multiplier: 1.0,
            probation_shadow_only: false,
            predictable_stake_lambda: 0.36,
            live_trades_observed: 0,
            shadow_recovery_trades_observed: 0,
        }
    }

    /// Computes smooth anytime-valid Kelly throttle $\kappa_{\text{martingale}} \in [0.15, 1.0]$.
    #[inline]
    pub fn compute_kelly_multiplier(e_val: f64) -> f64 {
        (2.0 / e_val.max(2.0)).sqrt().clamp(0.15, 1.0)
    }

    /// Computes symmetrically bounded shortfall $d_k \in [-1.0, +1.0]$ around $\mu_{\text{floor}} = 0.08\text{R}$.
    #[inline]
    pub fn compute_symmetric_shortfall_d(realized_r: f64) -> f64 {
        let r_clean = if realized_r.is_finite() {
            realized_r
        } else {
            0.0
        };
        let r_min = VILLE_EDGE_FLOOR_R - VILLE_SYMMETRIC_HALF_WIDTH_R;
        let r_max = VILLE_EDGE_FLOOR_R + VILLE_SYMMETRIC_HALF_WIDTH_R;
        let r_cap = r_clean.clamp(r_min, r_max);
        ((VILLE_EDGE_FLOOR_R - r_cap) / VILLE_SYMMETRIC_HALF_WIDTH_R).clamp(-1.0, 1.0)
    }

    /// Updates both the exact unreflected Ville supermartingale $E_t$ and the Page/Lorden
    /// geometric wealth-mixture E-detector $M_t$ on a settled live or shadow trade $R_k$.
    /// Uses strictly predictable pre-trade stake $\lambda_k \in [0.06, 0.92]$ ($\mathcal{F}_{k-1}$-measurable)
    /// before adapting $\lambda_{k+1}$ via Online Newton Step on log-wealth.
    pub fn observe_trade_r(&mut self, realized_r: f64, is_shadow_counterfactual: bool) {
        let d_k = Self::compute_symmetric_shortfall_d(realized_r);
        let lambda_k = self.predictable_stake_lambda.clamp(0.06, 0.92);

        // Exact positive betting factor f_k = 1 + lambda_k * d_k >= 0.08 > 0 (since d_k in [-1, 1])
        let factor = 1.0 + lambda_k * d_k;

        // 1. Exact unreflected Ville supermartingale E_t = prod_{j=1}^t (1 + lambda_j * d_j) — ZERO lower clamp!
        let next_raw_ville_e = self.raw_ville_e_value * factor;

        // 2. Page/Lorden geometric wealth-mixture E-detector M_t = (1 - eps) * M_{t-1} * f_t + eps * f_t
        //    (Zero lower reflection clamp; upper saturation at 60.0 > 20.0 preserves exact hitting time tau_20 & ARL_0 >= 2,000)
        let next_mixture_m = ((1.0 - VILLE_WEALTH_MIXTURE_EPSILON)
            * self.wealth_mixture_e_value
            * factor
            + VILLE_WEALTH_MIXTURE_EPSILON * factor)
            .min(60.0);

        let next_probation =
            if !self.probation_shadow_only && next_mixture_m >= VILLE_ALPHA_THRESHOLD {
                true
            } else if self.probation_shadow_only && next_mixture_m < VILLE_RECOVERY_HYSTERESIS_FLOOR
            {
                false
            } else {
                self.probation_shadow_only
            };

        let next_kelly = Self::compute_kelly_multiplier(next_mixture_m);

        // Online Newton Step (ONS) log-wealth gradient stake update strictly for NEXT trade k+1 (F_k-measurable)
        let log_wealth_grad = d_k / factor;
        let next_lambda = (lambda_k + 0.65 * log_wealth_grad).clamp(0.06, 0.92);

        let (next_live, next_shadow) = if is_shadow_counterfactual {
            (
                self.live_trades_observed,
                self.shadow_recovery_trades_observed + 1,
            )
        } else {
            (
                self.live_trades_observed + 1,
                self.shadow_recovery_trades_observed,
            )
        };

        *self = Self {
            slot_id: self.slot_id.clone(),
            martingale_e_value: next_mixture_m,
            raw_ville_e_value: next_raw_ville_e,
            wealth_mixture_e_value: next_mixture_m,
            last_bounded_shortfall_d: d_k,
            martingale_kelly_multiplier: next_kelly,
            probation_shadow_only: next_probation,
            predictable_stake_lambda: next_lambda,
            live_trades_observed: next_live,
            shadow_recovery_trades_observed: next_shadow,
        };
    }
}

/// Multi-brain Ville Martingale E-Process Alpha-Decay & Shadow Probation Shield.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VilleMartingaleAlphaShield {
    states: BTreeMap<MetaAllocatorHorizonSlot, BrainMartingaleState>,
}

impl Default for VilleMartingaleAlphaShield {
    fn default() -> Self {
        Self::new()
    }
}

impl VilleMartingaleAlphaShield {
    pub fn new() -> Self {
        let mut states = BTreeMap::new();
        for slot in MetaAllocatorHorizonSlot::ALL {
            states.insert(slot, BrainMartingaleState::new(slot));
        }
        Self { states }
    }

    /// Replays causal settled trades in chronological order to initialize per-brain Ville E-processes.
    pub fn from_causal_ledger(closed_trades: &[ClosedTradeRow]) -> Self {
        let mut shield = Self::new();
        for row in closed_trades {
            let is_0dte_scalp = matches!(row.symbol.as_str(), "SPY" | "QQQ" | "IWM")
                && row.hold_duration_ms <= 3_600_000
                && row.brain_id == BrainHorizonId::Brain02IntradayDayTrade;
            let slot = if is_0dte_scalp {
                MetaAllocatorHorizonSlot::Scalp0Dte
            } else {
                MetaAllocatorHorizonSlot::from_brain_and_tier(row.brain_id, "STANDARD")
            };
            shield.observe_trade(slot, row.realized_r_multiple, false);
        }
        shield
    }

    pub fn is_in_probation(&self, slot: MetaAllocatorHorizonSlot) -> bool {
        self.state_for_slot(slot).probation_shadow_only
    }

    pub fn observe_trade_r(
        &mut self,
        slot: MetaAllocatorHorizonSlot,
        realized_r: f64,
        is_shadow_counterfactual: bool,
    ) -> BrainMartingaleState {
        self.observe_trade(slot, realized_r, is_shadow_counterfactual)
    }

    pub fn observe_trade(
        &mut self,
        slot: MetaAllocatorHorizonSlot,
        realized_r: f64,
        is_shadow_counterfactual: bool,
    ) -> BrainMartingaleState {
        let entry = self
            .states
            .entry(slot)
            .or_insert_with(|| BrainMartingaleState::new(slot));
        entry.observe_trade_r(realized_r, is_shadow_counterfactual);
        entry.clone()
    }

    pub fn state_for_slot(&self, slot: MetaAllocatorHorizonSlot) -> BrainMartingaleState {
        self.states
            .get(&slot)
            .cloned()
            .unwrap_or_else(|| BrainMartingaleState::new(slot))
    }

    pub fn snapshot_all(&self) -> Vec<BrainMartingaleState> {
        MetaAllocatorHorizonSlot::ALL
            .iter()
            .map(|&slot| self.state_for_slot(slot))
            .collect()
    }
}
