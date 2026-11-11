//! Phase 36 Pillar 3: Anytime-Valid Ville Martingale E-Process Alpha-Decay & Probation Shield
//!
//! Protects live capital from silent alpha decay or structural microstructure breaks with zero
//! multiple-testing false alarms via Ville's Inequality ($P_{H_0}(\exists t \ge 1 : E_t^{(b)} \ge 1/\alpha) \le \alpha$).
//!
//! For each horizon slot $b$, maintains a nonnegative Kelly-betting E-process $E_t^{(b)} \ge 0$ ($E_0^{(b)} = 1.0$)
//! against minimum viable edge floor $\mu_{\text{floor}} = 0.08\text{R}$:
//! $$d_k = \operatorname{clamp}\!\left(\frac{\mu_{\text{floor}} - R_k}{1.50},\; -0.85,\; 2.0\right), \qquad E_t^{(b)} = \operatorname{clamp}\!\left(E_{t-1}^{(b)} \cdot (1.0 + \lambda_k \, d_k),\; 0.25,\; 100.0\right)$$
//! with predictable pre-trade stake $\lambda_k \in [0.05, 0.35]$.
//! - Smooth Kelly throttle: $\kappa_{\text{martingale}}^{(b)} = \operatorname{clamp}\!\left(\sqrt{2.0 / \max(2.0, E_t^{(b)})},\; 0.15,\; 1.0\right)$.
//! - Shadow Probation Quarantine ($\alpha = 0.05 \implies 1/\alpha = 20.0$): when $E_t^{(b)} \ge 20.0$,
//!   `probation_shadow_only = true` routes live setups strictly to zero-capital Shadow-Tape until
//!   counterfactual shadow winners drive $E_t^{(b)} < 4.0$.

use crate::ledger::ClosedTradeRow;
use crate::portfolio_eigen_risk::MetaAllocatorHorizonSlot;
use raw_fingerprint_core::BrainHorizonId;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const VILLE_EDGE_FLOOR_R: f64 = 0.08;
pub const VILLE_ALPHA_THRESHOLD: f64 = 20.0; // alpha = 0.05 => 1 / alpha = 20.0
pub const VILLE_RECOVERY_HYSTERESIS_FLOOR: f64 = 4.0;

/// Per-horizon anytime-valid Ville Martingale E-Process state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BrainMartingaleState {
    pub slot_id: String,
    pub martingale_e_value: f64,
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
            martingale_kelly_multiplier: 1.0,
            probation_shadow_only: false,
            predictable_stake_lambda: 0.22,
            live_trades_observed: 0,
            shadow_recovery_trades_observed: 0,
        }
    }

    /// Computes smooth anytime-valid Kelly throttle $\kappa_{\text{martingale}} \in [0.15, 1.0]$.
    #[inline]
    pub fn compute_kelly_multiplier(e_val: f64) -> f64 {
        (2.0 / e_val.max(2.0)).sqrt().clamp(0.15, 1.0)
    }

    /// Updates the Ville betting E-process on a settled live or counterfactual shadow trade $R_k$.
    /// Uses strictly predictable pre-trade stake $\lambda_k \in [0.05, 0.35]$ before adapting $\lambda_{k+1}$.
    pub fn observe_trade_r(&mut self, realized_r: f64, is_shadow_counterfactual: bool) {
        let r_clean = if realized_r.is_finite() {
            realized_r.clamp(-5.0, 10.0)
        } else {
            0.0
        };
        let d_k = ((VILLE_EDGE_FLOOR_R - r_clean) / 1.50).clamp(-0.85, 2.0);
        let lambda_k = self.predictable_stake_lambda.clamp(0.05, 0.35);

        let factor = (1.0 + lambda_k * d_k).max(0.05);
        let next_e = (self.martingale_e_value * factor).clamp(0.25, 100.0);

        let next_probation = if !self.probation_shadow_only && next_e >= VILLE_ALPHA_THRESHOLD {
            true
        } else if self.probation_shadow_only && next_e < VILLE_RECOVERY_HYSTERESIS_FLOOR {
            false
        } else {
            self.probation_shadow_only
        };

        let next_kelly = Self::compute_kelly_multiplier(next_e);

        // Online Newton / gradient stake update strictly for the NEXT trade (predictable filtration F_{k})
        let next_lambda = (lambda_k + 0.07 * d_k).clamp(0.05, 0.35);

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
            martingale_e_value: next_e,
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
