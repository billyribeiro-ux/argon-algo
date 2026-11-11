//! Phase 36 Pillar 4: Microprice & Queue-Depletion Adverse-Selection Execution Router
//!
//! Deterministically routes approved equity/option tickets using causal L1 queue-imbalance and
//! Stream-3 orderflow toxicity (`bid_price`, `ask_price`, `bid_size`, `ask_size`,
//! `orderflow_toxicity_vpin`, `atr_bar`, `direction`, `regime_bucket`):
//! - Volume-Weighted Microprice:
//!   $$P_{\text{micro}} = \frac{Q_{\text{bid}} \cdot P_{\text{ask}} + Q_{\text{ask}} \cdot P_{\text{bid}}}{Q_{\text{bid}} + Q_{\text{ask}}}$$
//! - Signed Microprice Alpha Skew:
//!   $$\delta_{\text{micro}} = \text{dir} \cdot \frac{P_{\text{micro}} - P_{\text{mid}}}{\max(P_{\text{ask}} - P_{\text{bid}}, 10^{-4})} \in [-0.5, +0.5]$$
//! - Adverse-Selection Urgency Score:
//!   $$u_{\text{exec}} = \operatorname{sigmoid}\big(3.0 \cdot \delta_{\text{micro}} + 2.2 \cdot (\text{toxicity} - 0.50) - 2.5 \cdot (s_{\text{atr}} - 0.12)\big) \in [0, 1]$$

use crate::autopsy_evolver::RegimeBucket;
use raw_fingerprint_core::TradeDirection;
use serde::{Deserialize, Serialize};

/// Deterministic microstructure order-execution tactic selected by `MicropriceExecutionRouter`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionTactic {
    #[serde(rename = "PASSIVE_MID_PEG_LIMIT")]
    PassiveMidPegLimit,
    #[serde(rename = "ADAPTIVE_MICROPRICE_PEG")]
    AdaptiveMicropricePeg,
    #[serde(rename = "IMMEDIATE_ALMGREN_CHRISS_SWEEP")]
    ImmediateAlmgrenChrissSweep,
}

impl ExecutionTactic {
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::PassiveMidPegLimit => "PASSIVE_MID_PEG_LIMIT",
            Self::AdaptiveMicropricePeg => "ADAPTIVE_MICROPRICE_PEG",
            Self::ImmediateAlmgrenChrissSweep => "IMMEDIATE_ALMGREN_CHRISS_SWEEP",
        }
    }
}

/// Causal L1 + Stream-3 microstructure quote and toxicity inputs.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MicrostructureQuoteInput {
    pub bid_price: f64,
    pub ask_price: f64,
    pub bid_size: f64,
    pub ask_size: f64,
    pub orderflow_toxicity_vpin: f64,
    pub atr_bar: f64,
    pub structural_risk_per_share: f64,
    pub direction: TradeDirection,
    pub regime_bucket: RegimeBucket,
}

/// Concrete limit price, tactic, step-up chase, and expected slippage telemetry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MicropriceExecutionPlan {
    pub execution_tactic: ExecutionTactic,
    pub mid_price: f64,
    pub microprice: f64,
    pub microprice_limit_price: f64,
    pub microprice_alpha_skew: f64,
    pub spread_to_atr_ratio: f64,
    pub execution_urgency_score: f64,
    pub step_up_chase_ms: u64,
    pub expected_slippage_bps: f64,
    pub expected_slippage_r: f64,
}

pub struct MicropriceExecutionRouter;

impl MicropriceExecutionRouter {
    #[inline]
    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x.clamp(-25.0, 25.0)).exp())
    }

    /// Computes Volume-Weighted Microprice $P_{\text{micro}}$, signed skew $\delta_{\text{micro}}$,
    /// urgency $u_{\text{exec}}$, and optimal execution tactic + limit price.
    pub fn route_order(input: &MicrostructureQuoteInput) -> MicropriceExecutionPlan {
        let bid = input.bid_price.max(0.0001);
        let ask = input.ask_price.max(bid + 1e-4);
        let spread = (ask - bid).max(1e-4);
        let mid_price = 0.5 * (bid + ask);

        let q_bid = input.bid_size.max(1.0);
        let q_ask = input.ask_size.max(1.0);
        let microprice = (q_bid * ask + q_ask * bid) / (q_bid + q_ask);

        let dir = input.direction.sign();
        let microprice_alpha_skew = (dir * (microprice - mid_price) / spread).clamp(-0.5, 0.5);
        let spread_to_atr_ratio = spread / input.atr_bar.max(1e-4);
        let toxicity = input.orderflow_toxicity_vpin.clamp(0.0, 1.0);

        let logit = 3.0 * microprice_alpha_skew + 2.2 * (toxicity - 0.50)
            - 2.5 * (spread_to_atr_ratio - 0.12);
        let execution_urgency_score = Self::sigmoid(logit);

        let (execution_tactic, raw_limit_px, step_up_chase_ms, expected_slip_dollars) =
            if execution_urgency_score < 0.38
                || (input.regime_bucket == RegimeBucket::PosGammaPinMeanRevert
                    && spread_to_atr_ratio > 0.15)
            {
                (
                    ExecutionTactic::PassiveMidPegLimit,
                    mid_price,
                    0_u64,
                    0.10 * spread,
                )
            } else if execution_urgency_score < 0.72 {
                (
                    ExecutionTactic::AdaptiveMicropricePeg,
                    microprice,
                    1_500_u64,
                    (microprice - mid_price).abs() + 0.20 * spread,
                )
            } else {
                let sweep_cap = if dir > 0.0 {
                    ask + 0.10 * spread
                } else {
                    (bid - 0.10 * spread).max(0.0001)
                };
                (
                    ExecutionTactic::ImmediateAlmgrenChrissSweep,
                    sweep_cap,
                    0_u64,
                    0.55 * spread,
                )
            };

        let expected_slippage_bps = (expected_slip_dollars / mid_price.max(1e-4)) * 10_000.0;
        let expected_slippage_r =
            expected_slip_dollars / input.structural_risk_per_share.abs().max(1e-4);

        MicropriceExecutionPlan {
            execution_tactic,
            mid_price,
            microprice,
            microprice_limit_price: raw_limit_px,
            microprice_alpha_skew,
            spread_to_atr_ratio,
            execution_urgency_score,
            step_up_chase_ms,
            expected_slippage_bps,
            expected_slippage_r,
        }
    }
}

