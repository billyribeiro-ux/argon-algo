//! Phase 35 Deliverable 3: Causal Options Greeks Convexity & 0DTE Theta-Bleed Hazard Engine
//!
//! Models second-order option value dynamics
//! $$\delta V \approx \Delta \, \delta S + \frac{1}{2} \Gamma \, (\delta S)^2 + \Theta \, \delta t + \mathcal{V} \, \delta \sigma_{\text{IV}}$$
//! to:
//! 1. Route options setups between `SINGLE_LONG_CONVEX` and `DEFINED_RISK_VERTICAL_BAG` via causal
//!    Convexity-to-Theta Efficiency $\eta_{\Gamma/\Theta}$ and IV Rank, and
//! 2. Couple cumulative 0DTE Theta-bleed $\tau_{k-1}$ into Stage 1/2 stall evaluation and Stage 3
//!    reversal hazard retention so flat intraday chop exits before time decay destroys option premium.

use crate::autopsy_evolver::RegimeBucket;
use raw_fingerprint_core::{RawBarRecord, TradeDirection};
use serde::{Deserialize, Serialize};

/// Recommended options execution structure emitted by `OptionsConvexityEngine`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptionConvexityStructure {
    #[serde(rename = "SINGLE_LONG_CONVEX")]
    SingleLongConvex,
    #[serde(rename = "DEFINED_RISK_VERTICAL_BAG")]
    DefinedRiskVerticalBag,
}

impl OptionConvexityStructure {
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::SingleLongConvex => "SINGLE_LONG_CONVEX",
            Self::DefinedRiskVerticalBag => "DEFINED_RISK_VERTICAL_BAG",
        }
    }
}

/// Causal inputs for evaluating an option setup's convexity-to-theta efficiency and structure routing.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct OptionGreeksConvexityInput {
    pub direction: TradeDirection,
    pub spot_price: f64,
    pub structural_stop_price: f64,
    pub atr_bar: f64,
    pub option_entry_premium: f64,
    /// Signed contract Delta ($\Delta \in [-1, 1]$).
    pub contract_delta: f64,
    /// Contract Gamma ($\Gamma \ge 0$).
    pub contract_gamma: f64,
    /// Daily contract Theta ($\Theta_{\text{daily}} \le 0$).
    pub contract_theta_daily: f64,
    /// Contract Vega ($\mathcal{V}$ per `1.00` IV point / `100` vol pts).
    pub contract_vega: f64,
    /// Implied volatility ($\sigma_{\text{IV}}$, e.g. `0.28` for 28%).
    pub contract_iv: f64,
    /// Causal IV Rank percentile in `[0.0, 1.0]`.
    pub iv_rank: f64,
    /// Fractional trading days to expiration (`dte_days > 0`).
    pub dte_days: f64,
    /// Duration of one bar as a fraction of a 390-minute US regular session day (e.g. `5.0 / 390.0`).
    pub bar_session_fraction: f64,
    pub regime_bucket: RegimeBucket,
    pub is_scalp_0dte: bool,
}

/// Output telemetry for the Smart Option Structure Router & Convexity Engine.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OptionConvexityEvaluation {
    pub recommended_option_structure: OptionConvexityStructure,
    pub gamma_theta_efficiency: f64,
    pub iv_rank: f64,
    pub theta_per_bar_abs: f64,
    pub initial_option_risk_usd: f64,
    pub theta_burn_per_bar_ratio: f64,
    pub theta_burn_hazard: f64,
}

/// Step-by-step causal outcome of a 0DTE option trajectory with vs. without Theta-burn stall protection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ZeroDteOptionSimOutcome {
    pub realized_option_r: f64,
    pub exit_bar_index: usize,
    pub cumulative_theta_burn_ratio: f64,
    pub peak_theta_burn_hazard: f64,
    pub exited_via_theta_stall: bool,
    pub outcome_label: String,
}

pub struct OptionsConvexityEngine;

impl OptionsConvexityEngine {
    /// Computes causal mean bar True Range (`ATR_bar`) from pre-entry `RawBarRecord`s (zero lookahead).
    pub fn compute_causal_atr_bar(bars: &[RawBarRecord], fallback_spot: f64) -> f64 {
        if bars.is_empty() {
            return (fallback_spot * 0.0035).max(0.01);
        }
        let window = if bars.len() > 14 {
            &bars[bars.len() - 14..]
        } else {
            bars
        };
        let mut tr_sum = 0.0_f64;
        let mut prev_close: Option<f64> = None;
        for b in window {
            let hi = b.high.to_f64();
            let lo = b.low.to_f64();
            let tr = if let Some(pc) = prev_close {
                (hi - lo).max((hi - pc).abs()).max((lo - pc).abs())
            } else {
                (hi - lo).max(0.0)
            };
            tr_sum += tr;
            prev_close = Some(b.close.to_f64());
        }
        (tr_sum / (window.len() as f64)).max(fallback_spot * 0.0005).max(1e-4)
    }

    /// Derives causal IV Rank percentile $\in [0.0, 1.0]$ comparing `contract_iv` against realized
    /// bar range volatility over the causal pre-entry window.
    pub fn derive_causal_iv_rank(
        bars: &[RawBarRecord],
        contract_iv: f64,
        bar_session_fraction: f64,
    ) -> f64 {
        let spot = bars.last().map(|b| b.close.to_f64()).unwrap_or(100.0).max(1e-4);
        let atr = Self::compute_causal_atr_bar(bars, spot);
        let bars_per_year = (252.0 / bar_session_fraction.max(1.0 / 390.0)).max(252.0);
        let realized_vol_ann = ((atr / spot) * bars_per_year.sqrt()).clamp(0.08, 2.50);
        let vol_low = (realized_vol_ann * 0.65).min(contract_iv * 0.85);
        let vol_high = (realized_vol_ann * 1.55).max(vol_low + 0.12);
        ((contract_iv - vol_low) / (vol_high - vol_low)).clamp(0.0, 1.0)
    }

    /// Second-order Taylor + Vega option value change:
    /// $$\delta V = \Delta \, \delta S + \frac{1}{2} \Gamma \, (\delta S)^2 + \Theta_{\text{daily}} \, \delta t_{\text{days}} + \mathcal{V} \, \delta \sigma_{\text{IV}}$$
    #[inline]
    pub fn second_order_option_pnl_delta(
        structure: OptionConvexityStructure,
        delta_signed: f64,
        gamma: f64,
        theta_daily: f64,
        vega: f64,
        delta_s: f64,
        delta_t_days: f64,
        delta_iv: f64,
    ) -> f64 {
        let (eff_delta, eff_gamma, eff_theta, eff_vega) = match structure {
            OptionConvexityStructure::SingleLongConvex => {
                (delta_signed, gamma.max(0.0), -theta_daily.abs(), vega)
            }
            OptionConvexityStructure::DefinedRiskVerticalBag => (
                0.58 * delta_signed,
                0.22 * gamma.max(0.0),
                -0.18 * theta_daily.abs(),
                0.20 * vega,
            ),
        };
        eff_delta * delta_s
            + 0.5 * eff_gamma * delta_s * delta_s
            + eff_theta * delta_t_days.max(0.0)
            + eff_vega * delta_iv
    }

    /// Evaluates Convexity-to-Theta Efficiency $\eta_{\Gamma/\Theta}$, routes between
    /// `SINGLE_LONG_CONVEX` and `DEFINED_RISK_VERTICAL_BAG`, and computes per-bar Theta hazard.
    pub fn evaluate_option_convexity(input: &OptionGreeksConvexityInput) -> OptionConvexityEvaluation {
        let iv_rank = input.iv_rank.clamp(0.0, 1.0);
        let session_frac = input.bar_session_fraction.clamp(1.0 / 390.0, 5.0);

        // Intraday 0DTE square-root time acceleration factor when < 1 trading day remains
        let zero_dte_accel = if input.is_scalp_0dte || input.dte_days <= 1.25 {
            (1.0 / input.dte_days.clamp(0.08, 1.25).sqrt()).clamp(1.0, 3.5)
        } else {
            1.0
        };
        let theta_per_bar_abs =
            input.contract_theta_daily.abs() * session_frac * zero_dte_accel;
        let gamma_bar_dollar = 0.5 * input.contract_gamma.abs() * input.atr_bar * input.atr_bar;
        let eta_gamma_theta = gamma_bar_dollar / (theta_per_bar_abs + 1e-6);

        let recommended_option_structure = if iv_rank > 0.65
            || eta_gamma_theta < 1.15
            || input.regime_bucket == RegimeBucket::PosGammaPinMeanRevert
        {
            OptionConvexityStructure::DefinedRiskVerticalBag
        } else {
            OptionConvexityStructure::SingleLongConvex
        };

        let underlying_stop_move = -input.direction.sign()
            * (input.spot_price - input.structural_stop_price).abs().max(0.01);
        let raw_stop_loss = Self::second_order_option_pnl_delta(
            recommended_option_structure,
            input.contract_delta,
            input.contract_gamma,
            input.contract_theta_daily,
            input.contract_vega,
            underlying_stop_move,
            session_frac,
            0.0,
        )
        .abs();
        let initial_option_risk_usd = raw_stop_loss
            .max(input.option_entry_premium * 0.25)
            .min(input.option_entry_premium.max(0.05))
            .max(1e-4);

        let eff_theta_bar = match recommended_option_structure {
            OptionConvexityStructure::SingleLongConvex => theta_per_bar_abs,
            OptionConvexityStructure::DefinedRiskVerticalBag => 0.18 * theta_per_bar_abs,
        };
        let theta_burn_per_bar_ratio = (eff_theta_bar / initial_option_risk_usd).clamp(0.0, 1.0);
        let theta_burn_hazard =
            (1.0 / (1.0 + (-(3.2 * (theta_burn_per_bar_ratio * 3.0 - 0.28))).exp())).clamp(0.0, 1.0);

        OptionConvexityEvaluation {
            recommended_option_structure,
            gamma_theta_efficiency: eta_gamma_theta,
            iv_rank,
            theta_per_bar_abs: eff_theta_bar,
            initial_option_risk_usd,
            theta_burn_per_bar_ratio,
            theta_burn_hazard,
        }
    }

    /// Computes causal cumulative 0DTE Theta-bleed ratio $\tau_{m} = \frac{\sum_{j=1}^{m} |\Theta_j \delta t_j|}{\text{InitialOptionRisk}}$
    /// and Stage 1/2 stall hazard decision over $m$ completed causal bars.
    pub fn evaluate_0dte_theta_burn_stall(
        completed_bars_count: usize,
        theta_burn_per_bar_ratio: f64,
        running_peak_r: f64,
    ) -> (f64, f64, bool) {
        let cumulative_tau = (completed_bars_count as f64) * theta_burn_per_bar_ratio.max(0.0);
        let stall_logit = 4.5 * (cumulative_tau - 0.28) - 1.8 * running_peak_r.max(0.0);
        let theta_hazard = 1.0 / (1.0 + (-stall_logit).exp());
        let should_exit_theta_stall =
            completed_bars_count >= 4 && running_peak_r < 0.45 && cumulative_tau >= 0.30;
        (cumulative_tau, theta_hazard, should_exit_theta_stall)
    }

    /// Deterministic causal bar-by-bar simulator for a `SCALP_0DTE` option trade under second-order
    /// $(\Delta, \Gamma, \Theta, \mathcal{V})$ physics, comparing active Theta-burn stall protection
    /// (`enable_theta_stall_guard = true`) against passive hold-until-underlying-stop (`false`).
    pub fn simulate_0dte_option_bars(
        input: &OptionGreeksConvexityInput,
        post_entry_bars: &[RawBarRecord],
        iv_deltas_per_bar: Option<&[f64]>,
        enable_theta_stall_guard: bool,
    ) -> ZeroDteOptionSimOutcome {
        let eval = Self::evaluate_option_convexity(input);
        let ep = input.spot_price;
        let dir = input.direction.sign();
        let underlying_risk = (ep - input.structural_stop_price).abs().max(1e-4);
        let opt_risk = eval.initial_option_risk_usd;
        let session_frac = input.bar_session_fraction.clamp(1.0 / 390.0, 5.0);
        let zero_dte_accel = (1.0 / input.dte_days.clamp(0.08, 1.25).sqrt()).clamp(1.0, 3.5);

        let mut running_peak_r = 0.0_f64;
        let mut peak_hazard = 0.0_f64;
        let mut prev_close = ep;

        for (idx, bar) in post_entry_bars.iter().enumerate() {
            let bar_num = idx + 1;
            let bar_open = bar.open.to_f64();
            let bar_high = bar.high.to_f64();
            let bar_low = bar.low.to_f64();
            let bar_close = bar.close.to_f64();

            let favorable_r = if dir > 0.0 {
                (bar_high - ep) / underlying_risk
            } else {
                (ep - bar_low) / underlying_risk
            };
            let adverse_r = if dir > 0.0 {
                (bar_low - ep) / underlying_risk
            } else {
                (ep - bar_high) / underlying_risk
            };

            let cum_div = iv_deltas_per_bar
                .map(|s| s.iter().take(bar_num).sum::<f64>())
                .unwrap_or(0.0);

            if adverse_r <= -1.0 {
                let stop_px = if dir > 0.0 {
                    input.structural_stop_price.min(bar_open)
                } else {
                    input.structural_stop_price.max(bar_open)
                };
                let ds = stop_px - ep;
                let dv = Self::second_order_option_pnl_delta(
                    eval.recommended_option_structure,
                    input.contract_delta,
                    input.contract_gamma,
                    input.contract_theta_daily * zero_dte_accel,
                    input.contract_vega,
                    ds,
                    (bar_num as f64) * session_frac,
                    cum_div,
                );
                let opt_r = (dv / opt_risk).clamp(-2.5, 6.0);
                return ZeroDteOptionSimOutcome {
                    realized_option_r: opt_r,
                    exit_bar_index: bar_num,
                    cumulative_theta_burn_ratio: (bar_num as f64) * eval.theta_burn_per_bar_ratio,
                    peak_theta_burn_hazard: peak_hazard,
                    exited_via_theta_stall: false,
                    outcome_label: "SCALP_0DTE_UNDERLYING_STOP".to_string(),
                };
            }

            if favorable_r > running_peak_r {
                running_peak_r = favorable_r;
            }
            prev_close = bar_close;

            // End-of-bar Stage 1/2 causal 0DTE Theta-Burn Stall Check (bar_num completed bars)
            let (tau_closed, th_hazard, stall_fire) = Self::evaluate_0dte_theta_burn_stall(
                bar_num,
                eval.theta_burn_per_bar_ratio,
                running_peak_r,
            );
            if th_hazard > peak_hazard {
                peak_hazard = th_hazard;
            }
            if enable_theta_stall_guard && stall_fire {
                let ds_close = bar_close - ep;
                let dv = Self::second_order_option_pnl_delta(
                    eval.recommended_option_structure,
                    input.contract_delta,
                    input.contract_gamma,
                    input.contract_theta_daily * zero_dte_accel,
                    input.contract_vega,
                    ds_close,
                    (bar_num as f64) * session_frac,
                    cum_div,
                );
                let opt_r = (dv / opt_risk).clamp(-1.5, 6.0);
                return ZeroDteOptionSimOutcome {
                    realized_option_r: opt_r,
                    exit_bar_index: bar_num,
                    cumulative_theta_burn_ratio: tau_closed,
                    peak_theta_burn_hazard: peak_hazard,
                    exited_via_theta_stall: true,
                    outcome_label: "SCALP_0DTE_THETA_BURN_STALL_EXIT".to_string(),
                };
            }
        }

        let total_bars = post_entry_bars.len();
        let ds_final = prev_close - ep;
        let cum_div = iv_deltas_per_bar
            .map(|s| s.iter().take(total_bars).sum::<f64>())
            .unwrap_or(0.0);
        let dv = Self::second_order_option_pnl_delta(
            eval.recommended_option_structure,
            input.contract_delta,
            input.contract_gamma,
            input.contract_theta_daily * zero_dte_accel,
            input.contract_vega,
            ds_final,
            (total_bars as f64) * session_frac,
            cum_div,
        );
        ZeroDteOptionSimOutcome {
            realized_option_r: (dv / opt_risk).clamp(-2.5, 6.0),
            exit_bar_index: total_bars,
            cumulative_theta_burn_ratio: (total_bars as f64) * eval.theta_burn_per_bar_ratio,
            peak_theta_burn_hazard: peak_hazard,
            exited_via_theta_stall: false,
            outcome_label: "SCALP_0DTE_TIME_SETTLED".to_string(),
        }
    }
}
