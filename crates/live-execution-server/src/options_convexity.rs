//! Phase 35 & Phase 36 Pillar 1: Analytical 1st & 2nd-Order BSM Surface Greeks (`Charm`, `Vanna`, `Vomma`),
//! Optimal Strike-Pair Solver (`solve_optimal_option_structure`), and Causal 0DTE Theta+Charm Hazard Engine.
//!
//! Purges all scalar multi-leg heuristics: every 2-leg `DEFINED_RISK_VERTICAL_BAG` Greek
//! ($\Delta, \Gamma, \Theta, \mathcal{V}, \text{Vanna}, \text{Charm}, \text{Vomma}$) is computed as
//! the exact analytical BSM surface difference $\text{Leg}_{\text{Long}}(K_1, \sigma(K_1)) - \text{Leg}_{\text{Short}}(K_2, \sigma(K_2))$.

use crate::autopsy_evolver::RegimeBucket;
use raw_fingerprint_core::{RawBarRecord, TradeDirection};
use serde::{Deserialize, Serialize};

/// Option contract right (`Call` for bullish setups, `Put` for bearish setups).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptionRight {
    #[serde(rename = "CALL")]
    Call,
    #[serde(rename = "PUT")]
    Put,
}

impl OptionRight {
    pub const fn from_direction(dir: TradeDirection) -> Self {
        match dir {
            TradeDirection::Long => Self::Call,
            TradeDirection::Short => Self::Put,
        }
    }
}

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

/// Analytical 1st & 2nd-order Black-Scholes-Merton surface Greeks for a single option leg.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BsmLegGreeks {
    pub right: OptionRight,
    pub strike: f64,
    pub implied_vol: f64,
    pub time_years: f64,
    pub theoretical_price: f64,
    pub delta: f64,
    pub gamma: f64,
    pub theta_annual: f64,
    pub theta_daily: f64,
    pub theta_per_bar: f64,
    pub vega: f64,
    /// $\text{Vanna} = \frac{\partial \Delta}{\partial \sigma} = -e^{-qT}\phi(d_1)\frac{d_2}{\sigma}$.
    pub vanna: f64,
    /// Annualized calendar $\text{Charm} = -\frac{\partial \Delta}{\partial T}$.
    pub charm_annual: f64,
    /// Intraday trading-session $\text{Charm}_{\text{daily}} = (-\frac{\partial \Delta}{\partial T}) / 252.0$.
    pub charm_daily: f64,
    pub charm_per_bar: f64,
    /// $\text{Vomma} = \mathcal{V} \frac{d_1 d_2}{\sigma}$.
    pub vomma: f64,
}

pub type BsmSurfaceLegGreeks = BsmLegGreeks;

/// Solved single-leg $(K_1^*)$ or 2-leg vertical `BAG` $(K_1^*, K_2^*)$ with exact leg-difference Greeks.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OptimalOptionStructurePlan {
    pub structure: OptionConvexityStructure,
    pub optimal_long_strike: f64,
    pub optimal_short_strike: Option<f64>,
    pub net_debit: f64,
    pub net_delta: f64,
    pub net_gamma: f64,
    pub net_theta_daily: f64,
    pub net_theta_per_bar: f64,
    pub net_vega: f64,
    pub net_vanna: f64,
    pub net_charm_daily: f64,
    pub net_charm_per_bar: f64,
    pub net_vomma: f64,
    pub convexity_utility_score: f64,
    pub leg_long: BsmLegGreeks,
    pub leg_short: Option<BsmLegGreeks>,
}

pub type SolvedOptionStructurePlan = OptimalOptionStructurePlan;

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

/// Output telemetry for the Smart Option Structure Router & Analytical Surface Convexity Engine.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OptionConvexityEvaluation {
    pub recommended_option_structure: OptionConvexityStructure,
    pub gamma_theta_efficiency: f64,
    pub iv_rank: f64,
    pub theta_per_bar_abs: f64,
    pub initial_option_risk_usd: f64,
    pub theta_burn_per_bar_ratio: f64,
    pub theta_burn_hazard: f64,
    pub optimal_long_strike: f64,
    pub optimal_short_strike: Option<f64>,
    pub net_debit: f64,
    pub net_delta: f64,
    pub net_gamma: f64,
    pub net_theta_daily: f64,
    pub net_theta_per_bar: f64,
    pub net_vega: f64,
    pub net_vanna: f64,
    pub net_charm: f64,
    pub net_charm_per_bar: f64,
    pub net_vomma: f64,
    pub convexity_utility_score: f64,
    pub solved_plan: OptimalOptionStructurePlan,
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

/// Standard normal PDF $\phi(x) = \frac{1}{\sqrt{2\pi}} e^{-x^2 / 2}$.
#[inline]
pub fn norm_pdf(x: f64) -> f64 {
    const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;
    INV_SQRT_2PI * (-0.5 * x * x).exp()
}

/// High-precision standard normal CDF $\Phi(x)$ (Abramowitz & Stegun 7.1.26, max error $< 1.5 \times 10^{-7}$).
#[inline]
pub fn norm_cdf(x: f64) -> f64 {
    if !x.is_finite() {
        return if x.is_sign_positive() { 1.0 } else { 0.0 };
    }
    let z = x.abs() * std::f64::consts::FRAC_1_SQRT_2;
    let t = 1.0 / (1.0 + 0.327_591_1 * z);
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    let erfc_z = poly * (-z * z).exp();
    if x >= 0.0 {
        1.0 - 0.5 * erfc_z
    } else {
        0.5 * erfc_z
    }
}

/// Computes exact 1st & 2nd-order Black-Scholes-Merton surface Greeks (`Delta`, `Gamma`, `Theta`,
/// `Vega`, `Vanna`, `Charm`, `Vomma`) with skew-smile volatility $\sigma(K)$.
#[allow(clippy::too_many_arguments)]
pub fn compute_bsm_leg_greeks(
    right: OptionRight,
    spot: f64,
    strike: f64,
    dte_days: f64,
    bar_session_fraction: f64,
    r_rate: f64,
    div_yield_q: f64,
    atm_iv: f64,
    skew_beta: f64,
    smile_beta: f64,
) -> BsmLegGreeks {
    let s = spot.max(0.01);
    let k = strike.max(0.01);
    let min_t = 1.0 / (252.0 * 390.0);
    let t = (dte_days / 252.0).max(min_t);
    let sqrt_t = t.sqrt();
    let sess_frac = bar_session_fraction.clamp(1.0 / 390.0, 5.0);

    let log_m = (k / s).ln();
    let base_atm_iv = atm_iv.max(0.05);
    let raw_sigma = base_atm_iv * (1.0 + skew_beta * log_m + smile_beta * log_m * log_m);
    let sigma = raw_sigma.clamp(0.05, 4.00);
    // Sticky-strike skew slope d(sigma)/dS holding K fixed: d/dS ln(K/S) = -1/S
    let d_sigma_d_spot = if (0.05..=4.00).contains(&raw_sigma) {
        -(base_atm_iv / s) * (skew_beta + 2.0 * smile_beta * log_m)
    } else {
        0.0
    };
    let sig_sqrt_t = (sigma * sqrt_t).max(1e-8);

    let d1 = ((s / k).ln() + (r_rate - div_yield_q + 0.5 * sigma * sigma) * t) / sig_sqrt_t;
    let d2 = d1 - sig_sqrt_t;

    let pdf_d1 = norm_pdf(d1);
    let cdf_d1 = norm_cdf(d1);
    let cdf_d2 = norm_cdf(d2);
    let cdf_neg_d1 = norm_cdf(-d1);
    let cdf_neg_d2 = norm_cdf(-d2);

    let df_q = (-div_yield_q * t).exp();
    let df_r = (-r_rate * t).exp();
    let vega = s * df_q * pdf_d1 * sqrt_t;

    let (theoretical_price, delta, theta_annual, d_delta_d_tau) = match right {
        OptionRight::Call => {
            let px = (s * df_q * cdf_d1 - k * df_r * cdf_d2).max(1e-4);
            let del_bsm = df_q * cdf_d1;
            let del_skew = (del_bsm + vega * d_sigma_d_spot).clamp(1e-4, df_q);
            let th_ann = -(s * df_q * pdf_d1 * sigma) / (2.0 * sqrt_t)
                - r_rate * k * df_r * cdf_d2
                + div_yield_q * s * df_q * cdf_d1;
            let dd_dtau = -div_yield_q * df_q * cdf_d1
                + df_q * pdf_d1 * (2.0 * (r_rate - div_yield_q) * t - d2 * sig_sqrt_t)
                    / (2.0 * t * sig_sqrt_t);
            (px, del_skew, th_ann, dd_dtau)
        }
        OptionRight::Put => {
            let px = (k * df_r * cdf_neg_d2 - s * df_q * cdf_neg_d1).max(1e-4);
            let del_bsm = -df_q * cdf_neg_d1;
            let del_skew = (del_bsm + vega * d_sigma_d_spot).clamp(-df_q, -1e-4);
            let th_ann = -(s * df_q * pdf_d1 * sigma) / (2.0 * sqrt_t)
                + r_rate * k * df_r * cdf_neg_d2
                - div_yield_q * s * df_q * cdf_neg_d1;
            let dd_dtau = div_yield_q * df_q * cdf_neg_d1
                + df_q * pdf_d1 * (2.0 * (r_rate - div_yield_q) * t - d2 * sig_sqrt_t)
                    / (2.0 * t * sig_sqrt_t);
            (px, del_skew, th_ann, dd_dtau)
        }
    };

    let gamma = (df_q * pdf_d1) / (s * sig_sqrt_t);
    let theta_daily = theta_annual / 252.0;
    let theta_per_bar = theta_daily * sess_frac;

    // 2nd-order cross-Greeks: Vanna = d(Delta)/d(sigma), Charm = -d(Delta)/d(T), Vomma = Vega * d1 * d2 / sigma
    let vanna = -df_q * pdf_d1 * (d2 / sigma);
    let charm_annual = -d_delta_d_tau;
    let charm_daily = charm_annual / 252.0;
    let charm_per_bar = charm_daily * sess_frac;
    let vomma = vega * (d1 * d2) / sigma;

    BsmLegGreeks {
        right,
        strike: k,
        implied_vol: sigma,
        time_years: t,
        theoretical_price,
        delta,
        gamma,
        theta_annual,
        theta_daily,
        theta_per_bar,
        vega,
        vanna,
        charm_annual,
        charm_daily,
        charm_per_bar,
        vomma,
    }
}

/// Rounds an option strike cleanly to institutional CBOE strike increments based on underlying spot tier.
#[inline]
pub fn round_cboe_strike(raw_strike: f64, spot: f64) -> f64 {
    let step = if spot < 10.0 {
        0.50
    } else if spot < 100.0 {
        1.00
    } else if spot < 300.0 {
        2.50
    } else {
        1.00
    };
    ((raw_strike / step).round() * step).max(0.50)
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
        (tr_sum / (window.len() as f64))
            .max(fallback_spot * 0.0005)
            .max(1e-4)
    }

    /// Derives causal IV Rank percentile $\in [0.0, 1.0]$ comparing `contract_iv` against realized
    /// bar range volatility over the causal pre-entry window.
    pub fn derive_causal_iv_rank(
        bars: &[RawBarRecord],
        contract_iv: f64,
        bar_session_fraction: f64,
    ) -> f64 {
        let spot = bars
            .last()
            .map(|b| b.close.to_f64())
            .unwrap_or(100.0)
            .max(1e-4);
        let atr = Self::compute_causal_atr_bar(bars, spot);
        let bars_per_year = (252.0 / bar_session_fraction.max(1.0 / 390.0)).max(252.0);
        let realized_vol_ann = ((atr / spot) * bars_per_year.sqrt()).clamp(0.08, 2.50);
        let vol_low = (realized_vol_ann * 0.65).min(contract_iv * 0.85);
        let vol_high = (realized_vol_ann * 1.55).max(vol_low + 0.12);
        ((contract_iv - vol_low) / (vol_high - vol_low)).clamp(0.0, 1.0)
    }

    /// Deterministic Optimal Strike-Pair Solver (`Phase 36 Pillar 1`):
    /// Evaluates candidate long strikes $K_1 \in \{-1.0, -0.5, 0.0, +0.5\} \times \text{ATR}_{\text{daily}}$
    /// and short strikes $K_2$ anchored near structural `target_1_price` ($\pm 0.5 \times \text{ATR}_{\text{bar}}$),
    /// maximizing causal Charm/Vanna-Adjusted Convexity Utility $U(K_1, K_2)$ with exact two-leg BSM differences.
    #[allow(clippy::too_many_arguments)]
    pub fn solve_optimal_option_structure(
        direction: TradeDirection,
        spot_price: f64,
        structural_stop_price: f64,
        target_1_price: f64,
        atr_bar: f64,
        atm_iv: f64,
        iv_rank: f64,
        dte_days: f64,
        bar_session_fraction: f64,
        regime_bucket: RegimeBucket,
    ) -> OptimalOptionStructurePlan {
        let right = OptionRight::from_direction(direction);
        let dir = direction.sign();
        let spot = spot_price.max(0.05);
        let sess_frac = bar_session_fraction.clamp(1.0 / 390.0, 5.0);
        let atr_b = atr_bar.max(spot * 0.0005).max(1e-4);
        let bars_per_day = (1.0 / sess_frac).clamp(1.0, 390.0);
        let atr_daily = (atr_b * bars_per_day.sqrt()).max(atr_b * 1.5);

        let risk_dist = (spot - structural_stop_price).abs().max(atr_b * 0.8);
        let eff_t1 = if (target_1_price - spot) * dir > 0.25 * atr_b {
            target_1_price
        } else {
            spot + dir * (2.2 * risk_dist).max(2.0 * atr_b)
        };
        let delta_s_t1 = eff_t1 - spot;
        let delta_s_bar = dir * atr_b;

        // Expected causal IV shift: negative (IV crush) in high-IV / PosGamma pin, positive in low-IV NegGamma trend or bearish put momentum spikes
        let expected_delta_iv = match regime_bucket {
            RegimeBucket::PosGammaPinMeanRevert => -0.20 * atm_iv * (0.35 + iv_rank.clamp(0.0, 1.0)),
            RegimeBucket::NegGammaMomentumTrend if iv_rank <= 0.55 => {
                let put_spike_boost = if direction == TradeDirection::Short { 1.25 } else { 1.0 };
                0.12 * atm_iv * (0.65 - iv_rank) * put_spike_boost
            }
            RegimeBucket::LiquidityVacuumHighToxicity
                if direction == TradeDirection::Short && iv_rank <= 0.70 =>
            {
                0.15 * atm_iv * (0.80 - iv_rank)
            }
            _ => -0.10 * atm_iv * (iv_rank.clamp(0.0, 1.0) - 0.40),
        };

        let r_rate = 0.045;
        let div_q = 0.012;
        let skew_beta = -0.15;
        let smile_beta = 0.25;

        let long_offsets = [-1.0_f64, -0.5, 0.0, 0.5];
        let short_offsets = [-0.5_f64, 0.0, 0.5];

        // Evaluate best 1-leg candidate K_1^* and best 2-leg vertical BAG candidate (K_1^*, K_2^*)
        let mut best_single: Option<OptimalOptionStructurePlan> = None;
        let mut best_vertical: Option<OptimalOptionStructurePlan> = None;

        // Convex runner horizon move in uncapped negative-gamma momentum trends vs bounded pin targets
        let trend_runner_move = match regime_bucket {
            RegimeBucket::NegGammaMomentumTrend if iv_rank <= 0.65 => {
                delta_s_t1.signum() * delta_s_t1.abs().max(2.6 * atr_daily)
            }
            _ => delta_s_t1,
        };

        for &m1 in &long_offsets {
            let k1 = round_cboe_strike(spot + dir * m1 * atr_daily, spot);
            let leg_l = compute_bsm_leg_greeks(
                right,
                spot,
                k1,
                dte_days,
                sess_frac,
                r_rate,
                div_q,
                atm_iv,
                skew_beta,
                smile_beta,
            );

            // 1. Single-leg utility U(K_1): consistent Taylor horizon payoff + separate per-bar Gamma convexity + signed Vega/Vanna
            let single_t1_payoff = (leg_l.delta * trend_runner_move
                + 0.5 * leg_l.gamma * trend_runner_move * trend_runner_move)
                .max(0.0);
            let single_bar_convexity = 0.5 * leg_l.gamma * atr_b * atr_b;
            let single_num = single_t1_payoff
                + single_bar_convexity
                + leg_l.vanna * delta_s_bar * expected_delta_iv
                + leg_l.vega * expected_delta_iv;
            let single_den = leg_l.theoretical_price.max(0.05)
                + 3.0 * (leg_l.theta_per_bar.abs() + leg_l.charm_per_bar.abs() * atr_b);
            let u_single = single_num / single_den.max(1e-6);

            let cand_single = OptimalOptionStructurePlan {
                structure: OptionConvexityStructure::SingleLongConvex,
                optimal_long_strike: k1,
                optimal_short_strike: None,
                net_debit: leg_l.theoretical_price,
                net_delta: leg_l.delta,
                net_gamma: leg_l.gamma,
                net_theta_daily: leg_l.theta_daily,
                net_theta_per_bar: leg_l.theta_per_bar,
                net_vega: leg_l.vega,
                net_vanna: leg_l.vanna,
                net_charm_daily: leg_l.charm_daily,
                net_charm_per_bar: leg_l.charm_per_bar,
                net_vomma: leg_l.vomma,
                convexity_utility_score: u_single,
                leg_long: leg_l,
                leg_short: None,
            };
            if best_single
                .as_ref()
                .is_none_or(|b| u_single > b.convexity_utility_score)
            {
                best_single = Some(cand_single);
            }

            // 2. Two-leg vertical BAG candidates (K_1, K_2) anchored near T1
            for &m2 in &short_offsets {
                let raw_k2 = eff_t1 + dir * m2 * atr_b;
                let min_width = (0.5 * atr_b).max(if spot < 10.0 { 0.50 } else { 1.00 });
                let enforced_raw_k2 = if (raw_k2 - k1) * dir < min_width {
                    k1 + dir * min_width
                } else {
                    raw_k2
                };
                let mut k2 = round_cboe_strike(enforced_raw_k2, spot);
                if (k2 - k1) * dir <= 0.05 {
                    k2 = k1 + dir * (if spot < 10.0 { 0.50 } else { 1.00 });
                }

                let leg_s = compute_bsm_leg_greeks(
                    right,
                    spot,
                    k2,
                    dte_days,
                    sess_frac,
                    r_rate,
                    div_q,
                    atm_iv,
                    skew_beta,
                    smile_beta,
                );

                // Exact two-leg difference: Leg_Long(K_1, sigma_1) - Leg_Short(K_2, sigma_2)
                let net_debit = (leg_l.theoretical_price - leg_s.theoretical_price).max(0.02);
                let net_delta = leg_l.delta - leg_s.delta;
                let net_gamma = leg_l.gamma - leg_s.gamma;
                let net_theta_daily = leg_l.theta_daily - leg_s.theta_daily;
                let net_theta_per_bar = leg_l.theta_per_bar - leg_s.theta_per_bar;
                let net_vega = leg_l.vega - leg_s.vega;
                let net_vanna = leg_l.vanna - leg_s.vanna;
                let net_charm_daily = leg_l.charm_daily - leg_s.charm_daily;
                let net_charm_per_bar = leg_l.charm_per_bar - leg_s.charm_per_bar;
                let net_vomma = leg_l.vomma - leg_s.vomma;

                let spread_width = (k2 - k1).abs();
                let max_spread_gain = (spread_width - net_debit).max(0.01);
                // Consistent Taylor step Delta S_{T1} for Target-1 payoff (clamped to spread saturation) + separate 1-bar convexity + signed Vega
                let raw_t1_gain = (net_delta * delta_s_t1
                    + 0.5 * net_gamma * delta_s_t1 * delta_s_t1)
                    .clamp(0.0, max_spread_gain);
                let bar_gamma_convexity = 0.5 * net_gamma * atr_b * atr_b;
                let vert_num = raw_t1_gain
                    + bar_gamma_convexity
                    + net_vanna * delta_s_bar * expected_delta_iv
                    + net_vega * expected_delta_iv;
                let vert_den = net_debit.max(0.05)
                    + 3.0 * (net_theta_per_bar.abs() + net_charm_per_bar.abs() * atr_b);
                let u_vert = vert_num / vert_den.max(1e-6);

                let cand_vert = OptimalOptionStructurePlan {
                    structure: OptionConvexityStructure::DefinedRiskVerticalBag,
                    optimal_long_strike: k1,
                    optimal_short_strike: Some(k2),
                    net_debit,
                    net_delta,
                    net_gamma,
                    net_theta_daily,
                    net_theta_per_bar,
                    net_vega,
                    net_vanna,
                    net_charm_daily,
                    net_charm_per_bar,
                    net_vomma,
                    convexity_utility_score: u_vert,
                    leg_long: leg_l,
                    leg_short: Some(leg_s),
                };
                if best_vertical
                    .as_ref()
                    .is_none_or(|b| u_vert > b.convexity_utility_score)
                {
                    best_vertical = Some(cand_vert);
                }
            }
        }

        let single_plan = best_single.expect("4 long strikes evaluated");
        let vert_plan = best_vertical.expect("12 strike pairs evaluated");

        // Select between solved optimal single-leg vs solved 2-leg vertical BAG using surface convexity & regime physics
        let single_eta = (0.5 * single_plan.net_gamma.abs() * atr_b * atr_b)
            / (single_plan.net_theta_per_bar.abs() + 1e-6);

        if iv_rank > 0.65
            || single_eta < 1.15
            || regime_bucket == RegimeBucket::PosGammaPinMeanRevert
            || vert_plan.convexity_utility_score > single_plan.convexity_utility_score * 1.15
        {
            vert_plan
        } else {
            single_plan
        }
    }

    /// Second-order Taylor + Vanna/Charm/Vega option value change from an exact solved `OptimalOptionStructurePlan`:
    /// $$\delta V = \Delta_{\text{net}} \delta S + \frac{1}{2}\Gamma_{\text{net}} (\delta S)^2 + \Theta_{\text{net, daily}} \delta t_{\text{days}} + \mathcal{V}_{\text{net}} \delta \sigma + \text{Vanna}_{\text{net}} \delta S \delta \sigma + \text{Charm}_{\text{net, daily}} \delta S \delta t_{\text{days}}$$
    #[inline]
    pub fn second_order_option_pnl_from_plan(
        plan: &OptimalOptionStructurePlan,
        delta_s: f64,
        delta_t_days: f64,
        delta_iv: f64,
    ) -> f64 {
        let dt = delta_t_days.max(0.0);
        let raw_dv = plan.net_delta * delta_s
            + 0.5 * plan.net_gamma * delta_s * delta_s
            + plan.net_theta_daily * dt
            + plan.net_vega * delta_iv
            + plan.net_vanna * delta_s * delta_iv
            + plan.net_charm_daily * delta_s * dt;
        if let Some(k2) = plan.optimal_short_strike {
            let max_gain = ((k2 - plan.optimal_long_strike).abs() - plan.net_debit).max(0.01);
            raw_dv.clamp(-plan.net_debit, max_gain)
        } else {
            raw_dv.max(-plan.net_debit)
        }
    }

    /// Evaluates analytical BSM surface Greeks, solves optimal 1-leg vs 2-leg vertical `BAG` strikes,
    /// computes Convexity-to-Theta Efficiency $\eta_{\Gamma/\Theta}$, and derives per-bar Theta+Charm hazard.
    pub fn evaluate_option_convexity(input: &OptionGreeksConvexityInput) -> OptionConvexityEvaluation {
        let iv_rank = input.iv_rank.clamp(0.0, 1.0);
        let session_frac = input.bar_session_fraction.clamp(1.0 / 390.0, 5.0);
        let risk_dist = (input.spot_price - input.structural_stop_price).abs().max(0.01);
        let t1_est = input.spot_price + input.direction.sign() * (2.2 * risk_dist);

        let solved = Self::solve_optimal_option_structure(
            input.direction,
            input.spot_price,
            input.structural_stop_price,
            t1_est,
            input.atr_bar,
            input.contract_iv.max(0.08),
            iv_rank,
            input.dte_days,
            session_frac,
            input.regime_bucket,
        );

        // Intraday 0DTE time-decay & Charm acceleration when < 1.25 trading days remain
        let zero_dte_accel = if input.is_scalp_0dte || input.dte_days <= 1.25 {
            (1.0 / input.dte_days.clamp(0.08, 1.25).sqrt()).clamp(1.0, 3.5)
        } else {
            1.0
        };

        // Single-leg benchmark efficiency eta_{Gamma/Theta} and exact solved net Theta + Charm bleed
        let raw_single_theta_bar =
            input.contract_theta_daily.abs() * session_frac * zero_dte_accel;
        let gamma_bar_dollar = 0.5 * input.contract_gamma.abs() * input.atr_bar * input.atr_bar;
        let eta_gamma_theta = gamma_bar_dollar / (raw_single_theta_bar + 1e-6);

        let recommended_option_structure = if iv_rank > 0.65
            || eta_gamma_theta < 1.15
            || input.regime_bucket == RegimeBucket::PosGammaPinMeanRevert
        {
            OptionConvexityStructure::DefinedRiskVerticalBag
        } else {
            solved.structure
        };

        // Ensure `active_plan` matches `recommended_option_structure` using exact BSM leg differences
        let active_plan = if solved.structure == recommended_option_structure {
            solved
        } else if recommended_option_structure == OptionConvexityStructure::DefinedRiskVerticalBag {
            let right = OptionRight::from_direction(input.direction);
            let dir = input.direction.sign();
            let k1 = solved.optimal_long_strike;
            let k2 = round_cboe_strike(
                t1_est.max(0.50),
                input.spot_price,
            );
            let eff_k2 = if (k2 - k1) * dir <= 0.05 {
                k1 + dir * (if input.spot_price < 10.0 { 0.50 } else { 1.00 })
            } else {
                k2
            };
            let leg_l = solved.leg_long;
            let leg_s = compute_bsm_leg_greeks(
                right,
                input.spot_price,
                eff_k2,
                input.dte_days,
                session_frac,
                0.045,
                0.012,
                input.contract_iv.max(0.08),
                -0.15,
                0.25,
            );
            OptimalOptionStructurePlan {
                structure: OptionConvexityStructure::DefinedRiskVerticalBag,
                optimal_long_strike: k1,
                optimal_short_strike: Some(eff_k2),
                net_debit: (leg_l.theoretical_price - leg_s.theoretical_price).max(0.02),
                net_delta: leg_l.delta - leg_s.delta,
                net_gamma: leg_l.gamma - leg_s.gamma,
                net_theta_daily: leg_l.theta_daily - leg_s.theta_daily,
                net_theta_per_bar: leg_l.theta_per_bar - leg_s.theta_per_bar,
                net_vega: leg_l.vega - leg_s.vega,
                net_vanna: leg_l.vanna - leg_s.vanna,
                net_charm_daily: leg_l.charm_daily - leg_s.charm_daily,
                net_charm_per_bar: leg_l.charm_per_bar - leg_s.charm_per_bar,
                net_vomma: leg_l.vomma - leg_s.vomma,
                convexity_utility_score: solved.convexity_utility_score,
                leg_long: leg_l,
                leg_short: Some(leg_s),
            }
        } else {
            solved
        };

        let underlying_stop_move = -input.direction.sign() * risk_dist;
        let raw_stop_loss = Self::second_order_option_pnl_from_plan(
            &active_plan,
            underlying_stop_move,
            session_frac,
            0.0,
        )
        .abs();
        let entry_ref_debit = active_plan
            .net_debit
            .min(input.option_entry_premium.max(0.05))
            .max(0.05);
        let initial_option_risk_usd = raw_stop_loss
            .max(entry_ref_debit * 0.25)
            .min(active_plan.net_debit.max(0.05))
            .max(1e-4);

        // Exact net bar Theta + Charm bleed from the solved analytical BSM plan (zero scalar 0.18 heuristic)
        let eff_theta_charm_bar = (active_plan.net_theta_per_bar.abs()
            + active_plan.net_charm_per_bar.abs() * input.atr_bar * 0.25)
            * zero_dte_accel;
        let theta_burn_per_bar_ratio =
            (eff_theta_charm_bar / initial_option_risk_usd).clamp(0.0, 1.0);
        let theta_burn_hazard = (1.0
            / (1.0 + (-(3.2 * (theta_burn_per_bar_ratio * 3.0 - 0.28))).exp()))
        .clamp(0.0, 1.0);

        OptionConvexityEvaluation {
            recommended_option_structure,
            gamma_theta_efficiency: eta_gamma_theta,
            iv_rank,
            theta_per_bar_abs: eff_theta_charm_bar,
            initial_option_risk_usd,
            theta_burn_per_bar_ratio,
            theta_burn_hazard,
            optimal_long_strike: active_plan.optimal_long_strike,
            optimal_short_strike: active_plan.optimal_short_strike,
            net_debit: active_plan.net_debit,
            net_delta: active_plan.net_delta,
            net_gamma: active_plan.net_gamma,
            net_theta_daily: active_plan.net_theta_daily,
            net_theta_per_bar: active_plan.net_theta_per_bar,
            net_vega: active_plan.net_vega,
            net_vanna: active_plan.net_vanna,
            net_charm: active_plan.net_charm_daily,
            net_charm_per_bar: active_plan.net_charm_per_bar,
            net_vomma: active_plan.net_vomma,
            convexity_utility_score: active_plan.convexity_utility_score,
            solved_plan: active_plan,
        }
    }

    /// Computes causal cumulative 0DTE Theta+Charm bleed ratio $\tau_{m}$
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
            completed_bars_count >= 4 && running_peak_r < 0.45 && cumulative_tau >= 0.26;
        (cumulative_tau, theta_hazard, should_exit_theta_stall)
    }

    /// Computes Stage 3 dynamic MFE profit-retention factor $\rho_{\text{eff}}(k) \in [0.40, 0.92]$
    /// tightened causally by exact 2-leg/single-leg `net_theta_per_bar` and `net_charm_per_bar` bleed.
    pub fn compute_stage3_convexity_retention_rho(
        base_rho: f64,
        completed_bars_count: usize,
        eval: &OptionConvexityEvaluation,
        atr_bar: f64,
    ) -> f64 {
        let bar_bleed_usd =
            eval.net_theta_per_bar.abs() + eval.net_charm_per_bar.abs() * atr_bar.max(1e-4);
        let bleed_pressure = ((completed_bars_count as f64) * bar_bleed_usd
            / eval.initial_option_risk_usd.max(1e-4))
        .clamp(0.0, 1.5);
        (base_rho + 0.14 * bleed_pressure).clamp(0.40, 0.92)
    }

    /// Deterministic causal bar-by-bar simulator for a `SCALP_0DTE` option trade under full analytical
    /// $(\Delta, \Gamma, \Theta, \mathcal{V}, \text{Vanna}, \text{Charm})$ physics, comparing active
    /// Theta+Charm stall protection (`enable_theta_stall_guard = true`) against passive hold (`false`).
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
                let dv = Self::second_order_option_pnl_from_plan(
                    &eval.solved_plan,
                    ds,
                    (bar_num as f64) * session_frac * zero_dte_accel,
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
                let dv = Self::second_order_option_pnl_from_plan(
                    &eval.solved_plan,
                    ds_close,
                    (bar_num as f64) * session_frac * zero_dte_accel,
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
        let dv = Self::second_order_option_pnl_from_plan(
            &eval.solved_plan,
            ds_final,
            (total_bars as f64) * session_frac * zero_dte_accel,
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
