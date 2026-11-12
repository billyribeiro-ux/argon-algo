//! Phase 37 Pillar 2: Spatial Dealer Gamma-Flip & Charm/Vanna Pinning Vector Field
//!
//! Upgrades discrete regime labels into a continuous spatial dealer hedging vector field
//! derived causally from Stream-2 GEX/Charm/Vanna features `[12..20]`, underlying spot $S$,
//! bar ATR, and option strike lattices:
//! - Causal Zero-Gamma Flip Level: $S_{\gamma=0}$
//! - Overhead Call Gamma Wall: $K_{\text{call}}^{\text{wall}} > S$
//! - Underfoot Put Gamma Wall: $K_{\text{put}}^{\text{wall}} < S$
//! - Signed Net Dealer Hedging Multiplier: $\mathcal{G}_{\text{dealer}}(S) \in [-1.0, +1.0]$
//! - Wall Pinning Resistance:
//!   $$\Psi_{\text{pin}} = \max(0, \mathcal{G}_{\text{dealer}}) \cdot \exp\!\left(-\frac{|K_{\text{wall}} - S|}{0.75 \cdot \text{ATR}}\right) \in [0.0, 1.0]$$
//!
//! Closed-Loop Target & Strike Warping:
//! When a setup's raw target overshoots $K_{\text{wall}}$ under high positive-gamma pinning ($\Psi_{\text{pin}} \ge 0.55$),
//! pulls `effective_target_1_price` inside the wall ($K_{\text{call}}^{\text{wall}} - 0.12 \cdot \text{ATR}$ for Long;
//! $K_{\text{put}}^{\text{wall}} + 0.12 \cdot \text{ATR}$ for Short) and anchors the short leg $K_2^*$ of
//! `DEFINED_RISK_VERTICAL_BAG` directly onto $K_{\text{wall}}$ to harvest dealer pin decay.

use raw_fingerprint_core::TradeDirection;
use serde::{Deserialize, Serialize};

pub const DEALER_PINNING_WARP_THRESHOLD: f64 = 0.55;

/// Spatial snapshot of dealer options positioning and pinning vector field.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpatialDealerGammaField {
    pub spot_price: f64,
    pub zero_gamma_flip_price: f64,
    pub overhead_call_wall: f64,
    pub underfoot_put_wall: f64,
    pub net_dealer_hedging_multiplier: f64,
    pub call_wall_pin_resistance: f64,
    pub put_wall_pin_resistance: f64,
    pub active_pin_resistance: f64,
    pub is_positive_gamma_regime: bool,
}

impl SpatialDealerGammaField {
    /// Computes the spatial dealer gamma field from causal Stream-2 features `[12..20]`,
    /// underlying spot price, and ATR.
    pub fn compute_spatial_field(
        spot: f64,
        atr_bar: f64,
        features_32: &[f64; 32],
        direction: TradeDirection,
    ) -> Self {
        let s = spot.max(0.0001);
        let atr = atr_bar.max(s * 0.001).max(0.01);

        // Stream 2 dealer Greeks: features[12..20]
        let gex_proxy: f64 = features_32[12..20].iter().sum::<f64>() / 8.0;
        let vanna_proxy = features_32[14];
        let charm_proxy = features_32[15];

        // Strike increment based on price tier
        let strike_inc = if s >= 200.0 {
            5.0
        } else if s >= 50.0 {
            1.0
        } else if s >= 10.0 {
            0.50
        } else {
            0.10
        };

        // Zero-Gamma Flip level S_{gamma=0}
        let gex_bias = (gex_proxy * 1.5 + vanna_proxy * 0.5).clamp(-4.0, 4.0);
        let s_gamma_zero = s * (1.0 - 0.012 * gex_bias);

        // Overhead Call Wall K_{call}^{wall}
        let call_dist_atrs = (1.80 + 0.65 * (1.0 + gex_proxy.max(0.0))).clamp(1.20, 4.50);
        let raw_call_wall = s + call_dist_atrs * atr;
        let k_call_wall = (raw_call_wall / strike_inc).round() * strike_inc;

        // Underfoot Put Wall K_{put}^{wall}
        let put_dist_atrs = (1.80 + 0.65 * (1.0 + (-gex_proxy).max(0.0))).clamp(1.20, 4.50);
        let raw_put_wall = s - put_dist_atrs * atr;
        let k_put_wall = (raw_put_wall / strike_inc).round() * strike_inc;

        // Signed Net Dealer Hedging Multiplier G_{dealer}(S) in [-1.0, 1.0]
        // Positive when S > S_{gamma=0} and gex_proxy > 0 (dealers long gamma, buy dips, sell rips)
        // Negative when S < S_{gamma=0} and gex_proxy < 0 (dealers short gamma, accelerate trend)
        let s_diff_norm = (s - s_gamma_zero) / atr.max(s * 0.01);
        let net_multiplier = ((gex_proxy * 0.85 + charm_proxy * 0.35) + s_diff_norm * 0.50)
            .tanh()
            .clamp(-1.0, 1.0);

        let g_dealer_pos = net_multiplier.max(0.0);

        // Pinning resistances
        let d_call = (k_call_wall - s).abs();
        let psi_call = (g_dealer_pos * (-d_call / (0.75 * atr)).exp()).clamp(0.0, 1.0);

        let d_put = (s - k_put_wall).abs();
        let psi_put = (g_dealer_pos * (-d_put / (0.75 * atr)).exp()).clamp(0.0, 1.0);

        let active_pin = match direction {
            TradeDirection::Long => psi_call,
            TradeDirection::Short => psi_put,
        };

        Self {
            spot_price: s,
            zero_gamma_flip_price: s_gamma_zero,
            overhead_call_wall: k_call_wall,
            underfoot_put_wall: k_put_wall,
            net_dealer_hedging_multiplier: net_multiplier,
            call_wall_pin_resistance: psi_call,
            put_wall_pin_resistance: psi_put,
            active_pin_resistance: active_pin,
            is_positive_gamma_regime: net_multiplier > 0.0,
        }
    }

    /// Warps raw `target_1_price` inside the dealer gamma wall and selects the short strike
    /// for defined-risk vertical bags when pinning resistance $\Psi_{\text{pin}} \ge 0.55$.
    pub fn warp_target_and_short_strike(
        &self,
        direction: TradeDirection,
        raw_target_1: f64,
        atr_bar: f64,
    ) -> WarpedTargetResolution {
        let atr = atr_bar.max(0.01);
        match direction {
            TradeDirection::Long => {
                let wall = self.overhead_call_wall;
                if self.call_wall_pin_resistance >= DEALER_PINNING_WARP_THRESHOLD
                    && raw_target_1 >= wall
                {
                    let warped_t1 = wall - 0.12 * atr;
                    WarpedTargetResolution {
                        effective_target_1_price: warped_t1,
                        nearest_gamma_wall_price: wall,
                        dealer_pin_resistance: self.call_wall_pin_resistance,
                        pinned_short_strike: Some(wall),
                        was_warped_by_pin: true,
                    }
                } else {
                    WarpedTargetResolution {
                        effective_target_1_price: raw_target_1,
                        nearest_gamma_wall_price: wall,
                        dealer_pin_resistance: self.call_wall_pin_resistance,
                        pinned_short_strike: None,
                        was_warped_by_pin: false,
                    }
                }
            }
            TradeDirection::Short => {
                let wall = self.underfoot_put_wall;
                if self.put_wall_pin_resistance >= DEALER_PINNING_WARP_THRESHOLD
                    && raw_target_1 <= wall
                {
                    let warped_t1 = wall + 0.12 * atr;
                    WarpedTargetResolution {
                        effective_target_1_price: warped_t1,
                        nearest_gamma_wall_price: wall,
                        dealer_pin_resistance: self.put_wall_pin_resistance,
                        pinned_short_strike: Some(wall),
                        was_warped_by_pin: true,
                    }
                } else {
                    WarpedTargetResolution {
                        effective_target_1_price: raw_target_1,
                        nearest_gamma_wall_price: wall,
                        dealer_pin_resistance: self.put_wall_pin_resistance,
                        pinned_short_strike: None,
                        was_warped_by_pin: false,
                    }
                }
            }
        }
    }
}

/// Result of evaluating spatial dealer gamma wall pinning and strike warping.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WarpedTargetResolution {
    pub effective_target_1_price: f64,
    pub nearest_gamma_wall_price: f64,
    pub dealer_pin_resistance: f64,
    pub pinned_short_strike: Option<f64>,
    pub was_warped_by_pin: bool,
}
