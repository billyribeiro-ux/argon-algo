use raw_fingerprint_core::BrainHorizonId;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Canonical 32-feature names across the 4 raw institutional streams (`0..32`).
pub const CANONICAL_32_FEATURE_METADATA: [(&str, &str); 32] = [
    ("f00_bar_r_close_or_pos10", "STREAM_1_PRICE_VOLUME"),
    ("f01_bar_r_open_or_disp1", "STREAM_1_PRICE_VOLUME"),
    ("f02_bar_r_fav_or_max_opp12", "STREAM_1_PRICE_VOLUME"),
    ("f03_bar_r_adv_or_disp2", "STREAM_1_PRICE_VOLUME"),
    ("f04_candle_body_efficiency", "STREAM_1_PRICE_VOLUME"),
    ("f05_opposing_rejection_wick", "STREAM_1_PRICE_VOLUME"),
    ("f06_favorable_demand_wick", "STREAM_1_PRICE_VOLUME"),
    ("f07_vwap_anchored_displacement", "STREAM_1_PRICE_VOLUME"),
    ("f08_entry_candle_body_impulse", "STREAM_1_PRICE_VOLUME"),
    ("f09_entry_opposing_wick_trap", "STREAM_1_PRICE_VOLUME"),
    ("f10_post_entry_volume_surge", "STREAM_1_PRICE_VOLUME"),
    ("f11_prior_bar_bm1_body_vector", "STREAM_1_PRICE_VOLUME"),
    ("f12_prior_bar_bm1_wick_ratio", "STREAM_1_PRICE_VOLUME"),
    ("f13_10bar_range_position_anchor", "STREAM_1_PRICE_VOLUME"),
    ("f14_short_volume_ratio_velocity", "STREAM_2_FINRA_DARKPOOL"),
    ("f15_darkpool_block_absorption", "STREAM_2_FINRA_DARKPOOL"),
    ("f16_cost_to_borrow_pressure", "STREAM_2_FINRA_DARKPOOL"),
    ("f17_days_to_cover_squeeze_z", "STREAM_2_FINRA_DARKPOOL"),
    ("f18_cumulative_float_turnover", "STREAM_2_FINRA_DARKPOOL"),
    ("f19_intraday_short_covering_flow", "STREAM_2_FINRA_DARKPOOL"),
    ("f20_nyse_tick_impulse_tide", "STREAM_3_MACRO_INTERNALS"),
    ("f21_nasdaq_tick_divergence", "STREAM_3_MACRO_INTERNALS"),
    ("f22_advance_decline_breadth", "STREAM_3_MACRO_INTERNALS"),
    ("f23_vold_up_down_volume_tide", "STREAM_3_MACRO_INTERNALS"),
    ("f24_vix_term_structure_slope", "STREAM_3_MACRO_INTERNALS"),
    ("f25_sector_relative_rotation", "STREAM_3_MACRO_INTERNALS"),
    ("f26_cboe_net_dealer_gamma_gex", "STREAM_4_CBOE_OPTIONS"),
    ("f27_failed_pop_gamma_wall_reject", "STREAM_4_CBOE_OPTIONS"),
    ("f28_open_slam_sweep_aggression", "STREAM_4_CBOE_OPTIONS"),
    ("f29_net_dealer_vanna_exposure", "STREAM_4_CBOE_OPTIONS"),
    ("f30_zero_dte_volume_concentration", "STREAM_4_CBOE_OPTIONS"),
    ("f31_front_back_iv_skew_ratio", "STREAM_4_CBOE_OPTIONS"),
];

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SynapticFeatureMutation {
    pub feature_index: usize,
    pub feature_name: String,
    pub stream_id: String,
    pub gen0_weight: f64,
    pub current_weight: f64,
    pub delta_pct: f64,
    /// `"BOOSTED_WARNING_SENSOR"`, `"SLASHED_DECEIVER_FEATURE"`, or `"NEUTRAL_ANCHOR"`
    pub role_classification: String,
}

fn default_eb_lambda() -> f64 {
    0.50
}

fn default_mid_sprt_thresholds() -> [f64; 3] {
    [1.48, 1.44, 1.40]
}

fn default_mid_mfe_norm_quantile() -> f64 {
    0.58
}

fn default_mid_stop_ratchet_r() -> f64 {
    -0.19
}

fn default_itm_mfe_activation_r() -> f64 {
    1.04
}

fn default_giveback_exhaustion_threshold() -> f64 {
    1.38
}

fn default_runner_continuation_threshold() -> f64 {
    -0.25
}

fn default_giveback_retention_ratio() -> f64 {
    0.81
}

fn default_profit_lock_activation_r() -> f64 {
    0.44
}

fn default_house_money_reinvestment_fraction() -> f64 {
    0.34
}

fn default_late_blowoff_scratch_threshold() -> f64 {
    1.85
}

fn default_shakeout_absorption_quantile() -> f64 {
    0.98
}

fn default_cross_brain_recycle_multiplier() -> f64 {
    1.00
}

fn default_almgren_chriss_risk_aversion_lambda() -> f64 {
    0.18
}

fn default_almgren_chriss_urgency_kappa() -> f64 {
    0.42
}

fn default_correlation_spike_quantile() -> f64 {
    0.52
}

fn default_vanna_tail_hedge_ratio_theta() -> f64 {
    0.00
}

fn default_sor_venue_weights_4() -> [f64; 4] {
    [0.25, 0.25, 0.25, 0.25]
}

fn default_sor_markout_toxicity_bps() -> f64 {
    0.94
}

fn default_charm_vanna_pin_quantile() -> f64 {
    0.32
}

fn default_charm_vanna_unpin_quantile() -> f64 {
    0.68
}

fn default_occ_tims_sigma_shock_pct() -> f64 {
    0.115
}

fn default_asymmetric_hurdle_prob_p_star() -> f64 {
    0.56
}

fn default_asymmetric_hurdle_llr_lambda_star() -> f64 {
    0.88
}

fn default_conformal_alpha() -> f64 {
    0.05
}

fn default_conformal_veto_quantile() -> f64 {
    0.958
}

fn default_conformal_early_scratch_quantile() -> f64 {
    1.46
}

fn default_conformal_mid_scratch_quantile() -> f64 {
    1.44
}

fn default_fisher_sparsity_exponent_p() -> f64 {
    1.35
}

fn default_velocity_absorption_damping_weight() -> f64 {
    0.62
}

fn default_recovery_velocity_guard_quantile() -> f64 {
    0.020
}

fn default_pareto_gate_passed() -> bool {
    true
}

/// Per-Brain evolving synaptic weight & boundary plasticity state (`Phase 18`..`Phase 25`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BrainSynapticPlasticityState {
    pub brain_id: BrainHorizonId,
    pub brain_name: String,
    pub generation_id: u32,
    pub synaptic_feature_weights_32: [f64; 32],
    pub stream_prior_multipliers: [f64; 4],
    /// Sequential stopping boundaries $[A_1, A_2, A_3]$ across `Bar +1, Bar +2, Bar +3`
    pub wald_sprt_thresholds: [f64; 3],
    /// Minimum similarity margin $S_{\text{trap}} - S_{\text{win}}$ required to trigger an entry veto
    pub entry_veto_margin_delta: f64,
    /// Adaptive gravity conviction weight per causal Hard-Negative Trap ID (initialized to `1.0`)
    pub trap_conviction_weights: BTreeMap<String, f64>,
    /// Phase 19 Upgrade 1: Empirical-Bayes Hierarchical Shrinkage factor
    /// $$\lambda_b = \frac{N_{\text{causal}, b}}{N_{\text{causal}, b} + \sigma^2_{\text{within}, b} / \sigma^2_{\text{between}}}$$
    #[serde(default = "default_eb_lambda")]
    pub empirical_bayes_shrinkage_lambda: f64,
    /// Phase 19 Upgrade 2: Self-learned mid-trade (`Bar +4, Bar +5, Bar +6`) sequential stall stopping boundaries $[A_{\text{mid}, 4}, A_{\text{mid}, 5}, A_{\text{mid}, 6}]$
    #[serde(default = "default_mid_sprt_thresholds")]
    pub mid_trade_wald_sprt_thresholds: [f64; 3],
    /// Phase 19 Upgrade 2: Self-learned normalized cumulative MFE stall quantile (relative to Brain's Era-1 median MFE $\tilde{\mu}_{\text{MFE}, k}$)
    #[serde(default = "default_mid_mfe_norm_quantile")]
    pub learned_mid_trade_mfe_norm_quantile: f64,
    /// Phase 19 Upgrade 2: Self-learned 80th-percentile adverse excursion exit R quantile from causal mid-trade stalls (zero hardcoded `-0.35R` or `+0.40R` constants)
    #[serde(default = "default_mid_stop_ratchet_r")]
    pub learned_mid_trade_stop_ratchet_r: f64,
    /// Phase 20 Upgrade 1: Self-learned in-the-money MFE activation quantile ($q_{\text{ITM}, b}^*$) from causal completed MFE distributions (`Bar +5..+16`)
    #[serde(default = "default_itm_mfe_activation_r")]
    pub learned_itm_mfe_activation_r: f64,
    /// Phase 20 Upgrade 1: Evolving Dual $32 \times 32$ Ledoit-Wolf profit-giveback exhaustion boundary ($A_{\text{giveback}, b}^{(g)}$)
    #[serde(default = "default_giveback_exhaustion_threshold")]
    pub giveback_exhaustion_threshold: f64,
    /// Phase 20 Upgrade 1: Evolving Dual $32 \times 32$ Ledoit-Wolf multi-sigma runner continuation boundary ($B_{\text{runner}, b}^{(g)}$)
    #[serde(default = "default_runner_continuation_threshold")]
    pub runner_continuation_threshold: f64,
    /// Phase 20 Upgrade 1: Self-learned empirical MFE retention lock quantile ($\rho_{\text{lock}, b}^*$) from causal winner retention distributions
    #[serde(default = "default_giveback_retention_ratio")]
    pub learned_giveback_retention_ratio: f64,
    /// Phase 21 Upgrade 2: Self-learned empirical profit-lock activation quantile ($q_{\text{lock}, b}^*$) from causal runner retention distributions
    #[serde(default = "default_profit_lock_activation_r")]
    pub learned_profit_lock_activation_r: f64,
    /// Phase 21 Upgrade 2 & 3: Self-learned Principal-Protected House-Money reinvestment & CBOE Gamma-Wall roll-up fraction ($\phi_b^{(g)} \in [0.18, 0.55]$)
    #[serde(default = "default_house_money_reinvestment_fraction")]
    pub learned_house_money_reinvestment_fraction: f64,
    /// Phase 21 Upgrade 1: Self-learned Late-Horizon (`Bar +9..+16`) Failed-Pop Blow-Off Reversal LLR boundary ($A_{\text{blowoff}, b}^{(g)}$)
    #[serde(default = "default_late_blowoff_scratch_threshold")]
    pub late_blowoff_scratch_threshold: f64,
    /// Phase 22 Upgrade 1: Self-learned Cross-Stream Shakeout Absorption quantile ($q_{\text{absorption}, b}^*$) at `Bar+2..+3` (`Gen 17..Gen 20`)
    #[serde(default = "default_shakeout_absorption_quantile")]
    pub learned_shakeout_absorption_quantile: f64,
    /// Phase 22 Upgrade 2: Self-learned Cross-Brain Freed-Margin Capital Recycling multiplier ($\gamma_{\text{recycle}, b}^{(g)} \in [1.00, 1.35]$)
    #[serde(default = "default_cross_brain_recycle_multiplier")]
    pub learned_cross_brain_recycle_multiplier: f64,
    /// Phase 23 Upgrade 1: Self-learned Almgren-Chriss execution risk-aversion urgency multiplier ($\lambda_{\text{risk}, b}^{(g)} \in [0.12, 0.92]$)
    #[serde(default = "default_almgren_chriss_risk_aversion_lambda")]
    pub learned_almgren_chriss_risk_aversion_lambda: f64,
    /// Phase 23 Upgrade 1: Self-learned Almgren-Chriss hyperbolic execution urgency parameter ($\kappa_b^{(g)} = \sqrt{\lambda_{\text{risk}, b}^{(g)} \sigma_{\text{bar}}^2 / \eta_{\text{temp}}}$)
    #[serde(default = "default_almgren_chriss_urgency_kappa")]
    pub learned_almgren_chriss_urgency_kappa: f64,
    /// Phase 23 Upgrade 2: Self-learned empirical cross-position Ledoit-Wolf correlation spike activation quantile ($q_{\text{corr}, b}^*$)
    #[serde(default = "default_correlation_spike_quantile")]
    pub learned_correlation_spike_quantile: f64,
    /// Phase 23 Upgrade 2: Self-learned Minimum-Variance Vanna-Convexity Tail-Hedge Ratio ($\theta_{\text{hedge}, b}^{(g)}$)
    #[serde(default = "default_vanna_tail_hedge_ratio_theta")]
    pub learned_vanna_tail_hedge_ratio_theta: f64,
    /// Phase 24 Upgrade 1: Self-learned Cross-Venue NBBO Smart Order Router simplex weights $w_{\text{venue}, b}^{(g)} \in \Delta^4$
    /// across `[IEX_DLIMIT, IBKR_PEG_MID, MEMX_MIDPOINT, CBOE_EDGX_LIT]` solved via Exponential-Weights Mirror-Descent
    #[serde(default = "default_sor_venue_weights_4")]
    pub learned_sor_venue_weights_4: [f64; 4],
    /// Phase 24 Upgrade 1: Self-learned expected post-fill adverse selection markout toxicity (bps) $\widehat{\mathcal{M}}_{b}^{(g)}$
    #[serde(default = "default_sor_markout_toxicity_bps")]
    pub learned_sor_markout_toxicity_bps: f64,
    /// Phase 24 Upgrade 2: Self-learned Dealer Charm-Vanna Gamma Pinning quantile $Q_{\alpha_{\text{pin}, b}^{(g)}}(\Upsilon_{0..t})$
    #[serde(default = "default_charm_vanna_pin_quantile")]
    pub learned_charm_vanna_pin_quantile: f64,
    /// Phase 24 Upgrade 2: Self-learned Dealer Charm-Vanna Unpinning Squeeze quantile $Q_{1 - \alpha_{\text{unpin}, b}^{(g)}}(\Upsilon_{0..t})$
    #[serde(default = "default_charm_vanna_unpin_quantile")]
    pub learned_charm_vanna_unpin_quantile: f64,
    /// Phase 24 Upgrade 3: Self-learned OCC TIMS scenario base volatility shock $\hat{\sigma}_{\text{LW}, b, 0..t}$ from $32 \times 32$ Ledoit-Wolf covariance
    #[serde(default = "default_occ_tims_sigma_shock_pct")]
    pub learned_occ_tims_sigma_shock_pct: f64,
    /// Phase 25 Deliverable A: Causal Asymmetric Hurdle Probability $p_{\text{hurdle}, b, t}^{\star}$
    #[serde(default = "default_asymmetric_hurdle_prob_p_star")]
    pub learned_asymmetric_hurdle_prob_p_star: f64,
    /// Phase 25 Deliverable A: Causal Asymmetric Hurdle Log-Likelihood Ratio $\Lambda_{\text{hurdle}, b, t}^{\star}$
    #[serde(default = "default_asymmetric_hurdle_llr_lambda_star")]
    pub learned_asymmetric_hurdle_llr_lambda_star: f64,
    /// Phase 25 Deliverable A: Mirror-Descent Conformal False-Positive Risk Level $\alpha_g$
    #[serde(default = "default_conformal_alpha")]
    pub learned_conformal_alpha: f64,
    /// Phase 25 Deliverable A: Conformal Stage 0 Veto Upper Quantile $Q_{1 - \alpha_g}(\mathcal{W}_{b, 0..t})$
    #[serde(default = "default_conformal_veto_quantile")]
    pub learned_conformal_veto_quantile: f64,
    /// Phase 25 Deliverable A: Conformal Stage 1 Early Scratch Upper Quantile
    #[serde(default = "default_conformal_early_scratch_quantile")]
    pub learned_conformal_early_scratch_quantile: f64,
    /// Phase 25 Deliverable A: Conformal Stage 2 Mid-Trade Scratch Upper Quantile
    #[serde(default = "default_conformal_mid_scratch_quantile")]
    pub learned_conformal_mid_scratch_quantile: f64,
    /// Phase 25 Deliverable C: Sparse Fisher-Mahalanobis SNR Exponent $p_g$
    #[serde(default = "default_fisher_sparsity_exponent_p")]
    pub learned_fisher_sparsity_exponent_p: f64,
    /// Phase 25 Deliverable B/C: Sequential Velocity/Absorption Damping Weight
    #[serde(default = "default_velocity_absorption_damping_weight")]
    pub learned_velocity_absorption_damping_weight: f64,
    /// Phase 25 Deliverable B: Recovering-Winner Velocity Guard Quantile $q_{v, \text{rec}, b, t}$
    #[serde(default = "default_recovery_velocity_guard_quantile")]
    pub learned_recovery_velocity_guard_quantile: f64,
    pub top_boosted_warning_features: Vec<SynapticFeatureMutation>,
    pub top_slashed_deceiver_features: Vec<SynapticFeatureMutation>,
}

impl BrainSynapticPlasticityState {
    pub fn new_gen0(brain_id: BrainHorizonId) -> Self {
        let (itm_mfe_q, ret_ratio): (f64, f64) = match brain_id {
            BrainHorizonId::Brain01PennyAnomaly => (1.01, 0.80),
            BrainHorizonId::Brain02IntradayDayTrade => (0.94, 0.80),
            BrainHorizonId::Brain03ShortTermSwing => (1.18, 0.81),
            BrainHorizonId::Brain04CoreSwing => (1.08, 0.82),
            BrainHorizonId::Brain05LongTermWealth => (1.01, 0.83),
        };
        let lock_act_q = ((itm_mfe_q * ret_ratio * 0.52_f64) * 1000.0).round() / 1000.0;
        let hm_phi_0 = ((1.0_f64 - ret_ratio) * 1.45_f64).clamp(0.22, 0.32);
        Self {
            brain_id,
            brain_name: brain_id.display_name().to_string(),
            generation_id: 0,
            synaptic_feature_weights_32: [1.0; 32],
            stream_prior_multipliers: [1.0; 4],
            wald_sprt_thresholds: [1.35, 1.45, 1.52],
            entry_veto_margin_delta: 0.025,
            trap_conviction_weights: BTreeMap::new(),
            empirical_bayes_shrinkage_lambda: 0.50,
            mid_trade_wald_sprt_thresholds: [1.48, 1.44, 1.40],
            learned_mid_trade_mfe_norm_quantile: 0.58,
            learned_mid_trade_stop_ratchet_r: -0.19,
            learned_itm_mfe_activation_r: itm_mfe_q,
            giveback_exhaustion_threshold: 1.38,
            runner_continuation_threshold: -0.25,
            learned_giveback_retention_ratio: ret_ratio,
            learned_profit_lock_activation_r: lock_act_q,
            learned_house_money_reinvestment_fraction: (hm_phi_0 * 1000.0).round() / 1000.0,
            late_blowoff_scratch_threshold: 1.85,
            learned_shakeout_absorption_quantile: 0.98,
            learned_cross_brain_recycle_multiplier: 1.00,
            learned_almgren_chriss_risk_aversion_lambda: 0.18,
            learned_almgren_chriss_urgency_kappa: 0.42,
            learned_correlation_spike_quantile: 0.52,
            learned_vanna_tail_hedge_ratio_theta: 0.00,
            learned_sor_venue_weights_4: [0.25, 0.25, 0.25, 0.25],
            learned_sor_markout_toxicity_bps: 0.94,
            learned_charm_vanna_pin_quantile: 0.32,
            learned_charm_vanna_unpin_quantile: 0.68,
            learned_occ_tims_sigma_shock_pct: 0.115,
            learned_asymmetric_hurdle_prob_p_star: 0.56,
            learned_asymmetric_hurdle_llr_lambda_star: 0.88,
            learned_conformal_alpha: 0.05,
            learned_conformal_veto_quantile: 0.92,
            learned_conformal_early_scratch_quantile: 1.46,
            learned_conformal_mid_scratch_quantile: 1.44,
            learned_fisher_sparsity_exponent_p: 1.35,
            learned_velocity_absorption_damping_weight: 0.45,
            learned_recovery_velocity_guard_quantile: 0.020,
            top_boosted_warning_features: Vec::new(),
            top_slashed_deceiver_features: Vec::new(),
        }
    }

    pub fn refresh_inspector_rankings(&mut self) {
        let mut mutations: Vec<SynapticFeatureMutation> = self
            .synaptic_feature_weights_32
            .iter()
            .enumerate()
            .map(|(idx, &w)| {
                let (name, stream) = CANONICAL_32_FEATURE_METADATA[idx];
                let delta_pct = (((w - 1.0) * 100.0) * 100.0).round() / 100.0;
                let role = if w >= 1.05 {
                    "BOOSTED_WARNING_SENSOR"
                } else if w <= 0.95 {
                    "SLASHED_DECEIVER_FEATURE"
                } else {
                    "NEUTRAL_ANCHOR"
                };
                SynapticFeatureMutation {
                    feature_index: idx,
                    feature_name: name.to_string(),
                    stream_id: stream.to_string(),
                    gen0_weight: 1.0,
                    current_weight: (w * 10000.0).round() / 10000.0,
                    delta_pct,
                    role_classification: role.to_string(),
                }
            })
            .collect();

        mutations.sort_by(|a, b| {
            b.current_weight
                .partial_cmp(&a.current_weight)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        self.top_boosted_warning_features = mutations.iter().take(6).cloned().collect();

        mutations.sort_by(|a, b| {
            a.current_weight
                .partial_cmp(&b.current_weight)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        self.top_slashed_deceiver_features = mutations.iter().take(6).cloned().collect();
    }
}

/// Complete audit record for a single generation (`Gen 0 .. Gen 24`)
/// in the Autonomous Multi-Generation Loss-Evolution loop.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvolutionGenerationRecord {
    pub generation_id: u32,
    pub status_label: String,
    pub is_current_champion: bool,
    pub timestamp_iso: String,
    pub active_trades: usize,
    pub wins: usize,
    pub losses: usize,
    pub tp_losses_vetoed: usize,
    pub tp_losses_scratched: usize,
    #[serde(default)]
    pub early_scratches_bar1_to_3: usize,
    #[serde(default)]
    pub mid_trade_stall_saves_bar4_to_6: usize,
    pub fp_winners_vetoed: usize,
    pub fp_winners_scratched: usize,
    #[serde(default)]
    pub fp_mid_trade_scratches: usize,
    #[serde(default)]
    pub giveback_saves_count: usize,
    #[serde(default)]
    pub total_giveback_r_recovered: f64,
    #[serde(default)]
    pub house_money_scaled_runners_count: usize,
    #[serde(default)]
    pub total_house_money_r_gained: f64,
    #[serde(default)]
    pub recycled_risk_pool_usd: f64,
    #[serde(default)]
    pub recycled_runner_boost_count: usize,
    #[serde(default)]
    pub recycled_additional_r_compounded: f64,
    #[serde(default)]
    pub avg_execution_slippage_bps: f64,
    #[serde(default)]
    pub slippage_saved_usd: f64,
    #[serde(default)]
    pub tail_hedge_ratio_theta: f64,
    #[serde(default)]
    pub active_tail_hedge_count: usize,
    #[serde(default)]
    pub tail_hedge_convexity_payout_usd: f64,
    #[serde(default)]
    pub sor_markout_toxicity_bps: f64,
    #[serde(default)]
    pub charm_vanna_unpin_boosts_count: usize,
    #[serde(default)]
    pub charm_vanna_additional_r_gained: f64,
    #[serde(default)]
    pub occ_tims_pm_relief_ratio_pct: f64,
    #[serde(default)]
    pub return_on_margin_rom_pct: f64,
    #[serde(default = "default_pareto_gate_passed")]
    pub pareto_gate_passed: bool,
    #[serde(default)]
    pub avg_winner_r: f64,
    pub remaining_unscratched_losses: usize,
    pub untouched_winners_count: usize,
    pub entry_veto_precision_pct: f64,
    pub scratch_precision_pct: f64,
    pub win_rate_pct: f64,
    pub profit_factor: f64,
    pub kelly_profit_factor: f64,
    pub max_drawdown_usd: f64,
    pub kelly_max_drawdown_usd: f64,
    pub fixed_net_pnl_usd: f64,
    pub kelly_net_pnl_usd: f64,
    /// $\text{Fitness}(G) = \text{KellyNetPnL}(G) - 1.5 \times \text{MaxDrawdown}(G) + 1500 \times \text{ProfitFactor}(G)$
    pub composite_fitness_score: f64,
    pub mutations_applied_summary: Vec<String>,
}

impl EvolutionGenerationRecord {
    pub fn compute_composite_fitness(
        kelly_net_pnl_usd: f64,
        max_drawdown_usd: f64,
        profit_factor: f64,
    ) -> f64 {
        let raw = kelly_net_pnl_usd - 1.5 * max_drawdown_usd + 1500.0 * profit_factor;
        (raw * 100.0).round() / 100.0
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MultiGenEvolutionResponse {
    pub current_champion_generation: u32,
    pub generations_history: Vec<EvolutionGenerationRecord>,
    pub synaptic_plasticity_states: Vec<BrainSynapticPlasticityState>,
}
