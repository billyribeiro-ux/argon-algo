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

/// Per-Brain evolving synaptic weight & boundary plasticity state (`Phase 18` & `Phase 19`).
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
    pub top_boosted_warning_features: Vec<SynapticFeatureMutation>,
    pub top_slashed_deceiver_features: Vec<SynapticFeatureMutation>,
}

impl BrainSynapticPlasticityState {
    pub fn new_gen0(brain_id: BrainHorizonId) -> Self {
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

/// Complete audit record for a single generation (`Gen 0 .. Gen 8`)
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
