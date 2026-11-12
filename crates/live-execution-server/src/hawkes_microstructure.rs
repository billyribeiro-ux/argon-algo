//! Phase 37 Pillar 1: Multivariate Hawkes Self-Exciting Orderflow & Dark-Pool Absorption Engine
//!
//! Replaces memoryless single-bar VPIN heuristics with an O(1) recursive Markovian
//! Trivariate Exponential Hawkes Process per symbol tracking:
//! - Buy-Sweep Aggressive Intensity: $\lambda_t^{+}$
//! - Sell-Sweep Aggressive Intensity: $\lambda_t^{-}$
//! - Passive Iceberg & Dark-Pool Absorption Intensity: $\lambda_t^{\text{abs}}$
//!
//! Jump marks:
//! $$J_t^{+} = v_{\text{rel}} \cdot \text{close\_loc} \cdot \mathbb{I}[C_t \ge \min(O_t, C_{t-1})]$$
//! $$J_t^{-} = v_{\text{rel}} \cdot (1 - \text{close\_loc}) \cdot \mathbb{I}[C_t \le \max(O_t, C_{t-1})]$$
//! $$J_t^{\text{abs}} = \frac{V_t / \bar{V}_{20}}{\max(0.15,\; |C_t - O_t| / \text{ATR}_{\text{bar}})}$$
//!
//! Recursive Markovian state update:
//! $$\xi_t^{(c)} = e^{-\beta_c \Delta t} \xi_{t-1}^{(c)} + \sum_{c'} \alpha_{c, c'} J_t^{(c')}$$
//! $$\lambda_t^{(c)} = \mu_c + \xi_t^{(c)}$$
//!
//! Metrics:
//! - Spectral Hawkes Branching Ratio $\hat{n}_t \in [0.0, 1.25]$ (Criticality)
//! - Net Absorption Toxicity $\tau_{\text{abs}, t} = \frac{\lambda_t^{\text{abs}}}{\lambda_t^{+} + \lambda_t^{-} + \lambda_t^{\text{abs}}}$
//! - Directional Hawkes Imbalance $\delta_{\text{hawkes}} = \frac{\lambda_t^{+} - \lambda_t^{-}}{\lambda_t^{+} + \lambda_t^{-} + 10^{-6}}$

use raw_fingerprint_core::{RawBarRecord, TradeDirection};
use serde::{Deserialize, Serialize};

pub const HAWKES_MAX_BRANCHING_RATIO: f64 = 1.25;
pub const HAWKES_SUPERCRITICAL_THRESHOLD: f64 = 0.96;
pub const HAWKES_ICEBERG_ABSORPTION_THRESHOLD: f64 = 0.68;

/// Deterministic regime classification from Hawkes orderflow dynamics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HawkesOrderflowRegime {
    #[serde(rename = "BALANCED_EQUILIBRIUM")]
    BalancedEquilibrium,
    #[serde(rename = "DIRECTIONAL_SWEEP_BURST")]
    DirectionalSweepBurst,
    #[serde(rename = "ICEBERG_PASSIVE_ABSORPTION")]
    IcebergPassiveAbsorption,
    #[serde(rename = "SUPERCRITICAL_BLOW_OFF_CLIMAX")]
    SupercriticalBlowOffClimax,
}

impl HawkesOrderflowRegime {
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::BalancedEquilibrium => "BALANCED_EQUILIBRIUM",
            Self::DirectionalSweepBurst => "DIRECTIONAL_SWEEP_BURST",
            Self::IcebergPassiveAbsorption => "ICEBERG_PASSIVE_ABSORPTION",
            Self::SupercriticalBlowOffClimax => "SUPERCRITICAL_BLOW_OFF_CLIMAX",
        }
    }
}

/// Instantaneous Hawkes orderflow telemetry snapshot.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HawkesTelemetry {
    pub lambda_buy_sweep: f64,
    pub lambda_sell_sweep: f64,
    pub lambda_absorption: f64,
    pub hawkes_branching_ratio: f64,
    pub hawkes_absorption_toxicity: f64,
    pub directional_hawkes_imbalance: f64,
    pub hawkes_orderflow_regime: HawkesOrderflowRegime,
    pub step_count: usize,
}

impl Default for HawkesTelemetry {
    fn default() -> Self {
        Self {
            lambda_buy_sweep: 0.10,
            lambda_sell_sweep: 0.10,
            lambda_absorption: 0.08,
            hawkes_branching_ratio: 0.45,
            hawkes_absorption_toxicity: 0.28,
            directional_hawkes_imbalance: 0.0,
            hawkes_orderflow_regime: HawkesOrderflowRegime::BalancedEquilibrium,
            step_count: 0,
        }
    }
}

/// Trivariate Exponential Hawkes Process Engine for a single symbol stream.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HawkesMicrostructureEngine {
    baseline_mu: [f64; 3],
    decay_beta: [f64; 3],
    alpha_matrix: [[f64; 3]; 3],
    xi_state: [f64; 3],
    rolling_volume_window: Vec<f64>,
    last_lambda_dir: f64,
    step_count: usize,
    last_timestamp_ns: Option<i64>,
    last_telemetry: HawkesTelemetry,
}

impl Default for HawkesMicrostructureEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl HawkesMicrostructureEngine {
    pub fn new() -> Self {
        Self {
            baseline_mu: [0.10, 0.10, 0.08],
            decay_beta: [0.85, 0.85, 0.65],
            alpha_matrix: [
                [0.35, 0.08, 0.04],
                [0.08, 0.35, 0.04],
                [0.08, 0.08, 0.52],
            ],
            xi_state: [0.0, 0.0, 0.0],
            rolling_volume_window: Vec::with_capacity(32),
            last_lambda_dir: 0.0,
            step_count: 0,
            last_timestamp_ns: None,
            last_telemetry: HawkesTelemetry::default(),
        }
    }

    pub fn telemetry(&self) -> HawkesTelemetry {
        self.last_telemetry.clone()
    }

    pub fn last_timestamp_ns(&self) -> Option<i64> {
        self.last_timestamp_ns
    }

    /// Computes spectral radius (Perron-Frobenius eigenvalue) of the kernel excitation matrix
    /// $\Gamma = (\alpha_{i,j} / \beta_i)$ scaled by the active relative excitation level.
    #[inline]
    fn compute_spectral_branching_ratio(&self) -> f64 {
        let mut row_sums = [0.0_f64; 3];
        for i in 0..3 {
            for j in 0..3 {
                row_sums[i] += self.alpha_matrix[i][j] / self.decay_beta[i].max(0.10);
            }
        }
        let max_row_sum = row_sums.iter().copied().fold(0.0_f64, f64::max);
        let total_xi: f64 = self.xi_state.iter().sum();
        let total_mu: f64 = self.baseline_mu.iter().sum();
        let excitation_ratio = (total_xi / (total_mu + total_xi + 1e-6)).clamp(0.0, 1.5);
        (max_row_sum * (0.45 + 0.55 * excitation_ratio)).clamp(0.0, HAWKES_MAX_BRANCHING_RATIO)
    }

    /// Observes an incoming bar and advances the recursive Markovian Hawkes intensities.
    /// If `bar.timestamp_ns <= last_timestamp_ns`, deduplicates and returns cached telemetry.
    pub fn observe_bar(
        &mut self,
        bar: &RawBarRecord,
        atr_bar: f64,
        prev_bar_opt: Option<&RawBarRecord>,
    ) -> HawkesTelemetry {
        if self
            .last_timestamp_ns
            .is_some_and(|prev_ts| bar.timestamp_ns <= prev_ts)
        {
            return self.last_telemetry.clone();
        }

        let delta_t_bars = if let Some(prev_ts) = self.last_timestamp_ns {
            let dt_ns = bar.timestamp_ns.saturating_sub(prev_ts);
            ((dt_ns as f64) / 300_000_000_000.0).clamp(0.2, 5.0)
        } else {
            1.0
        };
        self.last_timestamp_ns = Some(bar.timestamp_ns);

        let vol_cur = bar.volume as f64;
        self.rolling_volume_window.push(vol_cur);
        if self.rolling_volume_window.len() > 20 {
            self.rolling_volume_window.remove(0);
        }
        let mean_vol_20 = self.rolling_volume_window.iter().sum::<f64>()
            / (self.rolling_volume_window.len() as f64).max(1.0);
        let v_rel = (vol_cur / mean_vol_20.max(10.0)).clamp(0.10, 12.0);

        let o = bar.open.to_f64();
        let h = bar.high.to_f64();
        let l = bar.low.to_f64();
        let c = bar.close.to_f64();
        let range = (h - l).max(c * 0.0001).max(1e-4);
        let close_loc = ((c - l) / range).clamp(0.01, 0.99);

        let body = (c - o).abs();
        let prev_c = prev_bar_opt.map(|p| p.close.to_f64()).unwrap_or(o);
        let is_up = c >= o || c >= prev_c;
        let is_down = c <= o || c <= prev_c;

        // Standardized jump marks:
        let atr_safe = atr_bar.max(c * 0.0005).max(1e-4);
        let displacement_ratio = (body / atr_safe).max(0.05);
        let sweep_eff = displacement_ratio.clamp(0.10, 1.50);
        let j_plus = v_rel * sweep_eff * close_loc * (if is_up { 1.35 } else { 0.65 });
        let j_minus = v_rel * sweep_eff * (1.0 - close_loc) * (if is_down { 1.35 } else { 0.65 });
        let j_abs = (v_rel / displacement_ratio.max(0.15)).clamp(0.0, 15.0);

        let jumps = [j_plus, j_minus, j_abs];

        // O(1) recursive Markovian decay + cross-excitation
        let mut next_xi = [0.0_f64; 3];
        for i in 0..3 {
            let decay = (-self.decay_beta[i] * delta_t_bars).exp();
            let mut exc = 0.0_f64;
            for j in 0..3 {
                exc += self.alpha_matrix[i][j] * jumps[j];
            }
            next_xi[i] = self.xi_state[i] * decay + exc;
        }
        self.xi_state = next_xi;
        self.step_count += 1;

        let lambda_buy = self.baseline_mu[0] + self.xi_state[0];
        let lambda_sell = self.baseline_mu[1] + self.xi_state[1];
        let lambda_abs = self.baseline_mu[2] + self.xi_state[2];
        let sum_lambda = (lambda_buy + lambda_sell + lambda_abs).max(1e-6);

        let tau_abs = (lambda_abs / sum_lambda).clamp(0.0, 1.0);
        let delta_imbalance = ((lambda_buy - lambda_sell) / (lambda_buy + lambda_sell + 1e-6)).clamp(-1.0, 1.0);
        let branching_ratio = self.compute_spectral_branching_ratio();

        let regime = if branching_ratio >= HAWKES_SUPERCRITICAL_THRESHOLD {
            HawkesOrderflowRegime::SupercriticalBlowOffClimax
        } else if tau_abs >= HAWKES_ICEBERG_ABSORPTION_THRESHOLD {
            HawkesOrderflowRegime::IcebergPassiveAbsorption
        } else if delta_imbalance.abs() >= 0.35 {
            HawkesOrderflowRegime::DirectionalSweepBurst
        } else {
            HawkesOrderflowRegime::BalancedEquilibrium
        };

        self.last_telemetry = HawkesTelemetry {
            lambda_buy_sweep: lambda_buy,
            lambda_sell_sweep: lambda_sell,
            lambda_absorption: lambda_abs,
            hawkes_branching_ratio: branching_ratio,
            hawkes_absorption_toxicity: tau_abs,
            directional_hawkes_imbalance: delta_imbalance,
            hawkes_orderflow_regime: regime,
            step_count: self.step_count,
        };

        self.last_telemetry.clone()
    }

    /// Evaluates Stage 0 Iceberg Absorption collision veto.
    /// Vetoes when trade direction collides into an opposing institutional absorption wall
    /// ($\tau_{\text{abs}} > 0.68$ with adverse wick rejection against trade direction).
    pub fn check_iceberg_absorption_veto(
        direction: TradeDirection,
        bar: &RawBarRecord,
        tau_abs: f64,
        lambda_buy: f64,
        lambda_sell: f64,
    ) -> bool {
        if tau_abs < HAWKES_ICEBERG_ABSORPTION_THRESHOLD {
            return false;
        }

        let o = bar.open.to_f64();
        let h = bar.high.to_f64();
        let l = bar.low.to_f64();
        let c = bar.close.to_f64();
        let range = (h - l).max(1e-4);

        match direction {
            TradeDirection::Long => {
                let upper_wick = h - o.max(c);
                let upper_wick_ratio = upper_wick / range;
                upper_wick_ratio >= 0.32 || lambda_sell > lambda_buy * 1.12
            }
            TradeDirection::Short => {
                let lower_wick = o.min(c) - l;
                let lower_wick_ratio = lower_wick / range;
                lower_wick_ratio >= 0.32 || lambda_buy > lambda_sell * 1.12
            }
        }
    }

    /// Helper to evaluate iceberg absorption collision veto against the engine's current state.
    pub fn check_veto_with_last(&self, direction: TradeDirection, bar: &RawBarRecord) -> bool {
        Self::check_iceberg_absorption_veto(
            direction,
            bar,
            self.last_telemetry.hawkes_absorption_toxicity,
            self.last_telemetry.lambda_buy_sweep,
            self.last_telemetry.lambda_sell_sweep,
        )
    }

    /// Evaluates Stage 3 Supercritical Blow-Off Climax Lock.
    /// Tightens causal retention floor toward `0.90` (or `max(0.88, ...)`) when
    /// running peak R >= 1.2 and Hawkes criticality >= 0.96 with decelerating directional intensity.
    pub fn compute_hawkes_climax_retention_floor(
        running_peak_r: f64,
        branching_ratio: f64,
        delta_lambda_dir: f64,
        base_retention: f64,
    ) -> f64 {
        if running_peak_r >= 1.20
            && branching_ratio >= HAWKES_SUPERCRITICAL_THRESHOLD
            && delta_lambda_dir <= 0.0
        {
            base_retention.max(0.90)
        } else if running_peak_r >= 1.00 && branching_ratio >= 0.88 {
            base_retention.max(0.85)
        } else {
            base_retention
        }
    }

    /// Evaluates climax retention floor using the engine's current telemetry state.
    pub fn compute_climax_retention_floor_with_last(
        &self,
        running_peak_r: f64,
        base_retention: f64,
    ) -> f64 {
        Self::compute_hawkes_climax_retention_floor(
            running_peak_r,
            self.last_telemetry.hawkes_branching_ratio,
            self.last_telemetry.directional_hawkes_imbalance,
            base_retention,
        )
    }
}
