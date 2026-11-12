//! Phase 37 Pillar 3: 1-Wasserstein Distributionally Robust Kelly-CVaR Portfolio Optimizer
//!
//! Replaces ad-hoc multiplication of separate Kelly throttles with a unified convex program
//! over empirical BOCPD-weighted returns with a 1-Wasserstein ambiguity ball of radius $\varepsilon_t$:
//!
//! $$\varepsilon_t = \varepsilon_0 \cdot \left(1.0 + 1.4 \cdot p_{\text{cp, eff}} + 1.2 \cdot \text{ood} + 0.8 \cdot (1 - p_{\text{mondrian}}) + 0.5 \cdot \ln\max(1.0, M_t^{\text{ville}})\right)$$
//!
//! Master DRO Optimization Problem:
//! $$f_{\text{DRO}}^* = \arg\max_{f \in [f_{\min}, f_{\max}]} \left\{ \sum_{i=1}^M w_i \ln(1 + f \tilde{R}_i) - \varepsilon_t \frac{f}{1 - f |\tilde{R}_{\min}|} - \frac{\lambda_{\text{eigen}}}{2} (1 - \kappa_{\text{eigen}}) f^2 \right\}$$
//! subject to:
//! $$\text{BookHeat}_{\text{open}} + f \cdot \widehat{\text{CVaR}}_{0.95}(L) \le \text{MaxPortfolioCvarCap}$$
//!
//! Properties:
//! 1. Monotonic Comparative Statics: $\frac{\partial f^*}{\partial \varepsilon} < 0$ strictly.
//! 2. Anti-Crush Recovery: Under calm conditions ($\varepsilon_t \to \varepsilon_0$), recovers full Fractional Kelly.

use serde::{Deserialize, Serialize};

pub const DEFAULT_WASSERSTEIN_EPS0: f64 = 0.035;
pub const MAX_PORTFOLIO_CVAR_CAP: f64 = 0.065; // 6.5% portfolio tail heat cap

/// Inputs for 1-Wasserstein Distributionally Robust Kelly + CVaR allocation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DroKellyInput {
    pub p_cp_eff: f64,
    pub epistemic_ood_score: f64,
    pub p_mondrian: f64,
    pub martingale_mixture_m: f64,
    pub eigen_kappa: f64,
    pub open_book_heat_pct: f64,
    pub floor_kelly_fraction: f64,
    pub max_kelly_fraction: f64,
}

impl Default for DroKellyInput {
    fn default() -> Self {
        Self {
            p_cp_eff: 0.04,
            epistemic_ood_score: 0.10,
            p_mondrian: 0.85,
            martingale_mixture_m: 1.0,
            eigen_kappa: 1.0,
            open_book_heat_pct: 0.015,
            floor_kelly_fraction: 0.005,
            max_kelly_fraction: 0.45,
        }
    }
}

/// Telemetry output of the 1-Wasserstein DRO Kelly-CVaR optimization.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DroKellyAllocationResult {
    pub optimal_dro_fraction: f64,
    pub optimal_kelly_pct: f64,
    pub ambiguity_radius_eps: f64,
    pub marginal_cvar95_r: f64,
    pub cvar_tail_heat_pct: f64,
    pub was_cvar_capped: bool,
    pub nominal_unconstrained_fraction: f64,
}

pub struct DistributionallyRobustKellyAllocator;

impl DistributionallyRobustKellyAllocator {
    /// Computes the unified 1-Wasserstein ambiguity radius $\varepsilon_t$.
    #[inline]
    pub fn compute_ambiguity_radius(
        p_cp_eff: f64,
        ood_score: f64,
        p_mondrian: f64,
        martingale_m: f64,
        eps_base: f64,
    ) -> f64 {
        let cp_term = 1.40 * p_cp_eff.clamp(0.0, 1.0);
        let ood_term = 1.20 * ood_score.clamp(0.0, 2.5);
        let mondrian_term = 0.80 * (1.0 - p_mondrian.clamp(0.01, 1.0));
        let m_clean = martingale_m.max(1.0);
        let martingale_term = 0.50 * m_clean.ln();

        let multiplier = 1.0 + cp_term + ood_term + mondrian_term + martingale_term;
        (eps_base * multiplier).clamp(0.01, 0.45)
    }

    /// Evaluates empirical CVaR at alpha=0.95 over R-multiples.
    pub fn compute_cvar95_from_returns(returns: &[f64]) -> f64 {
        if returns.is_empty() {
            return 1.25;
        }
        let mut sorted_losses: Vec<f64> = returns
            .iter()
            .map(|&r| (-r).max(0.0))
            .collect();
        sorted_losses.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let cutoff_idx = (((sorted_losses.len() - 1) as f64) * 0.95).floor() as usize;
        let tail_slice = &sorted_losses[cutoff_idx..];
        let cvar = tail_slice.iter().sum::<f64>() / (tail_slice.len() as f64).max(1.0);
        cvar.max(0.85)
    }

    /// Solves the 1-Wasserstein Distributionally Robust Kelly optimization problem
    /// via a 42-step deterministic golden section search on the strictly concave domain.
    pub fn solve_dro_allocation(
        returns: &[f64],
        weights: &[f64],
        input: &DroKellyInput,
    ) -> DroKellyAllocationResult {
        let eps = Self::compute_ambiguity_radius(
            input.p_cp_eff,
            input.epistemic_ood_score,
            input.p_mondrian,
            input.martingale_mixture_m,
            DEFAULT_WASSERSTEIN_EPS0,
        );

        let default_returns = [2.20, 1.80, -1.00, -0.90, 2.50, -1.10];
        let default_weights = [0.20, 0.20, 0.20, 0.15, 0.15, 0.10];
        let (r_slice, w_slice) = if returns.is_empty() || weights.is_empty() {
            (&default_returns[..], &default_weights[..])
        } else {
            (returns, weights)
        };

        let mut r_min_abs = 1.0_f64;
        for &r in r_slice {
            if r < 0.0 && (-r) > r_min_abs {
                r_min_abs = -r;
            }
        }
        let r_min_abs = r_min_abs.clamp(0.50, 3.50);

        let f_min = input.floor_kelly_fraction.clamp(0.0025, 0.02);
        // Domain bound so (1 - f * |R_min|) > 0 and (1 + f * R_i) > 0:
        let max_safe_domain = (0.95 / r_min_abs).min(input.max_kelly_fraction.clamp(0.10, 0.50));

        let eigen_lambda = 0.50 * (1.0 - input.eigen_kappa.clamp(0.10, 1.0));

        // Objective function h(f)
        let eval_obj = |f: f64| -> f64 {
            let denom = 1.0 - f * r_min_abs;
            if denom <= 1e-4 {
                return -1e12;
            }
            let mut expected_log = 0.0_f64;
            let mut sum_w = 0.0_f64;
            for (&r, &w) in r_slice.iter().zip(w_slice.iter()) {
                let term = 1.0 + f * r;
                if term <= 1e-4 {
                    return -1e12;
                }
                expected_log += w * term.ln();
                sum_w += w;
            }
            expected_log /= sum_w.max(1e-6);

            let dro_penalty = eps * (f / denom);
            let eigen_penalty = 0.5 * eigen_lambda * f * f;

            expected_log - dro_penalty - eigen_penalty
        };

        // 42-step Golden Section search
        let phi = 0.618033988749895_f64;
        let mut a = f_min;
        let mut b = max_safe_domain;
        let mut c = b - phi * (b - a);
        let mut d = a + phi * (b - a);
        let mut fc = eval_obj(c);
        let mut fd = eval_obj(d);

        for _ in 0..42 {
            if fc > fd {
                b = d;
                d = c;
                fd = fc;
                c = b - phi * (b - a);
                fc = eval_obj(c);
            } else {
                a = c;
                c = d;
                fc = fd;
                d = a + phi * (b - a);
                fd = eval_obj(d);
            }
        }

        let f_unconstrained = 0.5 * (a + b);

        // CVaR tail-heat constraint:
        // Marginal portfolio tail heat of position fraction f with typical stop risk factor (0.15):
        let cvar95 = Self::compute_cvar95_from_returns(r_slice);
        let tail_risk_scale = 0.15_f64;
        let marginal_tail_heat_per_f = (cvar95 * tail_risk_scale).max(0.05);
        let remaining_heat_cap =
            (MAX_PORTFOLIO_CVAR_CAP - input.open_book_heat_pct.max(0.0)).max(0.002);
        let f_cvar_max = remaining_heat_cap / marginal_tail_heat_per_f;

        let was_capped = f_unconstrained > f_cvar_max;
        let f_optimal = f_unconstrained.min(f_cvar_max).clamp(f_min, max_safe_domain);
        let kelly_pct = ((f_optimal * 100.0) * 100.0).round() / 100.0;
        let tail_heat = input.open_book_heat_pct + f_optimal * marginal_tail_heat_per_f;

        DroKellyAllocationResult {
            optimal_dro_fraction: f_optimal,
            optimal_kelly_pct: kelly_pct,
            ambiguity_radius_eps: eps,
            marginal_cvar95_r: cvar95,
            cvar_tail_heat_pct: tail_heat,
            was_cvar_capped: was_capped,
            nominal_unconstrained_fraction: f_unconstrained,
        }
    }
}
