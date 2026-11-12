//! Phase 36 Pillar 2: Adams-MacKay Bayesian Online Changepoint Detection (BOCPD) Engine
//!
//! Tracks the causal run-length posterior $P(r_t = \ell \mid x_{1:t})$ over $\ell \in \{0, 1, \dots, 64\}$
//! using constant hazard prior $H(\ell) = \lambda_{\text{cp}}^{-1}$ ($\lambda_{\text{cp}} = 35.0$) and online
//! conjugate Normal-Gamma Student-$t$ predictive densities $\pi_\ell(x_t)$ over causal cross-sectional
//! regime surprise $x_t$ (standardized return shock, Stream 2 `[12..20]` Dealer GEX/Charm sign-flip
//! velocity, and Stream 3 `[20..26]` toxicity jump).
//!
//! Drives `NigPosteriorStats::observe_settled_trade_adaptive` with dynamic forgetting factor:
//! $$\gamma_t = \operatorname{clamp}\big(0.992 - 0.22 \cdot p_{\text{cp}, t},\; 0.76,\; 0.992\big)$$
//! where $p_{\text{cp}, t} = \sum_{\ell=0}^{3} P(r_t = \ell \mid x_{1:t}) \in [0, 1]$.

use serde::{Deserialize, Serialize};

pub const BOCPD_MAX_RUN_LENGTH: usize = 64;
pub const BOCPD_HAZARD_LAMBDA_CP: f64 = 35.0;

/// Computes the causal BOCPD-driven adaptive forgetting factor $\gamma_t \in [0.76, 0.992]$.
#[inline]
pub fn compute_adaptive_forgetting_gamma(p_cp: f64) -> f64 {
    let p = if p_cp.is_finite() {
        p_cp.clamp(0.0, 1.0)
    } else {
        0.0
    };
    (0.992 - 0.22 * p).clamp(0.76, 0.992)
}

/// High-precision Lanczos approximation for $\ln \Gamma(z)$ ($z > 0$).
#[inline]
fn ln_gamma_lanczos(z: f64) -> f64 {
    const COEFFS: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    let z_pos = z.max(1e-6);
    if z_pos < 0.5 {
        let pi = std::f64::consts::PI;
        return pi.ln() - (pi * z_pos).sin().abs().max(1e-15).ln() - ln_gamma_lanczos(1.0 - z_pos);
    }
    let zm1 = z_pos - 1.0;
    let mut x = COEFFS[0];
    for (i, &c) in COEFFS.iter().enumerate().skip(1) {
        x += c / (zm1 + (i as f64));
    }
    let t = zm1 + 7.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (zm1 + 0.5) * t.ln() - t + x.ln()
}

/// Exact Student-$t$ predictive PDF $\pi_\ell(x)$ under Normal-Gamma sufficient statistics $(\mu, \kappa, \alpha, \beta)$.
#[inline]
fn normal_gamma_student_t_pdf(x: f64, mu: f64, kappa: f64, alpha: f64, beta: f64) -> f64 {
    let nu = (2.0 * alpha).max(1.0);
    let scale_sq = ((beta * (kappa + 1.0)) / (alpha * kappa).max(1e-9)).max(1e-8);
    let scale = scale_sq.sqrt();
    let z = (x - mu) / scale;
    let ln_pdf = ln_gamma_lanczos(0.5 * (nu + 1.0))
        - ln_gamma_lanczos(0.5 * nu)
        - 0.5 * (nu * std::f64::consts::PI).ln()
        - scale.ln()
        - 0.5 * (nu + 1.0) * (1.0 + (z * z) / nu).ln();
    ln_pdf.exp().clamp(1e-30, 1e12)
}

/// Snapshot of BOCPD structural changepoint telemetry at step $t$.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BocpdStepTelemetry {
    pub step_count: usize,
    pub latest_regime_surprise: f64,
    /// Structural Changepoint Probability $p_{\text{cp}, t} = \sum_{\ell=0}^{3} P(r_t = \ell \mid x_{1:t}) \in [0, 1]$.
    pub bocpd_changepoint_prob: f64,
    /// Expected run length $\mathbb{E}[r_t] = \sum_{\ell=0}^{64} \ell \cdot P(r_t = \ell \mid x_{1:t})$.
    pub bocpd_expected_run_length: f64,
    /// Dynamic forgetting factor $\gamma_t = \operatorname{clamp}(0.992 - 0.22 \cdot p_{\text{cp}, t},\; 0.76,\; 0.992)$.
    pub adaptive_forgetting_gamma: f64,
}

impl Default for BocpdStepTelemetry {
    fn default() -> Self {
        let init_p_cp = 0.06;
        Self {
            step_count: 0,
            latest_regime_surprise: 0.0,
            bocpd_changepoint_prob: init_p_cp,
            bocpd_expected_run_length: 12.0,
            adaptive_forgetting_gamma: compute_adaptive_forgetting_gamma(init_p_cp),
        }
    }
}

/// Online Adams-MacKay (2007) Bayesian Online Changepoint Detection engine over $\ell \in \{0,\dots,64\}$.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdamsMacKayBocpdEngine {
    hazard_rate: f64,
    prior_mu0: f64,
    prior_kappa0: f64,
    prior_alpha0: f64,
    prior_beta0: f64,
    run_length_probs: Vec<f64>,
    mu: Vec<f64>,
    kappa: Vec<f64>,
    alpha: Vec<f64>,
    beta: Vec<f64>,
    prev_gex_signature: Option<f64>,
    prev_toxicity_signature: Option<f64>,
    #[serde(default)]
    last_timestamp_ns: Option<i64>,
    step_count: usize,
    last_telemetry: BocpdStepTelemetry,
}

impl Default for AdamsMacKayBocpdEngine {
    fn default() -> Self {
        Self::new(BOCPD_HAZARD_LAMBDA_CP)
    }
}

impl AdamsMacKayBocpdEngine {
    /// Initializes a causal BOCPD run-length filter with constant hazard $H = 1 / \lambda_{\text{cp}}$
    /// and a broad pre-break Normal-Gamma prior ($\beta_0 = 2.25$) so structural shocks concentrate
    /// posterior mass onto short run lengths $\ell \in \{0..3\}$ while calm regimes grow $\mathbb{E}[r_t]$.
    pub fn new(lambda_cp: f64) -> Self {
        let len = BOCPD_MAX_RUN_LENGTH + 1;
        let hazard = (1.0 / lambda_cp.max(5.0)).clamp(0.005, 0.25);
        let mu0 = 0.0;
        let kappa0 = 0.50;
        let alpha0 = 1.50;
        let beta0 = 2.25;

        let mut probs = vec![0.0_f64; len];
        // Start with stationary geometric prior seeded past immediate step-0 transient (ell = 6)
        probs[6] = 1.0;

        Self {
            hazard_rate: hazard,
            prior_mu0: mu0,
            prior_kappa0: kappa0,
            prior_alpha0: alpha0,
            prior_beta0: beta0,
            run_length_probs: probs,
            mu: vec![mu0; len],
            kappa: vec![kappa0; len],
            alpha: vec![alpha0; len],
            beta: vec![beta0; len],
            prev_gex_signature: None,
            prev_toxicity_signature: None,
            last_timestamp_ns: None,
            step_count: 0,
            last_telemetry: BocpdStepTelemetry {
                step_count: 0,
                latest_regime_surprise: 0.0,
                bocpd_changepoint_prob: hazard,
                bocpd_expected_run_length: 6.0,
                adaptive_forgetting_gamma: compute_adaptive_forgetting_gamma(hazard),
            },
        }
    }

    pub fn telemetry(&self) -> BocpdStepTelemetry {
        self.last_telemetry
    }

    pub fn last_timestamp_ns(&self) -> Option<i64> {
        self.last_timestamp_ns
    }

    /// Timestamp-deduplicated BOCPD update: if `timestamp_ns` was already observed (`<= last_timestamp_ns`),
    /// returns the current telemetry without double-stepping run lengths.
    pub fn observe_surprise_at_timestamp(
        &mut self,
        timestamp_ns: i64,
        x_t: f64,
    ) -> BocpdStepTelemetry {
        if self
            .last_timestamp_ns
            .is_some_and(|prev_ts| timestamp_ns <= prev_ts)
        {
            return self.last_telemetry;
        }
        self.last_timestamp_ns = Some(timestamp_ns);
        self.observe_surprise(x_t)
    }

    pub fn run_length_posterior(&self) -> &[f64] {
        &self.run_length_probs
    }

    /// Extracts causal cross-sectional regime surprise $x_t$ from a 32-feature entry/bar vector,
    /// combining standardized return impulse (`[1, 8]`), Stream 2 `[12..20]` Dealer GEX/Charm
    /// sign-flip velocity, and Stream 3 `[20..26]` orderflow toxicity jump.
    pub fn extract_causal_feature_surprise(&mut self, features_32: &[f64; 32]) -> f64 {
        let ret_shock = 0.55 * features_32[1].clamp(-6.0, 6.0) + 0.45 * features_32[8].clamp(-4.0, 4.0);
        let gex_sig: f64 = features_32[12..20].iter().sum::<f64>() / 8.0;
        let tox_sig: f64 = features_32[20..26].iter().sum::<f64>() / 6.0;

        let gex_flip_vel = if let Some(prev_g) = self.prev_gex_signature {
            let sign_flip_boost = if (gex_sig >= 0.0) != (prev_g >= 0.0) {
                1.65
            } else {
                1.0
            };
            (gex_sig - prev_g) * sign_flip_boost * 2.5
        } else {
            0.0
        };
        let tox_jump = if let Some(prev_t) = self.prev_toxicity_signature {
            (tox_sig - prev_t) * 2.0
        } else {
            0.0
        };

        self.prev_gex_signature = Some(gex_sig);
        self.prev_toxicity_signature = Some(tox_sig);

        (0.40 * ret_shock + 0.35 * gex_flip_vel + 0.25 * tox_jump).clamp(-12.0, 12.0)
    }

    /// Advances the BOCPD filter by one causal feature vector `features_32`.
    pub fn observe_features_32(&mut self, features_32: &[f64; 32]) -> BocpdStepTelemetry {
        let x_t = self.extract_causal_feature_surprise(features_32);
        self.observe_surprise(x_t)
    }

    pub fn run_length_probs(&self) -> &[f64] {
        &self.run_length_probs
    }

    pub fn kappa_at(&self, ell: usize) -> f64 {
        self.kappa.get(ell).copied().unwrap_or(self.prior_kappa0)
    }

    #[inline]
    fn update_normal_gamma_branch(
        mu_prev: f64,
        kappa_prev: f64,
        alpha_prev: f64,
        beta_prev: f64,
        x: f64,
    ) -> (f64, f64, f64, f64) {
        let k_new = kappa_prev + 1.0;
        let diff = x - mu_prev;
        let mu_new = (kappa_prev * mu_prev + x) / k_new;
        let alpha_new = alpha_prev + 0.5;
        let beta_new = beta_prev + (kappa_prev * diff * diff) / (2.0 * k_new);
        (mu_new, k_new, alpha_new, beta_new)
    }

    /// Exact Adams-MacKay (2007) run-length posterior update on scalar regime surprise $x_t$
    /// with precision-weighted Normal-Gamma moment matching at the truncation absorber $\ell = 64$.
    #[allow(clippy::needless_range_loop)]
    pub fn observe_surprise(&mut self, x_t: f64) -> BocpdStepTelemetry {
        let x = if x_t.is_finite() {
            x_t.clamp(-15.0, 15.0)
        } else {
            0.0
        };
        let len = BOCPD_MAX_RUN_LENGTH + 1;
        let h = self.hazard_rate;

        let mut pred_pdf = [0.0_f64; BOCPD_MAX_RUN_LENGTH + 1];
        for ell in 0..len {
            pred_pdf[ell] = normal_gamma_student_t_pdf(
                x,
                self.mu[ell],
                self.kappa[ell],
                self.alpha[ell],
                self.beta[ell],
            );
        }

        let mut next_probs = vec![0.0_f64; len];
        let mut cp_mass = 0.0_f64;
        let mut w_63_to_64 = 0.0_f64;
        let mut w_64_to_64 = 0.0_f64;

        for ell in 0..len {
            let weighted = self.run_length_probs[ell] * pred_pdf[ell];
            cp_mass += weighted * h;
            let growth = weighted * (1.0 - h);
            if ell + 1 < BOCPD_MAX_RUN_LENGTH {
                next_probs[ell + 1] += growth;
            } else if ell + 1 == BOCPD_MAX_RUN_LENGTH {
                w_63_to_64 = growth;
                next_probs[BOCPD_MAX_RUN_LENGTH] += growth;
            } else {
                w_64_to_64 = growth;
                next_probs[BOCPD_MAX_RUN_LENGTH] += growth;
            }
        }
        next_probs[0] = cp_mass;

        let evidence: f64 = next_probs.iter().sum::<f64>().max(1e-30);
        for p in &mut next_probs {
            *p /= evidence;
        }

        // Update Normal-Gamma sufficient statistics shifted along interior run-lengths ell = 0..63 -> 1..=63
        let mut next_mu = vec![self.prior_mu0; len];
        let mut next_kappa = vec![self.prior_kappa0; len];
        let mut next_alpha = vec![self.prior_alpha0; len];
        let mut next_beta = vec![self.prior_beta0; len];

        for ell in 0..(BOCPD_MAX_RUN_LENGTH - 1) {
            let (m_n, k_n, a_n, b_n) = Self::update_normal_gamma_branch(
                self.mu[ell],
                self.kappa[ell],
                self.alpha[ell],
                self.beta[ell],
                x,
            );
            next_mu[ell + 1] = m_n;
            next_kappa[ell + 1] = k_n;
            next_alpha[ell + 1] = a_n;
            next_beta[ell + 1] = b_n;
        }

        // Moment-match the incoming ell=63->64 and stationary absorber ell=64->64 Normal-Gamma branches
        let idx_63 = BOCPD_MAX_RUN_LENGTH - 1;
        let idx_64 = BOCPD_MAX_RUN_LENGTH;
        let (mu_a, k_a, a_a, b_a) = Self::update_normal_gamma_branch(
            self.mu[idx_63],
            self.kappa[idx_63],
            self.alpha[idx_63],
            self.beta[idx_63],
            x,
        );
        let (mu_b, k_b, a_b, b_b) = Self::update_normal_gamma_branch(
            self.mu[idx_64],
            self.kappa[idx_64],
            self.alpha[idx_64],
            self.beta[idx_64],
            x,
        );
        let w_absorber_sum = w_63_to_64 + w_64_to_64;
        if w_absorber_sum > 1e-30 {
            let omega_63 = w_63_to_64 / w_absorber_sum;
            let omega_64 = w_64_to_64 / w_absorber_sum;
            let k_mix = omega_63 * k_a + omega_64 * k_b;
            let mu_mix = (omega_63 * k_a * mu_a + omega_64 * k_b * mu_b) / k_mix.max(1e-12);
            let a_mix = omega_63 * a_a + omega_64 * a_b;
            let dm_a = mu_a - mu_mix;
            let dm_b = mu_b - mu_mix;
            let b_mix = omega_63 * b_a
                + omega_64 * b_b
                + 0.5 * (omega_63 * k_a * dm_a * dm_a + omega_64 * k_b * dm_b * dm_b);
            next_mu[idx_64] = mu_mix;
            next_kappa[idx_64] = k_mix;
            next_alpha[idx_64] = a_mix;
            next_beta[idx_64] = b_mix;
        } else {
            next_mu[idx_64] = mu_a;
            next_kappa[idx_64] = k_a;
            next_alpha[idx_64] = a_a;
            next_beta[idx_64] = b_a;
        }

        self.run_length_probs = next_probs;
        self.mu = next_mu;
        self.kappa = next_kappa;
        self.alpha = next_alpha;
        self.beta = next_beta;
        self.step_count += 1;

        let p_cp: f64 = self.run_length_probs[0..=3]
            .iter()
            .sum::<f64>()
            .clamp(0.0, 1.0);
        let expected_rl: f64 = self
            .run_length_probs
            .iter()
            .enumerate()
            .map(|(ell, &p)| (ell as f64) * p)
            .sum();
        let gamma_t = compute_adaptive_forgetting_gamma(p_cp);

        let tel = BocpdStepTelemetry {
            step_count: self.step_count,
            latest_regime_surprise: x,
            bocpd_changepoint_prob: p_cp,
            bocpd_expected_run_length: expected_rl,
            adaptive_forgetting_gamma: gamma_t,
        };
        self.last_telemetry = tel;
        tel
    }
}

