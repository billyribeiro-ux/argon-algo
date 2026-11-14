# ZERO-CHEAT END-TO-END DIFFERENTIABLE BPTT + CLOSED-LOOP LOSS-AUTOPSY RL PROOF REPORT

- **Generated At (UTC):** `2026-11-14T02:11:49.205774+00:00`
- **Dataset 1 (`1m` Intraday + Causal `5m` Session Pre-Training):** `21` total `1m` trading days (`11` OOS days), `167411` real `1m` OHLCV bars loaded
- **Dataset 2 (`5m` & `1h` 60-Ticker Universe):** `59` real US symbols across `65` total trading days (`24` Warmup Calibration -> `41` Strictly Out-of-Sample)
- **Zero-Rule Neural Architecture (`src/pure_rl_brain.rs`):** Stage A Multi-Horizon Physics World Model (`1, 3, 8, 24`-bar Pinball Quantile Loss + `ln(Range/RT_Cost)`) + Stage B `K=5` Orthogonal Specialist Policy Ensemble (`D=64`, `[h_ln, wm_fused, self_state, pos_interact]` 92-d skip-head) with Localized Boundary Surrogate Deadzone (`sech^2((|z| - tau) / temp)`), Unconstrained Differentiable Log-Space Cost Sensitivity (`k_theta = exp(log_k_theta)` via analytical `dL_net / d log_k_theta`), Turnover-Amortized Opportunity Critic (`Q_hold(x_t, s_t) - Q_flat(x_t, s_t)`), Purged `T_holdout` Cross-Sectional LCB Promotion Gate, and `M=24` `rayon` PBT + Multiplicative-Weights Routing.
- **Anti-Cheat Execution Physics:** Zero `> t` lookahead, next-bar `t+1` open fills with `5/12/35 bps` adverse equity slippage, `09:35–15:15 ET` flatten `15:50 ET` (`SCALP_0DTE`), `09:35–15:30 ET` flatten `15:55 ET` (`DAY`), skew-adjusted BSM + `^VIX9D`/`^VIX` IV + `$0.03–$0.08`/sh/leg spread + `$0.65`/contract/leg RT commission.
- **BPTT & Closed-Loop Loss-Autopsy Activity:** `249577` Causal Warmup Episode Bars | `293212` OOS Causal Episode Bars | `105` Online EOD RL & Routing Cycles (`RUN_B_ONLINE_RL`) | `260650` Autonomous `FLAT` Abstentions.

## Causal Walk-Forward Inner PBT Calibration (`t <= T_oos_start`)

| Brain / Horizon | Population | Session Episodes | PBT Gens | Best Inner-Val Trades | Best Inner-Val Net PF | Best Inner-Val Net PnL ($) |
|---|---:|---:|---:|---:|---:|---:|
| SCALP_0DTE (1m) | 24 | 150 | 3 | 6 | **0.66** | $-47.16 |
| SCALP_0DTE (5m) | 24 | 192 | 3 | 1 | **12.00** | $+12.11 |
| BRAIN-01 (Penny Day) | 24 | 379 | 3 | 6 | **0.11** | $-322.96 |
| BRAIN-02 (Large-Cap Day) | 24 | 818 | 3 | 5 | **12.00** | $+270.57 |
| BRAIN-03 (Short Swing) | 24 | 411 | 3 | 8 | **4.35** | $+905.87 |
| BRAIN-04 (Core Swing) | 24 | 411 | 3 | 10 | **3.71** | $+2701.06 |

## Executive Scorecard: `RUN_A_FROZEN_WARMUP_ML` vs `RUN_B_ONLINE_RL` (Gross vs Net Post-Cost)

| Bucket | Run Mode | Trades | Win% | Gross PF | Net Post-Cost PF | Net Exp (R) | Net Exp ($) | Gross PnL ($) | Cost Drag ($) | Net PnL ($) | Max DD ($ / %) | Backstop% | Neural-Exit% | EOD Flatten% |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SCALP_0DTE (1m) | `RUN_A_FROZEN_WARMUP_ML` | 18 | 55.6% | 2.67 | **1.90** | +0.039R | $+22.48 | $+613.90 | $209.30 | **$+404.60** | $341.32 (0.34%) | 0.0% | 100.0% | 0.0% |
| SCALP_0DTE (1m) | `RUN_B_ONLINE_RL` | 15 | 46.7% | 1.25 | **0.94** | -0.022R | $-2.62 | $+153.30 | $192.53 | **$-39.23** | $429.77 (0.43%) | 0.0% | 100.0% | 0.0% |
| SCALP_0DTE (5m) | `RUN_A_FROZEN_WARMUP_ML` | 27 | 29.6% | 0.65 | **0.50** | -0.061R | $-21.50 | $-350.28 | $230.11 | **$-580.39** | $580.39 (0.58%) | 0.0% | 100.0% | 0.0% |
| SCALP_0DTE (5m) | `RUN_B_ONLINE_RL` | 16 | 25.0% | 0.39 | **0.26** | -0.093R | $-31.17 | $-346.85 | $151.93 | **$-498.78** | $560.68 (0.56%) | 0.0% | 100.0% | 0.0% |
| BRAIN-01 (Penny Day) | `RUN_A_FROZEN_WARMUP_ML` | 1 | 0.0% | 0.00 | **0.00** | -0.104R | $-37.67 | $+0.00 | $37.67 | **$-37.67** | $37.67 (0.04%) | 0.0% | 0.0% | 100.0% |
| BRAIN-01 (Penny Day) | `RUN_B_ONLINE_RL` | 1 | 0.0% | 0.00 | **0.00** | -0.104R | $-37.77 | $+0.00 | $37.77 | **$-37.77** | $37.77 (0.04%) | 0.0% | 0.0% | 100.0% |
| BRAIN-02 (Large-Cap Day) | `RUN_A_FROZEN_WARMUP_ML` | 17 | 35.3% | 0.37 | **0.23** | -0.112R | $-37.99 | $-461.40 | $184.45 | **$-645.85** | $685.38 (0.69%) | 0.0% | 47.1% | 52.9% |
| BRAIN-02 (Large-Cap Day) | `RUN_B_ONLINE_RL` | 22 | 50.0% | 0.88 | **0.72** | -0.048R | $-18.79 | $-162.84 | $250.63 | **$-413.47** | $652.09 (0.65%) | 0.0% | 100.0% | 0.0% |
| BRAIN-03 (Short Swing) | `RUN_A_FROZEN_WARMUP_ML` | 13 | 46.2% | 0.94 | **0.88** | +0.049R | $-16.15 | $-98.27 | $111.65 | **$-209.92** | $1312.57 (1.31%) | 15.4% | 69.2% | 0.0% |
| BRAIN-03 (Short Swing) | `RUN_B_ONLINE_RL` | 13 | 46.2% | 0.94 | **0.88** | +0.049R | $-16.15 | $-98.27 | $111.65 | **$-209.92** | $1312.57 (1.31%) | 15.4% | 69.2% | 0.0% |
| BRAIN-04 (Core Swing) | `RUN_A_FROZEN_WARMUP_ML` | 16 | 56.2% | 1.79 | **1.59** | +0.074R | $+49.06 | $+993.22 | $208.19 | **$+785.03** | $1154.22 (1.15%) | 6.2% | 75.0% | 0.0% |
| BRAIN-04 (Core Swing) | `RUN_B_ONLINE_RL` | 14 | 64.3% | 2.92 | **2.66** | +0.191R | $+151.24 | $+2315.23 | $197.83 | **$+2117.40** | $1154.22 (1.15%) | 7.1% | 71.4% | 0.0% |
| ALL_BUCKETS_COMBINED | `RUN_A_FROZEN_WARMUP_ML` | 92 | 42.4% | 1.14 | **0.95** | -0.012R | $-3.09 | $+697.17 | $981.37 | **$-284.20** | $2631.19 (2.62%) | 3.3% | 80.4% | 10.9% |
| ALL_BUCKETS_COMBINED | `RUN_B_ONLINE_RL` | 81 | 45.7% | 1.35 | **1.16** | +0.004R | $+11.34 | $+1860.57 | $942.34 | **$+918.23** | $2576.53 (2.57%) | 3.7% | 88.9% | 1.2% |

## Horizon Tier Rollups (`RUN_B_ONLINE_RL`)

| Tier | Run Mode | Trades | Win% | Gross PF | Net Post-Cost PF | Net Exp (R) | Net Exp ($) | Gross PnL ($) | Cost Drag ($) | Net PnL ($) | Max DD ($ / %) | Backstop% | Neural-Exit% | EOD Flatten% |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TIER: SCALP_0DTE | `RUN_B_ONLINE_RL` | 31 | 35.5% | 0.84 | **0.61** | -0.059R | $-17.36 | $-193.55 | $344.46 | **$-538.01** | $641.99 (0.64%) | 0.0% | 100.0% | 0.0% |
| TIER: DAY (BRAIN-01 + BRAIN-02) | `RUN_B_ONLINE_RL` | 23 | 47.8% | 0.88 | **0.70** | -0.050R | $-19.62 | $-162.84 | $288.40 | **$-451.24** | $689.86 (0.69%) | 0.0% | 95.7% | 4.3% |
| TIER: SWING (BRAIN-03 + BRAIN-04) | `RUN_B_ONLINE_RL` | 27 | 55.6% | 1.78 | **1.65** | +0.123R | $+70.65 | $+2216.96 | $309.48 | **$+1907.48** | $1916.89 (1.91%) | 11.1% | 70.4% | 0.0% |

## Factored Vehicle Performance Split (`V_SHARES` vs `V_OPTION`)

| Run Mode | Vehicle | Trades | Win% | Gross PF | Net Post-Cost PF | Net Exp (R) | Avg Cost/Trade ($) | Total Gross PnL ($) | Total Cost Drag ($) | Total Net PnL ($) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `RUN_A_FROZEN_WARMUP_ML` | **V_SHARES** | 92 | 42.4% | 1.14 | **0.95** | -0.012R | $10.67 | $+697.17 | $981.37 | **$-284.20** |
| `RUN_A_FROZEN_WARMUP_ML` | **V_OPTION** | 0 | 0.0% | 0.00 | **0.00** | +0.000R | $0.00 | $+0.00 | $0.00 | **$+0.00** |
| `RUN_B_ONLINE_RL` | **V_SHARES** | 81 | 45.7% | 1.35 | **1.16** | +0.004R | $11.63 | $+1860.57 | $942.34 | **$+918.23** |
| `RUN_B_ONLINE_RL` | **V_OPTION** | 0 | 0.0% | 0.00 | **0.00** | +0.000R | $0.00 | $+0.00 | $0.00 | **$+0.00** |

## A/B Online Learning Delta (`RUN_B_ONLINE_RL` minus `RUN_A_FROZEN_WARMUP_ML`)

| Bucket | Δ Trades | Δ Win% | Δ Net PF | Δ Net Exp (R) | Δ Net PnL ($) | Δ Max DD ($) |
|---|---:|---:|---:|---:|---:|---:|
| SCALP_0DTE (1m) | -3 | -8.9% | -0.96 | -0.060R | **$-443.83** | $+88.46 |
| SCALP_0DTE (5m) | -11 | -4.6% | -0.24 | -0.032R | **$+81.61** | $-19.71 |
| BRAIN-01 (Penny Day) | +0 | +0.0% | +0.00 | +0.000R | **$-0.10** | $+0.10 |
| BRAIN-02 (Large-Cap Day) | +5 | +14.7% | +0.49 | +0.064R | **$+232.38** | $-33.29 |
| BRAIN-03 (Short Swing) | +0 | +0.0% | +0.00 | +0.000R | **$+0.00** | $+0.00 |
| BRAIN-04 (Core Swing) | -2 | +8.0% | +1.07 | +0.117R | **$+1332.37** | $+0.00 |
| **DATASET_1_1M_TOTAL** | — | — | — | — | **$-443.83** | — |
| **DATASET_2_5M_1H_TOTAL** | — | — | — | — | **$+1646.26** | — |
| **ALL_BUCKETS_COMBINED** | **-11** | **+3.3%** | **+0.21** | **+0.017R** | **$+1202.43** | **$-54.66** |

## Loss-Autopsy -> Re-Test -> Generation Promotion Ledger

| Gen ID | Phase | Brain | Red Trades Audited | Entry vs Exit Fault Split | Saliency (`|dL/dx|` / `|dL/dh|`) | Re-Test Net PF (Before -> After) | Red Loss Reduced ($) | Winners Preserved | Outcome |
|---:|---|---|---:|---|---|---|---:|---|---|
| 1 | `WARMUP_CAL` | SCALP_0DTE (1m) | 3 | `2 Entry / 1 Exit` | `0.2585 / 2.0790` | `0.66 -> 0.04` | $+35.57 | `3->1 wins (+0.27R -> +0.01R)` | `ROLLED_BACK` |
| 1 | `WARMUP_CAL` | SCALP_0DTE (5m) | 0 | `0 Entry / 0 Exit` | `0.1146 / 2.3268` | `12.00 -> 12.00` | $+0.00 | `1->1 wins (+0.04R -> +0.04R)` | `ROLLED_BACK` |
| 1 | `WARMUP_CAL` | BRAIN-01 (Penny Day) | 0 | `0 Entry / 0 Exit` | `0.0658 / 1.0953` | `0.00 -> 12.00` | $+0.00 | `0->1 wins (+0.00R -> +0.06R)` | `ROLLED_BACK` |
| 2 | `WARMUP_CAL` | BRAIN-02 (Large-Cap Day) | 0 | `0 Entry / 0 Exit` | `0.2106 / 4.2184` | `12.00 -> 12.00` | $+0.00 | `2->5 wins (Net +0.22R -> +0.79R)` | **PROMOTED** |
| 2 | `OOS_D1` | SCALP_0DTE (1m) | 3 | `2 Entry / 1 Exit` | `0.2585 / 2.0790` | `5.44 -> 6.11` | $+9.54 | `8->8 wins (Net +1.43R -> +1.45R)` | **PROMOTED** |
| 2 | `OOS_D3` | SCALP_0DTE (1m) | 3 | `2 Entry / 1 Exit` | `0.2565 / 2.0575` | `3.63 -> 5.68` | $+67.93 | `6->2 wins (+1.29R -> +0.95R)` | `ROLLED_BACK` |
| 3 | `OOS_D4` | SCALP_0DTE (1m) | 6 | `3 Entry / 3 Exit` | `0.2212 / 1.8793` | `1.90 -> 3.75` | $+144.90 | `6->4 wins (Net +0.61R -> +0.82R)` | **PROMOTED** |
| 3 | `OOS_D5` | SCALP_0DTE (1m) | 3 | `2 Entry / 1 Exit` | `0.2707 / 2.2098` | `0.94 -> 3.83` | $+325.09 | `4->3 wins (+1.10R -> +1.00R)` | `ROLLED_BACK` |
| 3 | `OOS_D7` | SCALP_0DTE (1m) | 4 | `3 Entry / 1 Exit` | `0.2495 / 2.0726` | `1.22 -> 1.13` | $-421.92 | `6->16 wins (+1.22R -> +2.27R)` | `ROLLED_BACK` |
| 4 | `OOS_D8` | SCALP_0DTE (1m) | 5 | `4 Entry / 1 Exit` | `0.3027 / 2.7701` | `1.03 -> 1.35` | $+255.23 | `7->9 wins (Net -0.14R -> +0.29R)` | **PROMOTED** |
| 5 | `OOS_D10` | SCALP_0DTE (1m) | 7 | `3 Entry / 4 Exit` | `0.2296 / 2.6104` | `1.11 -> 6.42` | $+352.36 | `9->3 wins (Net +0.07R -> +0.69R)` | **PROMOTED** |
| 2 | `OOS_D1` | SCALP_0DTE (5m) | 2 | `2 Entry / 0 Exit` | `0.0184 / 0.1050` | `5.37 -> 12.00` | $+28.32 | `2->5 wins (Net +0.32R -> +0.46R)` | **PROMOTED** |
| 1 | `OOS_D1` | BRAIN-01 (Penny Day) | 0 | `0 Entry / 0 Exit` | `0.0380 / 0.7106` | `0.00 -> 12.00` | $+0.00 | `0->1 wins (+0.00R -> +0.07R)` | `ROLLED_BACK` |
| 2 | `OOS_D1` | BRAIN-02 (Large-Cap Day) | 2 | `2 Entry / 0 Exit` | `0.0211 / 0.1673` | `12.00 -> 5.08` | $-86.42 | `5->8 wins (+0.79R -> +1.15R)` | `ROLLED_BACK` |
| 3 | `OOS_D3` | SCALP_0DTE (5m) | 3 | `2 Entry / 1 Exit` | `0.0955 / 0.9636` | `3.20 -> 6.57` | $+46.20 | `6->6 wins (Net +0.43R -> +0.56R)` | **PROMOTED** |
| 1 | `OOS_D3` | BRAIN-01 (Penny Day) | 0 | `0 Entry / 0 Exit` | `0.0468 / 0.7699` | `0.00 -> 12.00` | $+0.00 | `0->1 wins (+0.00R -> +0.06R)` | `ROLLED_BACK` |
| 2 | `OOS_D3` | BRAIN-02 (Large-Cap Day) | 2 | `2 Entry / 0 Exit` | `0.0211 / 0.1673` | `12.00 -> 5.87` | $-37.39 | `5->4 wins (+0.79R -> +0.62R)` | `ROLLED_BACK` |
| 1 | `OOS_D5` | BRAIN-01 (Penny Day) | 0 | `0 Entry / 0 Exit` | `0.0314 / 0.6191` | `0.00 -> 0.00` | $+0.00 | `0->0 wins (+0.00R -> +0.00R)` | `ROLLED_BACK` |
| 2 | `OOS_D5` | BRAIN-02 (Large-Cap Day) | 2 | `2 Entry / 0 Exit` | `0.0211 / 0.1673` | `12.00 -> 0.95` | $-153.73 | `5->3 wins (+0.79R -> +0.38R)` | `ROLLED_BACK` |
| 3 | `OOS_D6` | SCALP_0DTE (5m) | 3 | `2 Entry / 1 Exit` | `0.1071 / 1.0864` | `5.03 -> 12.00` | $+54.40 | `6->1 wins (+0.71R -> +0.21R)` | `ROLLED_BACK` |
| 1 | `OOS_D7` | BRAIN-01 (Penny Day) | 0 | `0 Entry / 0 Exit` | `0.0518 / 0.6626` | `0.00 -> 0.00` | $+0.00 | `0->0 wins (+0.00R -> +0.00R)` | `ROLLED_BACK` |
| 2 | `OOS_D7` | BRAIN-02 (Large-Cap Day) | 2 | `2 Entry / 0 Exit` | `0.0211 / 0.1673` | `12.00 -> 11.84` | $-30.21 | `5->7 wins (+0.79R -> +1.02R)` | `ROLLED_BACK` |
| 1 | `OOS_D9` | BRAIN-01 (Penny Day) | 0 | `0 Entry / 0 Exit` | `0.0779 / 0.8731` | `0.00 -> 0.00` | $+0.00 | `0->0 wins (+0.00R -> +0.00R)` | `ROLLED_BACK` |
| 2 | `OOS_D9` | BRAIN-02 (Large-Cap Day) | 2 | `2 Entry / 0 Exit` | `0.0211 / 0.1673` | `12.00 -> 12.00` | $+0.00 | `5->1 wins (+0.79R -> +0.34R)` | `ROLLED_BACK` |
| 3 | `OOS_D10` | SCALP_0DTE (5m) | 3 | `2 Entry / 1 Exit` | `0.1071 / 1.0864` | `3.38 -> 1.17` | $-69.23 | `5->3 wins (+0.45R -> +0.31R)` | `ROLLED_BACK` |
| 1 | `OOS_D11` | BRAIN-01 (Penny Day) | 0 | `0 Entry / 0 Exit` | `0.1000 / 1.1434` | `0.00 -> 0.00` | $+0.00 | `0->0 wins (+0.00R -> +0.00R)` | `ROLLED_BACK` |
| 2 | `OOS_D11` | BRAIN-02 (Large-Cap Day) | 2 | `2 Entry / 0 Exit` | `0.0211 / 0.1673` | `12.00 -> 1.90` | $-335.94 | `2->15 wins (+0.60R -> +1.30R)` | `ROLLED_BACK` |
| 1 | `OOS_D13` | BRAIN-01 (Penny Day) | 0 | `0 Entry / 0 Exit` | `0.1645 / 1.8870` | `0.00 -> 0.00` | $-45.31 | `0->0 wins (+0.00R -> +0.00R)` | `ROLLED_BACK` |
| 2 | `OOS_D13` | BRAIN-02 (Large-Cap Day) | 2 | `2 Entry / 0 Exit` | `0.0211 / 0.1673` | `12.00 -> 12.00` | $+0.00 | `2->3 wins (+0.60R -> +0.44R)` | `ROLLED_BACK` |
| 3 | `OOS_D14` | SCALP_0DTE (5m) | 3 | `2 Entry / 1 Exit` | `0.1071 / 1.0864` | `2.64 -> 1.81` | $-27.28 | `3->3 wins (+0.33R -> +0.41R)` | `ROLLED_BACK` |
| 3 | `OOS_D15` | SCALP_0DTE (5m) | 5 | `4 Entry / 1 Exit` | `0.0740 / 0.7275` | `1.85 -> 1.95` | $+17.29 | `3->3 wins (+0.33R -> +0.32R)` | `ROLLED_BACK` |
| 1 | `OOS_D15` | BRAIN-01 (Penny Day) | 0 | `0 Entry / 0 Exit` | `0.0521 / 0.7687` | `0.00 -> 0.00` | $-120.72 | `0->0 wins (+0.00R -> +0.00R)` | `ROLLED_BACK` |
| 2 | `OOS_D15` | BRAIN-02 (Large-Cap Day) | 0 | `0 Entry / 0 Exit` | `0.0768 / 1.8092` | `0.00 -> 12.00` | $+0.00 | `0->1 wins (+0.00R -> +0.11R)` | `ROLLED_BACK` |
| 1 | `OOS_D17` | BRAIN-01 (Penny Day) | 0 | `0 Entry / 0 Exit` | `0.0410 / 0.5545` | `0.00 -> 0.00` | $+0.00 | `0->0 wins (+0.00R -> +0.00R)` | `ROLLED_BACK` |
| 3 | `OOS_D17` | BRAIN-02 (Large-Cap Day) | 0 | `0 Entry / 0 Exit` | `0.0742 / 1.8522` | `0.00 -> 2.71` | $-48.30 | `0->3 wins (Net +0.00R -> +0.21R)` | **PROMOTED** |
| 4 | `OOS_D19` | SCALP_0DTE (5m) | 3 | `3 Entry / 0 Exit` | `0.0467 / 0.4397` | `0.42 -> 2.67` | $+11.01 | `1->2 wins (Net -0.06R -> +0.11R)` | **PROMOTED** |
| 1 | `OOS_D19` | BRAIN-01 (Penny Day) | 0 | `0 Entry / 0 Exit` | `0.0645 / 0.8256` | `0.00 -> 0.00` | $+0.00 | `0->0 wins (+0.00R -> +0.00R)` | `ROLLED_BACK` |
| 3 | `OOS_D19` | BRAIN-02 (Large-Cap Day) | 3 | `3 Entry / 0 Exit` | `0.0135 / 0.0825` | `2.71 -> 5.03` | $-6.64 | `3->3 wins (+0.32R -> +0.60R)` | `ROLLED_BACK` |
| 4 | `OOS_D20` | BRAIN-02 (Large-Cap Day) | 3 | `3 Entry / 0 Exit` | `0.0386 / 0.3143` | `1.38 -> 5.36` | $+279.37 | `5->4 wins (Net +0.30R -> +0.94R)` | **PROMOTED** |
| 1 | `OOS_D21` | BRAIN-01 (Penny Day) | 0 | `0 Entry / 0 Exit` | `0.1604 / 2.5876` | `0.00 -> 0.00` | $+0.00 | `0->0 wins (+0.00R -> +0.00R)` | `ROLLED_BACK` |
| 5 | `OOS_D22` | BRAIN-02 (Large-Cap Day) | 1 | `1 Entry / 0 Exit` | `0.0111 / 0.0885` | `7.48 -> 46.82` | $+45.47 | `3->3 wins (Net +0.88R -> +1.17R)` | **PROMOTED** |
| 4 | `OOS_D23` | SCALP_0DTE (5m) | 0 | `0 Entry / 0 Exit` | `0.1920 / 2.3436` | `12.00 -> 0.00` | $-8.16 | `1->0 wins (+0.19R -> +0.00R)` | `ROLLED_BACK` |
| 6 | `OOS_D24` | BRAIN-02 (Large-Cap Day) | 1 | `1 Entry / 0 Exit` | `0.0124 / 0.0550` | `62.78 -> 65.94` | $+0.00 | `5->6 wins (Net +1.61R -> +1.65R)` | **PROMOTED** |
| 6 | `OOS_D25` | BRAIN-02 (Large-Cap Day) | 5 | `4 Entry / 1 Exit` | `0.1223 / 1.2428` | `1.41 -> 0.14` | $+173.08 | `7->2 wins (+1.82R -> +0.14R)` | `ROLLED_BACK` |
| 4 | `OOS_D27` | SCALP_0DTE (5m) | 0 | `0 Entry / 0 Exit` | `0.1828 / 2.4071` | `12.00 -> 12.00` | $+0.00 | `1->2 wins (+0.19R -> +0.15R)` | `ROLLED_BACK` |
| 5 | `OOS_D29` | SCALP_0DTE (5m) | 0 | `0 Entry / 0 Exit` | `0.3394 / 3.4098` | `0.00 -> 2.87` | $-9.06 | `0->2 wins (Net +0.00R -> +0.04R)` | **PROMOTED** |
| 6 | `OOS_D29` | BRAIN-02 (Large-Cap Day) | 6 | `5 Entry / 1 Exit` | `0.1039 / 1.0470` | `1.27 -> 1.00` | $+17.55 | `7->4 wins (+1.74R -> +1.40R)` | `ROLLED_BACK` |
| 6 | `OOS_D31` | SCALP_0DTE (5m) | 4 | `1 Entry / 3 Exit` | `0.0381 / 0.4699` | `0.16 -> 2.98` | $+147.36 | `2->2 wins (Net -0.39R -> +0.05R)` | **PROMOTED** |
| 6 | `OOS_D31` | BRAIN-02 (Large-Cap Day) | 7 | `5 Entry / 2 Exit` | `0.1132 / 1.0819` | `1.38 -> 1.23` | $+11.40 | `9->8 wins (+2.22R -> +2.03R)` | `ROLLED_BACK` |
| 6 | `OOS_D34` | SCALP_0DTE (5m) | 3 | `2 Entry / 1 Exit` | `0.0574 / 1.0263` | `0.28 -> 0.27` | $+1.19 | `3->3 wins (+0.16R -> +0.16R)` | `ROLLED_BACK` |
| 6 | `OOS_D35` | SCALP_0DTE (5m) | 4 | `3 Entry / 1 Exit` | `0.0631 / 1.1109` | `0.16 -> 0.14` | $+51.78 | `3->2 wins (+0.16R -> +0.13R)` | `ROLLED_BACK` |
| 7 | `OOS_D35` | BRAIN-02 (Large-Cap Day) | 10 | `8 Entry / 2 Exit` | `0.0985 / 1.0141` | `0.58 -> 12.00` | $+1148.43 | `9->2 wins (Net -1.14R -> +0.17R)` | **PROMOTED** |
| 6 | `OOS_D39` | SCALP_0DTE (5m) | 5 | `4 Entry / 1 Exit` | `0.0656 / 1.0173` | `0.15 -> 0.14` | $+62.41 | `3->2 wins (+0.16R -> +0.13R)` | `ROLLED_BACK` |
| 1 | `OOS_D1` | BRAIN-03 (Short Swing) | 2 | `0 Entry / 2 Exit` | `0.2403 / 2.8963` | `8.28 -> 3.34` | $-233.61 | `5->6 wins (+2.02R -> +1.79R)` | `ROLLED_BACK` |
| 1 | `OOS_D4` | BRAIN-03 (Short Swing) | 3 | `0 Entry / 3 Exit` | `0.4124 / 6.1042` | `7.81 -> 26.09` | $+170.72 | `5->5 wins (+3.09R -> +3.07R)` | `ROLLED_BACK` |
| 1 | `OOS_D6` | BRAIN-03 (Short Swing) | 3 | `1 Entry / 2 Exit` | `0.3546 / 4.7376` | `6.49 -> 9.58` | $+112.81 | `7->7 wins (+3.22R -> +3.04R)` | `ROLLED_BACK` |
| 1 | `OOS_D8` | BRAIN-03 (Short Swing) | 3 | `1 Entry / 2 Exit` | `0.3196 / 4.0181` | `5.37 -> 4.49` | $-39.84 | `7->6 wins (+3.12R -> +3.09R)` | `ROLLED_BACK` |
| 1 | `OOS_D9` | BRAIN-03 (Short Swing) | 3 | `2 Entry / 1 Exit` | `0.1854 / 2.0807` | `7.70 -> 6.99` | $-5.05 | `7->6 wins (+3.27R -> +3.19R)` | `ROLLED_BACK` |
| 1 | `OOS_D11` | BRAIN-03 (Short Swing) | 1 | `1 Entry / 0 Exit` | `0.1812 / 2.6767` | `20.45 -> 30.43` | $+29.05 | `7->8 wins (+3.64R -> +3.65R)` | `ROLLED_BACK` |
| 1 | `OOS_D13` | BRAIN-03 (Short Swing) | 1 | `1 Entry / 0 Exit` | `0.1812 / 2.6767` | `19.88 -> 5.21` | $-561.82 | `7->7 wins (+3.43R -> +3.49R)` | `ROLLED_BACK` |
| 1 | `OOS_D15` | BRAIN-03 (Short Swing) | 1 | `1 Entry / 0 Exit` | `0.1812 / 2.6767` | `12.30 -> 8.60` | $-408.87 | `7->9 wins (+3.81R -> +4.58R)` | `ROLLED_BACK` |
| 1 | `OOS_D17` | BRAIN-03 (Short Swing) | 1 | `1 Entry / 0 Exit` | `0.0881 / 1.0764` | `17.52 -> 7.30` | $-110.65 | `7->5 wins (+2.72R -> +2.84R)` | `ROLLED_BACK` |
| 1 | `OOS_D19` | BRAIN-03 (Short Swing) | 3 | `2 Entry / 1 Exit` | `0.1855 / 1.7998` | `3.67 -> 12.00` | $+444.92 | `7->1 wins (+2.72R -> +1.92R)` | `ROLLED_BACK` |
| 1 | `OOS_D20` | BRAIN-03 (Short Swing) | 3 | `2 Entry / 1 Exit` | `0.2690 / 2.9139` | `3.56 -> 3.56` | $+37.72 | `6->5 wins (+2.59R -> +2.71R)` | `ROLLED_BACK` |
| 1 | `OOS_D22` | BRAIN-03 (Short Swing) | 3 | `2 Entry / 1 Exit` | `0.2380 / 2.3805` | `5.07 -> 9.06` | $+261.90 | `5->4 wins (+4.25R -> +3.40R)` | `ROLLED_BACK` |
| 1 | `OOS_D1` | BRAIN-04 (Core Swing) | 9 | `2 Entry / 7 Exit` | `0.4847 / 6.9947` | `1.52 -> 0.00` | $+1052.47 | `1->0 wins (+1.64R -> +0.00R)` | `ROLLED_BACK` |
| 1 | `OOS_D3` | BRAIN-04 (Core Swing) | 6 | `1 Entry / 5 Exit` | `0.3670 / 6.8861` | `0.93 -> 3.47` | $+123.87 | `4->7 wins (+2.14R -> +4.67R)` | `ROLLED_BACK` |
| 1 | `OOS_D5` | BRAIN-04 (Core Swing) | 7 | `1 Entry / 6 Exit` | `0.3783 / 6.4241` | `3.69 -> 3.02` | $+69.87 | `2->2 wins (+4.85R -> +4.30R)` | `ROLLED_BACK` |
| 1 | `OOS_D7` | BRAIN-04 (Core Swing) | 4 | `1 Entry / 3 Exit` | `0.8093 / 11.3314` | `4.86 -> 4.86` | $-0.00 | `4->4 wins (+4.75R -> +4.75R)` | `ROLLED_BACK` |
| 1 | `OOS_D8` | BRAIN-04 (Core Swing) | 4 | `1 Entry / 3 Exit` | `0.9257 / 13.1009` | `4.48 -> 4.48` | $-0.00 | `5->5 wins (+4.85R -> +4.85R)` | `ROLLED_BACK` |
| 1 | `OOS_D10` | BRAIN-04 (Core Swing) | 8 | `3 Entry / 5 Exit` | `0.8035 / 11.1137` | `1.06 -> 2.48` | $+626.02 | `4->9 wins (+3.13R -> +3.67R)` | `ROLLED_BACK` |
| 1 | `OOS_D12` | BRAIN-04 (Core Swing) | 4 | `1 Entry / 3 Exit` | `0.2540 / 5.3944` | `4.34 -> 3.12` | $+281.25 | `5->9 wins (+5.94R -> +3.94R)` | `ROLLED_BACK` |
| 1 | `OOS_D14` | BRAIN-04 (Core Swing) | 4 | `1 Entry / 3 Exit` | `0.2077 / 4.4951` | `6.25 -> 4.74` | $+210.81 | `5->8 wins (+6.35R -> +5.84R)` | `ROLLED_BACK` |
| 1 | `OOS_D16` | BRAIN-04 (Core Swing) | 4 | `0 Entry / 4 Exit` | `0.6025 / 10.8067` | `3.43 -> 2.31` | $-54.46 | `5->5 wins (+3.92R -> +3.82R)` | `ROLLED_BACK` |
| 1 | `OOS_D17` | BRAIN-04 (Core Swing) | 3 | `0 Entry / 3 Exit` | `0.7631 / 13.2257` | `5.55 -> 6.16` | $-49.48 | `5->6 wins (+4.66R -> +6.10R)` | `ROLLED_BACK` |
| 1 | `OOS_D19` | BRAIN-04 (Core Swing) | 3 | `0 Entry / 3 Exit` | `0.7631 / 13.2257` | `5.32 -> 5.61` | $-49.48 | `5->6 wins (+4.44R -> +5.53R)` | `ROLLED_BACK` |
| 1 | `OOS_D22` | BRAIN-04 (Core Swing) | 4 | `2 Entry / 2 Exit` | `0.4145 / 7.4971` | `1.18 -> 0.56` | $-1308.25 | `6->4 wins (+2.30R -> +2.49R)` | `ROLLED_BACK` |
| 2 | `OOS_D28` | BRAIN-04 (Core Swing) | 2 | `2 Entry / 0 Exit` | `0.4860 / 7.6802` | `1.11 -> 1.92` | $+275.13 | `7->8 wins (Net +0.59R -> +0.97R)` | **PROMOTED** |
| 3 | `OOS_D34` | BRAIN-04 (Core Swing) | 4 | `1 Entry / 3 Exit` | `0.2840 / 4.1927` | `3.34 -> 4.67` | $+25.96 | `10->7 wins (Net +2.29R -> +4.34R)` | **PROMOTED** |
| 4 | `OOS_D36` | BRAIN-04 (Core Swing) | 1 | `1 Entry / 0 Exit` | `0.5839 / 10.6201` | `5.80 -> 7.44` | $+83.84 | `8->10 wins (Net +6.25R -> +6.88R)` | **PROMOTED** |

## Autonomous 4-Vector Loss-Autopsy Attribution (`RUN_B_ONLINE_RL`)

| Bucket | Friction Inversion | Holding / EOD Decay | Vehicle Regret | Paralysis Regret | Flat Windows Replayed | Cost Sens (Before -> After) | Promoted Gens |
|---|---:|---:|---:|---:|---:|---|---:|
| SCALP_0DTE (1m) | 9 | 9 | 0 | 64 | 1153 | `0.725 -> 0.764` | 5 |
| SCALP_0DTE (5m) | 12 | 5 | 0 | 192 | 1529 | `1.078 -> 1.529` | 6 |
| BRAIN-01 (Penny Day) | 1 | 0 | 0 | 704 | 2413 | `1.389 -> 1.389` | 1 |
| BRAIN-02 (Large-Cap Day) | 13 | 3 | 2 | 256 | 2028 | `1.127 -> 0.887` | 7 |
| BRAIN-03 (Short Swing) | 8 | 9 | 0 | 0 | 704 | `0.699 -> 0.699` | 1 |
| BRAIN-04 (Core Swing) | 13 | 31 | 2 | 0 | 640 | `0.689 -> 0.740` | 4 |
