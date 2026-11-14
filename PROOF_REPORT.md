# ZERO-CHEAT END-TO-END DIFFERENTIABLE BPTT + CLOSED-LOOP LOSS-AUTOPSY RL PROOF REPORT

- **Generated At (UTC):** `2026-11-14T02:52:50.880242+00:00`
- **Dataset 1 (`1m` Intraday + Causal `5m` Session Pre-Training):** `21` total `1m` trading days (`11` OOS days), `167411` real `1m` OHLCV bars loaded
- **Dataset 2 (`5m` & `1h` 60-Ticker Universe):** `59` real US symbols across `65` total trading days (`24` Warmup Calibration -> `41` Strictly Out-of-Sample)
- **Zero-Rule Neural Architecture (`src/pure_rl_brain.rs`):** Stage A Multi-Horizon Physics World Model (`1, 3, 8, 24`-bar Pinball Quantile Loss + `ln(Range/RT_Cost)`) + Stage B `K=5` Orthogonal Specialist Policy Ensemble (`D=64`, `[h_ln, wm_fused, self_state, pos_interact]` 92-d skip-head) with Localized Boundary Surrogate Deadzone (`sech^2((|z| - tau) / temp)`), Unconstrained Differentiable Log-Space Cost Sensitivity (`k_theta = exp(log_k_theta)` via analytical `dL_net / d log_k_theta`), Turnover-Amortized Opportunity Critic (`Q_hold(x_t, s_t) - Q_flat(x_t, s_t)`), Purged `T_holdout` Cross-Sectional LCB Promotion Gate, and `M=24` `rayon` PBT + Multiplicative-Weights Routing.
- **Anti-Cheat Execution Physics:** Zero `> t` lookahead, next-bar `t+1` open fills with `5/12/35 bps` adverse equity slippage, `09:35–15:15 ET` flatten `15:50 ET` (`SCALP_0DTE`), `09:35–15:30 ET` flatten `15:55 ET` (`DAY`), skew-adjusted BSM + `^VIX9D`/`^VIX` IV + `$0.03–$0.08`/sh/leg spread + `$0.65`/contract/leg RT commission.
- **BPTT & Closed-Loop Loss-Autopsy Activity:** `249577` Causal Warmup Episode Bars | `293212` OOS Causal Episode Bars | `125` Online EOD RL & Routing Cycles (`RUN_B_ONLINE_RL`) | `238428` Autonomous `FLAT` Abstentions.

## Causal Walk-Forward Inner PBT Calibration (`t <= T_oos_start`)

| Brain / Horizon | Population | Session Episodes | PBT Gens | Best Inner-Val Trades | Best Inner-Val Net PF | Best Inner-Val Net PnL ($) |
|---|---:|---:|---:|---:|---:|---:|
| SCALP_0DTE (1m) | 24 | 150 | 3 | 4 | **0.01** | $-147.48 |
| SCALP_0DTE (5m) | 24 | 192 | 3 | 7 | **3.28** | $+332.29 |
| BRAIN-01 (Penny Day) | 24 | 379 | 3 | 3 | **0.70** | $-17.51 |
| BRAIN-02 (Large-Cap Day) | 24 | 818 | 3 | 2 | **12.00** | $+93.40 |
| BRAIN-03 (Short Swing) | 24 | 411 | 3 | 10 | **1.35** | $+312.28 |
| BRAIN-04 (Core Swing) | 24 | 411 | 3 | 19 | **1.31** | $+972.95 |

## Executive Scorecard: `RUN_A_FROZEN_WARMUP_ML` vs `RUN_B_ONLINE_RL` (Gross vs Net Post-Cost)

| Bucket | Run Mode | Trades | Win% | Gross PF | Net Post-Cost PF | Net Exp (R) | Net Exp ($) | Gross PnL ($) | Cost Drag ($) | Net PnL ($) | Max DD ($ / %) | Backstop% | Neural-Exit% | EOD Flatten% |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SCALP_0DTE (1m) | `RUN_A_FROZEN_WARMUP_ML` | 12 | 58.3% | 3.78 | **2.66** | +0.077R | $+47.77 | $+757.47 | $184.26 | **$+573.21** | $298.52 (0.30%) | 0.0% | 100.0% | 0.0% |
| SCALP_0DTE (1m) | `RUN_B_ONLINE_RL` | 12 | 58.3% | 3.78 | **2.66** | +0.077R | $+47.77 | $+757.47 | $184.26 | **$+573.21** | $298.52 (0.30%) | 0.0% | 100.0% | 0.0% |
| SCALP_0DTE (5m) | `RUN_A_FROZEN_WARMUP_ML` | 67 | 20.9% | 0.28 | **0.18** | -0.092R | $-50.02 | $-2459.68 | $891.86 | **$-3351.54** | $3463.57 (3.46%) | 0.0% | 97.0% | 3.0% |
| SCALP_0DTE (5m) | `RUN_B_ONLINE_RL` | 82 | 23.2% | 0.35 | **0.22** | -0.068R | $-36.86 | $-2043.97 | $978.91 | **$-3022.88** | $3083.75 (3.08%) | 0.0% | 95.1% | 4.9% |
| BRAIN-01 (Penny Day) | `RUN_A_FROZEN_WARMUP_ML` | 35 | 28.6% | 1.16 | **0.64** | -0.092R | $-49.30 | $+542.51 | $2268.00 | **$-1725.49** | $1725.49 (1.73%) | 2.9% | 54.3% | 42.9% |
| BRAIN-01 (Penny Day) | `RUN_B_ONLINE_RL` | 46 | 32.6% | 1.44 | **0.54** | -0.066R | $-38.46 | $+977.08 | $2746.27 | **$-1769.18** | $1929.03 (1.93%) | 0.0% | 71.7% | 28.3% |
| BRAIN-02 (Large-Cap Day) | `RUN_A_FROZEN_WARMUP_ML` | 24 | 37.5% | 0.58 | **0.31** | -0.050R | $-26.43 | $-303.82 | $330.43 | **$-634.25** | $634.25 (0.63%) | 0.0% | 83.3% | 16.7% |
| BRAIN-02 (Large-Cap Day) | `RUN_B_ONLINE_RL` | 59 | 28.8% | 0.61 | **0.42** | -0.061R | $-32.37 | $-1061.32 | $848.50 | **$-1909.82** | $2265.36 (2.26%) | 0.0% | 93.2% | 6.8% |
| BRAIN-03 (Short Swing) | `RUN_A_FROZEN_WARMUP_ML` | 12 | 75.0% | 3.75 | **3.49** | +0.295R | $+170.10 | $+2171.12 | $129.91 | **$+2041.22** | $532.17 (0.53%) | 8.3% | 83.3% | 0.0% |
| BRAIN-03 (Short Swing) | `RUN_B_ONLINE_RL` | 16 | 56.2% | 3.17 | **2.81** | +0.209R | $+124.30 | $+2191.46 | $202.60 | **$+1988.86** | $734.77 (0.73%) | 6.2% | 75.0% | 0.0% |
| BRAIN-04 (Core Swing) | `RUN_A_FROZEN_WARMUP_ML` | 27 | 33.3% | 0.41 | **0.35** | -0.136R | $-99.91 | $-2300.44 | $397.11 | **$-2697.55** | $3352.08 (3.35%) | 7.4% | 85.2% | 0.0% |
| BRAIN-04 (Core Swing) | `RUN_B_ONLINE_RL` | 21 | 23.8% | 0.31 | **0.26** | -0.186R | $-104.62 | $-1922.65 | $274.33 | **$-2196.98** | $2517.71 (2.52%) | 4.8% | 90.5% | 0.0% |
| ALL_BUCKETS_COMBINED | `RUN_A_FROZEN_WARMUP_ML` | 177 | 32.8% | 0.87 | **0.62** | -0.056R | $-32.74 | $-1592.84 | $4201.57 | **$-5794.41** | $7320.67 (7.26%) | 2.3% | 84.2% | 11.9% |
| ALL_BUCKETS_COMBINED | `RUN_B_ONLINE_RL` | 236 | 30.5% | 0.91 | **0.59** | -0.050R | $-26.85 | $-1101.93 | $5234.86 | **$-6336.79** | $7611.50 (7.55%) | 0.8% | 88.6% | 8.9% |

## Horizon Tier Rollups (`RUN_B_ONLINE_RL`)

| Tier | Run Mode | Trades | Win% | Gross PF | Net Post-Cost PF | Net Exp (R) | Net Exp ($) | Gross PnL ($) | Cost Drag ($) | Net PnL ($) | Max DD ($ / %) | Backstop% | Neural-Exit% | EOD Flatten% |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TIER: SCALP_0DTE | `RUN_B_ONLINE_RL` | 94 | 27.7% | 0.62 | **0.42** | -0.049R | $-26.06 | $-1286.50 | $1163.17 | **$-2449.67** | $3382.27 (3.35%) | 0.0% | 95.7% | 4.3% |
| TIER: DAY (BRAIN-01 + BRAIN-02) | `RUN_B_ONLINE_RL` | 105 | 30.5% | 0.98 | **0.48** | -0.063R | $-35.04 | $-84.24 | $3594.76 | **$-3679.01** | $4036.42 (4.04%) | 0.0% | 83.8% | 16.2% |
| TIER: SWING (BRAIN-03 + BRAIN-04) | `RUN_B_ONLINE_RL` | 37 | 37.8% | 1.07 | **0.95** | -0.015R | $-5.62 | $+268.81 | $476.92 | **$-208.11** | $2568.77 (2.52%) | 5.4% | 83.8% | 0.0% |

## Factored Vehicle Performance Split (`V_SHARES` vs `V_OPTION`)

| Run Mode | Vehicle | Trades | Win% | Gross PF | Net Post-Cost PF | Net Exp (R) | Avg Cost/Trade ($) | Total Gross PnL ($) | Total Cost Drag ($) | Total Net PnL ($) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `RUN_A_FROZEN_WARMUP_ML` | **V_SHARES** | 177 | 32.8% | 0.87 | **0.62** | -0.056R | $23.74 | $-1592.84 | $4201.57 | **$-5794.41** |
| `RUN_A_FROZEN_WARMUP_ML` | **V_OPTION** | 0 | 0.0% | 0.00 | **0.00** | +0.000R | $0.00 | $+0.00 | $0.00 | **$+0.00** |
| `RUN_B_ONLINE_RL` | **V_SHARES** | 231 | 30.3% | 0.89 | **0.58** | -0.052R | $22.09 | $-1374.42 | $5102.16 | **$-6476.58** |
| `RUN_B_ONLINE_RL` | **V_OPTION** | 5 | 40.0% | 4.23 | **1.92** | +0.039R | $26.54 | $+272.49 | $132.70 | **$+139.79** |

## A/B Online Learning Delta (`RUN_B_ONLINE_RL` minus `RUN_A_FROZEN_WARMUP_ML`)

| Bucket | Δ Trades | Δ Win% | Δ Net PF | Δ Net Exp (R) | Δ Net PnL ($) | Δ Max DD ($) |
|---|---:|---:|---:|---:|---:|---:|
| SCALP_0DTE (1m) | +0 | +0.0% | +0.00 | +0.000R | **$+0.00** | $+0.00 |
| SCALP_0DTE (5m) | +15 | +2.3% | +0.04 | +0.025R | **$+328.66** | $-379.82 |
| BRAIN-01 (Penny Day) | +11 | +4.0% | -0.10 | +0.026R | **$-43.69** | $+203.54 |
| BRAIN-02 (Large-Cap Day) | +35 | -8.7% | +0.11 | -0.011R | **$-1275.57** | $+1631.11 |
| BRAIN-03 (Short Swing) | +4 | -18.8% | -0.68 | -0.086R | **$-52.35** | $+202.60 |
| BRAIN-04 (Core Swing) | -6 | -9.5% | -0.09 | -0.049R | **$+500.57** | $-834.37 |
| **DATASET_1_1M_TOTAL** | — | — | — | — | **$+0.00** | — |
| **DATASET_2_5M_1H_TOTAL** | — | — | — | — | **$-542.38** | — |
| **ALL_BUCKETS_COMBINED** | **+59** | **-2.3%** | **-0.03** | **+0.005R** | **$-542.38** | **$+290.83** |

## Loss-Autopsy -> Re-Test -> Generation Promotion Ledger

| Gen ID | Phase | Brain | Red Trades Audited | Entry vs Exit Fault Split | Saliency (`|dL/dx|` / `|dL/dh|`) | Re-Test Net PF (Before -> After) | Red Loss Reduced ($) | Winners Preserved | Outcome |
|---:|---|---|---:|---|---|---|---:|---|---|
| 1 | `WARMUP_CAL` | SCALP_0DTE (1m) | 3 | `2 Entry / 1 Exit` | `0.1686 / 1.6157` | `0.01 -> 0.01` | $+22.58 | `1->1 wins (+0.00R -> +0.00R)` | `ROLLED_BACK` |
| 1 | `WARMUP_CAL` | BRAIN-01 (Penny Day) | 3 | `1 Entry / 2 Exit` | `0.2242 / 1.2353` | `0.70 -> 12.00` | $+58.95 | `1->1 wins (+0.07R -> +0.07R)` | `ROLLED_BACK` |
| 1 | `WARMUP_CAL` | BRAIN-02 (Large-Cap Day) | 0 | `0 Entry / 0 Exit` | `0.1049 / 3.1253` | `12.00 -> 2.08` | $-254.44 | `2->7 wins (+0.17R -> +0.96R)` | `ROLLED_BACK` |
| 1 | `OOS_D1` | SCALP_0DTE (1m) | 3 | `2 Entry / 1 Exit` | `0.1686 / 1.6157` | `5.58 -> 4.47` | $-645.42 | `6->24 wins (+1.50R -> +6.13R)` | `ROLLED_BACK` |
| 1 | `OOS_D3` | SCALP_0DTE (1m) | 3 | `2 Entry / 1 Exit` | `0.1686 / 1.6157` | `5.58 -> 6.92` | $-197.41 | `6->5 wins (+1.50R -> +4.23R)` | `ROLLED_BACK` |
| 1 | `OOS_D4` | SCALP_0DTE (1m) | 4 | `3 Entry / 1 Exit` | `0.1753 / 1.7156` | `4.26 -> 7.83` | $-2.54 | `6->9 wins (+1.50R -> +2.82R)` | `ROLLED_BACK` |
| 1 | `OOS_D6` | SCALP_0DTE (1m) | 4 | `3 Entry / 1 Exit` | `0.1753 / 1.7156` | `6.50 -> 6.05` | $-17.97 | `10->9 wins (+2.30R -> +2.34R)` | `ROLLED_BACK` |
| 1 | `OOS_D8` | SCALP_0DTE (1m) | 5 | `3 Entry / 2 Exit` | `0.1423 / 1.3951` | `7.92 -> 3.12` | $-132.13 | `13->11 wins (+3.06R -> +2.01R)` | `ROLLED_BACK` |
| 1 | `OOS_D10` | SCALP_0DTE (1m) | 5 | `3 Entry / 2 Exit` | `0.1423 / 1.3951` | `7.92 -> 2.57` | $-239.05 | `13->12 wins (+3.06R -> +2.12R)` | `ROLLED_BACK` |
| 1 | `OOS_D1` | SCALP_0DTE (5m) | 8 | `5 Entry / 3 Exit` | `0.0974 / 0.9159` | `5.14 -> 3.49` | $+6.29 | `15->12 wins (+4.74R -> +3.20R)` | `ROLLED_BACK` |
| 1 | `OOS_D1` | BRAIN-01 (Penny Day) | 3 | `1 Entry / 2 Exit` | `0.2242 / 1.2353` | `1.67 -> 3.13` | $-126.46 | `2->5 wins (+0.17R -> +1.04R)` | `ROLLED_BACK` |
| 1 | `OOS_D1` | BRAIN-02 (Large-Cap Day) | 0 | `0 Entry / 0 Exit` | `0.1160 / 2.5313` | `12.00 -> 12.00` | $+0.00 | `2->2 wins (+0.17R -> +0.15R)` | `ROLLED_BACK` |
| 1 | `OOS_D2` | SCALP_0DTE (5m) | 14 | `10 Entry / 4 Exit` | `0.0787 / 0.7458` | `3.52 -> 3.98` | $+474.78 | `15->8 wins (+4.74R -> +2.11R)` | `ROLLED_BACK` |
| 1 | `OOS_D3` | SCALP_0DTE (5m) | 17 | `11 Entry / 6 Exit` | `0.0763 / 0.7909` | `2.58 -> 2.36` | $+149.01 | `15->11 wins (+4.45R -> +3.45R)` | `ROLLED_BACK` |
| 1 | `OOS_D3` | BRAIN-01 (Penny Day) | 3 | `1 Entry / 2 Exit` | `0.2242 / 1.2353` | `1.67 -> 0.00` | $-92.18 | `2->0 wins (+0.17R -> +0.00R)` | `ROLLED_BACK` |
| 1 | `OOS_D3` | BRAIN-02 (Large-Cap Day) | 0 | `0 Entry / 0 Exit` | `0.1110 / 2.3544` | `12.00 -> 12.00` | $+0.00 | `2->2 wins (+0.17R -> +0.15R)` | `ROLLED_BACK` |
| 1 | `OOS_D5` | SCALP_0DTE (5m) | 18 | `11 Entry / 7 Exit` | `0.0738 / 0.7604` | `1.75 -> 1.87` | $+562.68 | `11->7 wins (+3.43R -> +1.93R)` | `ROLLED_BACK` |
| 1 | `OOS_D5` | BRAIN-01 (Penny Day) | 3 | `1 Entry / 2 Exit` | `0.2242 / 1.2353` | `0.70 -> 0.35` | $-46.19 | `1->1 wins (+0.07R -> +0.07R)` | `ROLLED_BACK` |
| 2 | `OOS_D5` | BRAIN-02 (Large-Cap Day) | 0 | `0 Entry / 0 Exit` | `0.1187 / 3.3252` | `12.00 -> 12.00` | $+0.00 | `2->3 wins (Net +0.17R -> +0.38R)` | **PROMOTED** |
| 1 | `OOS_D6` | SCALP_0DTE (5m) | 20 | `13 Entry / 7 Exit` | `0.0874 / 0.8227` | `1.64 -> 1.63` | $+421.60 | `13->13 wins (+3.49R -> +2.35R)` | `ROLLED_BACK` |
| 1 | `OOS_D7` | BRAIN-01 (Penny Day) | 3 | `1 Entry / 2 Exit` | `0.2242 / 1.2353` | `0.70 -> 12.00` | $+58.95 | `1->1 wins (+0.07R -> +0.07R)` | `ROLLED_BACK` |
| 2 | `OOS_D7` | BRAIN-02 (Large-Cap Day) | 2 | `1 Entry / 1 Exit` | `0.1654 / 1.0768` | `12.00 -> 12.00` | $+0.00 | `3->3 wins (+0.38R -> +0.38R)` | `ROLLED_BACK` |
| 1 | `OOS_D9` | BRAIN-01 (Penny Day) | 3 | `1 Entry / 2 Exit` | `0.2242 / 1.2353` | `0.70 -> 0.69` | $-390.25 | `1->4 wins (+0.07R -> +0.55R)` | `ROLLED_BACK` |
| 2 | `OOS_D9` | BRAIN-02 (Large-Cap Day) | 2 | `1 Entry / 1 Exit` | `0.1654 / 1.0768` | `12.00 -> 6.76` | $-56.15 | `3->5 wins (+0.38R -> +0.71R)` | `ROLLED_BACK` |
| 1 | `OOS_D10` | SCALP_0DTE (5m) | 19 | `12 Entry / 7 Exit` | `0.0889 / 0.8371` | `0.98 -> 1.26` | $+818.13 | `10->5 wins (+1.96R -> +0.74R)` | `ROLLED_BACK` |
| 1 | `OOS_D10` | BRAIN-01 (Penny Day) | 4 | `2 Entry / 2 Exit` | `0.2812 / 1.7720` | `0.11 -> 0.88` | $+70.69 | `1->3 wins (+0.07R -> +0.47R)` | `ROLLED_BACK` |
| 2 | `OOS_D11` | BRAIN-02 (Large-Cap Day) | 2 | `1 Entry / 1 Exit` | `0.1654 / 1.0768` | `12.00 -> 6.47` | $-37.63 | `3->2 wins (+0.38R -> +0.46R)` | `ROLLED_BACK` |
| 1 | `OOS_D12` | SCALP_0DTE (5m) | 24 | `13 Entry / 11 Exit` | `0.0908 / 0.8520` | `0.52 -> 2.57` | $+1507.65 | `8->2 wins (+1.42R -> +0.33R)` | `ROLLED_BACK` |
| 1 | `OOS_D12` | BRAIN-01 (Penny Day) | 4 | `2 Entry / 2 Exit` | `0.2812 / 1.7720` | `0.11 -> 0.23` | $-296.42 | `1->3 wins (+0.07R -> +0.27R)` | `ROLLED_BACK` |
| 2 | `OOS_D13` | BRAIN-02 (Large-Cap Day) | 2 | `1 Entry / 1 Exit` | `0.1654 / 1.0768` | `12.00 -> 12.00` | $+0.00 | `3->2 wins (+0.38R -> +0.26R)` | `ROLLED_BACK` |
| 1 | `OOS_D14` | BRAIN-01 (Penny Day) | 4 | `2 Entry / 2 Exit` | `0.2812 / 1.7720` | `0.11 -> 0.00` | $-270.36 | `1->0 wins (+0.07R -> +0.00R)` | `ROLLED_BACK` |
| 2 | `OOS_D15` | BRAIN-02 (Large-Cap Day) | 2 | `1 Entry / 1 Exit` | `0.1654 / 1.0768` | `12.00 -> 1.99` | $-289.94 | `3->7 wins (+0.38R -> +1.03R)` | `ROLLED_BACK` |
| 1 | `OOS_D16` | SCALP_0DTE (5m) | 20 | `10 Entry / 10 Exit` | `0.0939 / 0.8680` | `0.23 -> 0.20` | $+1059.54 | `4->2 wins (+0.47R -> +0.07R)` | `ROLLED_BACK` |
| 1 | `OOS_D16` | BRAIN-01 (Penny Day) | 4 | `2 Entry / 2 Exit` | `0.2812 / 1.7720` | `0.11 -> 0.32` | $+250.23 | `1->1 wins (+0.07R -> +0.07R)` | `ROLLED_BACK` |
| 3 | `OOS_D17` | BRAIN-02 (Large-Cap Day) | 2 | `1 Entry / 1 Exit` | `0.1654 / 1.0768` | `12.00 -> 12.00` | $+0.00 | `2->4 wins (Net +0.36R -> +0.76R)` | **PROMOTED** |
| 1 | `OOS_D18` | SCALP_0DTE (5m) | 23 | `13 Entry / 10 Exit` | `0.1041 / 0.8860` | `0.14 -> 0.51` | $+1415.97 | `4->2 wins (+0.38R -> +0.34R)` | `ROLLED_BACK` |
| 1 | `OOS_D18` | BRAIN-01 (Penny Day) | 3 | `1 Entry / 2 Exit` | `0.2010 / 1.4951` | `0.13 -> 0.00` | $-0.78 | `1->0 wins (+0.07R -> +0.00R)` | `ROLLED_BACK` |
| 4 | `OOS_D19` | BRAIN-02 (Large-Cap Day) | 0 | `0 Entry / 0 Exit` | `0.1441 / 2.3584` | `12.00 -> 11.22` | $-28.45 | `1->6 wins (Net +0.15R -> +0.53R)` | **PROMOTED** |
| 1 | `OOS_D20` | BRAIN-01 (Penny Day) | 1 | `1 Entry / 0 Exit` | `0.4522 / 3.3820` | `0.00 -> 0.00` | $+309.49 | `0->0 wins (+0.00R -> +0.00R)` | `ROLLED_BACK` |
| 4 | `OOS_D21` | BRAIN-02 (Large-Cap Day) | 4 | `2 Entry / 2 Exit` | `0.0736 / 0.5809` | `3.03 -> 2.27` | $-29.68 | `7->4 wins (+0.91R -> +0.84R)` | `ROLLED_BACK` |
| 1 | `OOS_D22` | SCALP_0DTE (5m) | 21 | `12 Entry / 9 Exit` | `0.0992 / 0.8413` | `0.14 -> 0.62` | $+1654.50 | `4->1 wins (+0.38R -> +0.04R)` | `ROLLED_BACK` |
| 1 | `OOS_D23` | SCALP_0DTE (5m) | 16 | `8 Entry / 8 Exit` | `0.1130 / 0.9341` | `0.18 -> 0.57` | $+1458.11 | `5->2 wins (+0.44R -> +0.10R)` | `ROLLED_BACK` |
| 4 | `OOS_D23` | BRAIN-02 (Large-Cap Day) | 4 | `2 Entry / 2 Exit` | `0.0736 / 0.5809` | `3.03 -> 1.28` | $-201.77 | `7->2 wins (+0.91R -> +0.83R)` | `ROLLED_BACK` |
| 1 | `OOS_D25` | SCALP_0DTE (5m) | 14 | `8 Entry / 6 Exit` | `0.1157 / 0.8573` | `0.07 -> 0.07` | $-2.13 | `5->6 wins (+0.17R -> +0.17R)` | `ROLLED_BACK` |
| 4 | `OOS_D25` | BRAIN-02 (Large-Cap Day) | 10 | `6 Entry / 4 Exit` | `0.0922 / 0.9915` | `0.68 -> 0.72` | $+5.45 | `8->5 wins (+0.99R -> +1.03R)` | `ROLLED_BACK` |
| 2 | `OOS_D28` | SCALP_0DTE (5m) | 13 | `7 Entry / 6 Exit` | `0.1115 / 0.8864` | `0.11 -> 8.87` | $+1313.76 | `5->8 wins (Net -2.14R -> +0.72R)` | **PROMOTED** |
| 2 | `OOS_D28` | BRAIN-01 (Penny Day) | 12 | `7 Entry / 5 Exit` | `0.3160 / 3.2438` | `0.42 -> 1.93` | $+882.25 | `4->9 wins (Net -1.68R -> +1.29R)` | **PROMOTED** |
| 4 | `OOS_D29` | BRAIN-02 (Large-Cap Day) | 10 | `6 Entry / 4 Exit` | `0.0922 / 0.9915` | `0.53 -> 0.73` | $+313.09 | `8->7 wins (+0.76R -> +0.64R)` | `ROLLED_BACK` |
| 3 | `OOS_D31` | SCALP_0DTE (5m) | 3 | `1 Entry / 2 Exit` | `0.0201 / 0.2411` | `1.49 -> 2.52` | $+73.76 | `6->5 wins (Net +0.15R -> +0.28R)` | **PROMOTED** |
| 3 | `OOS_D33` | BRAIN-01 (Penny Day) | 20 | `13 Entry / 7 Exit` | `0.2216 / 1.6246` | `1.10 -> 1.58` | $+688.23 | `13->12 wins (Net +0.40R -> +1.38R)` | **PROMOTED** |
| 4 | `OOS_D34` | SCALP_0DTE (5m) | 8 | `6 Entry / 2 Exit` | `0.0286 / 0.2745` | `1.53 -> 1.62` | $+11.81 | `8->9 wins (Net +0.23R -> +0.25R)` | **PROMOTED** |
| 4 | `OOS_D34` | BRAIN-01 (Penny Day) | 15 | `10 Entry / 5 Exit` | `0.1932 / 1.6911` | `1.55 -> 1.94` | $+362.88 | `12->14 wins (Net +1.33R -> +1.71R)` | **PROMOTED** |
| 2 | `OOS_D1` | BRAIN-03 (Short Swing) | 3 | `2 Entry / 1 Exit` | `0.1909 / 2.6994` | `2.31 -> 2.43` | $+1.28 | `7->7 wins (Net +1.35R -> +1.46R)` | **PROMOTED** |
| 2 | `OOS_D3` | BRAIN-03 (Short Swing) | 3 | `3 Entry / 0 Exit` | `0.1530 / 2.0935` | `1.38 -> 2.74` | $-146.25 | `6->15 wins (+1.72R -> +3.90R)` | `ROLLED_BACK` |
| 2 | `OOS_D5` | BRAIN-03 (Short Swing) | 3 | `2 Entry / 1 Exit` | `0.1892 / 2.5140` | `1.23 -> 1.40` | $-748.17 | `6->8 wins (+1.78R -> +3.89R)` | `ROLLED_BACK` |
| 2 | `OOS_D7` | BRAIN-03 (Short Swing) | 2 | `1 Entry / 1 Exit` | `0.2598 / 3.4097` | `1.45 -> 1.65` | $-27.04 | `6->7 wins (+1.93R -> +2.28R)` | `ROLLED_BACK` |
| 2 | `OOS_D9` | BRAIN-03 (Short Swing) | 2 | `1 Entry / 1 Exit` | `0.2598 / 3.4097` | `1.41 -> 1.18` | $-1306.02 | `6->14 wins (+1.91R -> +4.13R)` | `ROLLED_BACK` |
| 2 | `OOS_D11` | BRAIN-03 (Short Swing) | 2 | `1 Entry / 1 Exit` | `0.2598 / 3.4097` | `1.65 -> 1.77` | $-492.40 | `7->10 wins (+2.27R -> +3.93R)` | `ROLLED_BACK` |
| 2 | `OOS_D13` | BRAIN-03 (Short Swing) | 4 | `3 Entry / 1 Exit` | `0.2174 / 3.2797` | `1.53 -> 3.34` | $+440.22 | `7->7 wins (+2.26R -> +2.25R)` | `ROLLED_BACK` |
| 2 | `OOS_D14` | BRAIN-03 (Short Swing) | 3 | `2 Entry / 1 Exit` | `0.2502 / 3.7515` | `1.68 -> 1.75` | $-516.64 | `8->11 wins (+2.36R -> +3.74R)` | `ROLLED_BACK` |
| 2 | `OOS_D15` | BRAIN-03 (Short Swing) | 3 | `2 Entry / 1 Exit` | `0.2416 / 3.5905` | `1.78 -> 3.52` | $-114.38 | `8->7 wins (+2.36R -> +5.26R)` | `ROLLED_BACK` |
| 2 | `OOS_D17` | BRAIN-03 (Short Swing) | 3 | `2 Entry / 1 Exit` | `0.1822 / 2.8213` | `7.75 -> 1.66` | $-1888.36 | `6->13 wins (+2.71R -> +5.58R)` | `ROLLED_BACK` |
| 2 | `OOS_D19` | BRAIN-03 (Short Swing) | 3 | `1 Entry / 2 Exit` | `0.3616 / 4.5527` | `4.48 -> 2.24` | $-1174.56 | `6->10 wins (+2.69R -> +5.80R)` | `ROLLED_BACK` |
| 2 | `OOS_D21` | BRAIN-03 (Short Swing) | 3 | `2 Entry / 1 Exit` | `0.1821 / 2.8220` | `3.39 -> 2.42` | $-1288.37 | `5->11 wins (+1.14R -> +5.90R)` | `ROLLED_BACK` |
| 2 | `OOS_D23` | BRAIN-03 (Short Swing) | 4 | `3 Entry / 1 Exit` | `0.1385 / 2.1337` | `1.62 -> 12.00` | $+314.44 | `6->1 wins (+0.95R -> +1.11R)` | `ROLLED_BACK` |
| 3 | `OOS_D26` | BRAIN-03 (Short Swing) | 4 | `3 Entry / 1 Exit` | `0.1389 / 2.1402` | `1.90 -> 6.03` | $+384.44 | `7->17 wins (Net +1.36R -> +3.59R)` | **PROMOTED** |
| 2 | `OOS_D1` | BRAIN-04 (Core Swing) | 10 | `3 Entry / 7 Exit` | `0.4097 / 3.6089` | `1.25 -> 1.78` | $+1616.64 | `11->6 wins (Net +0.91R -> +1.68R)` | **PROMOTED** |
| 2 | `OOS_D2` | BRAIN-04 (Core Swing) | 7 | `3 Entry / 4 Exit` | `0.3692 / 2.7838` | `2.79 -> 1.81` | $+135.99 | `8->8 wins (+5.85R -> +3.13R)` | `ROLLED_BACK` |
| 2 | `OOS_D4` | BRAIN-04 (Core Swing) | 11 | `4 Entry / 7 Exit` | `0.3174 / 2.6718` | `2.63 -> 2.59` | $+1.80 | `6->6 wins (+5.84R -> +5.82R)` | `ROLLED_BACK` |
| 2 | `OOS_D6` | BRAIN-04 (Core Swing) | 10 | `4 Entry / 6 Exit` | `0.3564 / 3.0920` | `2.19 -> 1.75` | $-183.49 | `6->10 wins (+5.57R -> +4.78R)` | `ROLLED_BACK` |
| 2 | `OOS_D7` | BRAIN-04 (Core Swing) | 11 | `6 Entry / 5 Exit` | `0.4285 / 3.7191` | `1.62 -> 1.69` | $-385.05 | `5->6 wins (+4.11R -> +5.74R)` | `ROLLED_BACK` |
| 2 | `OOS_D8` | BRAIN-04 (Core Swing) | 13 | `5 Entry / 8 Exit` | `0.4112 / 3.6561` | `0.73 -> 2.17` | $+767.57 | `4->8 wins (+3.02R -> +6.11R)` | `ROLLED_BACK` |
| 2 | `OOS_D10` | BRAIN-04 (Core Swing) | 12 | `4 Entry / 8 Exit` | `0.4851 / 4.3115` | `0.73 -> 2.03` | $+1063.39 | `6->8 wins (+3.29R -> +5.20R)` | `ROLLED_BACK` |
| 2 | `OOS_D11` | BRAIN-04 (Core Swing) | 9 | `3 Entry / 6 Exit` | `0.7121 / 6.4420` | `1.80 -> 1.66` | $+131.01 | `9->9 wins (+7.20R -> +6.25R)` | `ROLLED_BACK` |
| 2 | `OOS_D12` | BRAIN-04 (Core Swing) | 14 | `5 Entry / 9 Exit` | `0.4906 / 4.2440` | `1.09 -> 1.81` | $-301.89 | `4->7 wins (+4.83R -> +7.86R)` | `ROLLED_BACK` |
| 2 | `OOS_D13` | BRAIN-04 (Core Swing) | 11 | `5 Entry / 6 Exit` | `0.5596 / 4.6101` | `1.35 -> 1.73` | $+1543.90 | `6->8 wins (+5.29R -> +2.79R)` | `ROLLED_BACK` |
| 3 | `OOS_D16` | BRAIN-04 (Core Swing) | 13 | `8 Entry / 5 Exit` | `0.4510 / 3.9237` | `0.66 -> 2.29` | $+1838.88 | `6->8 wins (Net -1.74R -> +2.42R)` | **PROMOTED** |
| 3 | `OOS_D17` | BRAIN-04 (Core Swing) | 4 | `4 Entry / 0 Exit` | `0.0811 / 1.7371` | `2.01 -> 3.74` | $-76.35 | `7->11 wins (+2.40R -> +4.97R)` | `ROLLED_BACK` |
| 3 | `OOS_D21` | BRAIN-04 (Core Swing) | 3 | `3 Entry / 0 Exit` | `0.1062 / 2.2909` | `0.85 -> 1.82` | $-152.07 | `3->12 wins (+0.97R -> +2.53R)` | `ROLLED_BACK` |
| 3 | `OOS_D25` | BRAIN-04 (Core Swing) | 2 | `1 Entry / 1 Exit` | `0.1770 / 1.3097` | `2.14 -> 1.42` | $-401.15 | `5->7 wins (+0.79R -> +1.62R)` | `ROLLED_BACK` |
| 4 | `OOS_D36` | BRAIN-04 (Core Swing) | 3 | `2 Entry / 1 Exit` | `0.1892 / 4.2593` | `2.36 -> 56.70` | $+584.22 | `9->4 wins (Net +1.64R -> +4.56R)` | **PROMOTED** |

## Autonomous 4-Vector Loss-Autopsy Attribution (`RUN_B_ONLINE_RL`)

| Bucket | Friction Inversion | Holding / EOD Decay | Vehicle Regret | Paralysis Regret | Flat Windows Replayed | Cost Sens (Before -> After) | Promoted Gens |
|---|---:|---:|---:|---:|---:|---|---:|
| SCALP_0DTE (1m) | 3 | 2 | 0 | 64 | 1119 | `0.828 -> 0.828` | 1 |
| SCALP_0DTE (5m) | 52 | 22 | 0 | 0 | 736 | `0.904 -> 1.252` | 4 |
| BRAIN-01 (Penny Day) | 34 | 20 | 0 | 64 | 1099 | `1.155 -> 1.459` | 4 |
| BRAIN-02 (Large-Cap Day) | 21 | 18 | 297 | 320 | 2644 | `0.958 -> 0.478` | 4 |
| BRAIN-03 (Short Swing) | 17 | 13 | 3 | 0 | 704 | `0.592 -> 0.681` | 3 |
| BRAIN-04 (Core Swing) | 26 | 38 | 3 | 0 | 672 | `0.632 -> 0.625` | 4 |
