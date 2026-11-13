# ZERO-CHEAT END-TO-END DIFFERENTIABLE BPTT + CLOSED-LOOP LOSS-AUTOPSY RL PROOF REPORT

- **Generated At (UTC):** `2026-11-13T23:08:35.949769+00:00`
- **Dataset 1 (`1m` Intraday + Causal `5m` Session Pre-Training):** `21` total `1m` trading days (`11` OOS days), `167411` real `1m` OHLCV bars loaded
- **Dataset 2 (`5m` & `1h` 60-Ticker Universe):** `59` real US symbols across `65` total trading days (`24` Warmup Calibration -> `41` Strictly Out-of-Sample)
- **Zero-Rule Neural Architecture (`src/pure_rl_brain.rs`):** Stage A Multi-Horizon Physics World Model (`1, 3, 8, 24`-bar Pinball Quantile Loss + `ln(Range/RT_Cost)`) + Stage B `K=5` Orthogonal Specialist Policy Ensemble (`D=64`, `[h_ln, wm_fused, self_state, pos_interact]` 92-d skip-head) with Localized Boundary Surrogate Deadzone (`sech^2((|z| - tau) / temp)`), Unconstrained Differentiable Log-Space Cost Sensitivity (`k_theta = exp(log_k_theta)` via analytical `dL_net / d log_k_theta`), Turnover-Amortized Opportunity Critic (`Q_hold(x_t, s_t) - Q_flat(x_t, s_t)`), Purged `T_holdout` Cross-Sectional LCB Promotion Gate, and `M=24` `rayon` PBT + Multiplicative-Weights Routing.
- **Anti-Cheat Execution Physics:** Zero `> t` lookahead, next-bar `t+1` open fills with `5/12/35 bps` adverse equity slippage, `09:35–15:15 ET` flatten `15:50 ET` (`SCALP_0DTE`), `09:35–15:30 ET` flatten `15:55 ET` (`DAY`), skew-adjusted BSM + `^VIX9D`/`^VIX` IV + `$0.03–$0.08`/sh/leg spread + `$0.65`/contract/leg RT commission.
- **BPTT & Closed-Loop Loss-Autopsy Activity:** `249577` Causal Warmup Episode Bars | `293212` OOS Causal Episode Bars | `114` Online EOD RL & Routing Cycles (`RUN_B_ONLINE_RL`) | `255652` Autonomous `FLAT` Abstentions.

## Causal Walk-Forward Inner PBT Calibration (`t <= T_oos_start`)

| Brain / Horizon | Population | Session Episodes | PBT Gens | Best Inner-Val Trades | Best Inner-Val Net PF | Best Inner-Val Net PnL ($) |
|---|---:|---:|---:|---:|---:|---:|
| SCALP_0DTE (1m) | 24 | 150 | 3 | 2 | **0.00** | $-214.55 |
| SCALP_0DTE (5m) | 24 | 192 | 3 | 4 | **6.71** | $+191.60 |
| BRAIN-01 (Penny Day) | 24 | 379 | 3 | 16 | **1.18** | $+140.89 |
| BRAIN-02 (Large-Cap Day) | 24 | 818 | 3 | 13 | **4.07** | $+357.37 |
| BRAIN-03 (Short Swing) | 24 | 411 | 3 | 3 | **290.81** | $+567.58 |
| BRAIN-04 (Core Swing) | 24 | 411 | 3 | 19 | **1.24** | $+500.93 |

## Executive Scorecard: `RUN_A_FROZEN_WARMUP_ML` vs `RUN_B_ONLINE_RL` (Gross vs Net Post-Cost)

| Bucket | Run Mode | Trades | Win% | Gross PF | Net Post-Cost PF | Net Exp (R) | Net Exp ($) | Gross PnL ($) | Cost Drag ($) | Net PnL ($) | Max DD ($ / %) | Backstop% | Neural-Exit% | EOD Flatten% |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SCALP_0DTE (1m) | `RUN_A_FROZEN_WARMUP_ML` | 16 | 50.0% | 1.08 | **0.74** | -0.046R | $-13.82 | $+59.16 | $280.21 | **$-221.05** | $534.18 (0.53%) | 0.0% | 100.0% | 0.0% |
| SCALP_0DTE (1m) | `RUN_B_ONLINE_RL` | 34 | 52.9% | 1.32 | **0.90** | -0.015R | $-4.08 | $+393.32 | $532.09 | **$-138.76** | $1043.51 (1.04%) | 2.9% | 97.1% | 0.0% |
| SCALP_0DTE (5m) | `RUN_A_FROZEN_WARMUP_ML` | 5 | 20.0% | 1.71 | **0.99** | -0.010R | $-0.24 | $+65.12 | $66.31 | **$-1.19** | $143.22 (0.14%) | 0.0% | 100.0% | 0.0% |
| SCALP_0DTE (5m) | `RUN_B_ONLINE_RL` | 17 | 41.2% | 1.67 | **1.16** | +0.016R | $+4.63 | $+256.50 | $177.84 | **$+78.66** | $335.27 (0.34%) | 0.0% | 94.1% | 5.9% |
| BRAIN-01 (Penny Day) | `RUN_A_FROZEN_WARMUP_ML` | 9 | 22.2% | 0.74 | **0.48** | -0.112R | $-65.43 | $-219.99 | $368.87 | **$-588.87** | $1064.59 (1.06%) | 11.1% | 77.8% | 11.1% |
| BRAIN-01 (Penny Day) | `RUN_B_ONLINE_RL` | 9 | 22.2% | 0.74 | **0.48** | -0.112R | $-66.40 | $-222.56 | $375.06 | **$-597.63** | $1083.22 (1.08%) | 11.1% | 77.8% | 11.1% |
| BRAIN-02 (Large-Cap Day) | `RUN_A_FROZEN_WARMUP_ML` | 58 | 32.8% | 0.48 | **0.38** | -0.124R | $-51.29 | $-2270.49 | $704.14 | **$-2974.63** | $3419.75 (3.41%) | 8.6% | 89.7% | 1.7% |
| BRAIN-02 (Large-Cap Day) | `RUN_B_ONLINE_RL` | 38 | 31.6% | 0.92 | **0.71** | -0.049R | $-16.17 | $-144.27 | $470.18 | **$-614.45** | $1211.00 (1.21%) | 5.3% | 81.6% | 13.2% |
| BRAIN-03 (Short Swing) | `RUN_A_FROZEN_WARMUP_ML` | 21 | 33.3% | 0.39 | **0.32** | -0.159R | $-67.58 | $-1195.00 | $224.18 | **$-1419.18** | $1968.35 (1.96%) | 4.8% | 95.2% | 0.0% |
| BRAIN-03 (Short Swing) | `RUN_B_ONLINE_RL` | 57 | 45.6% | 1.02 | **0.87** | -0.013R | $-13.11 | $+125.19 | $872.20 | **$-747.01** | $2389.30 (2.35%) | 5.3% | 89.5% | 0.0% |
| BRAIN-04 (Core Swing) | `RUN_A_FROZEN_WARMUP_ML` | 15 | 40.0% | 0.49 | **0.42** | -0.307R | $-140.43 | $-1773.74 | $332.67 | **$-2106.41** | $3047.27 (3.05%) | 33.3% | 66.7% | 0.0% |
| BRAIN-04 (Core Swing) | `RUN_B_ONLINE_RL` | 33 | 48.5% | 0.78 | **0.66** | -0.065R | $-39.98 | $-794.25 | $525.25 | **$-1319.50** | $2527.28 (2.53%) | 12.1% | 81.8% | 0.0% |
| ALL_BUCKETS_COMBINED | `RUN_A_FROZEN_WARMUP_ML` | 124 | 34.7% | 0.53 | **0.42** | -0.136R | $-58.96 | $-5334.95 | $1976.38 | **$-7311.33** | $8415.87 (8.40%) | 9.7% | 88.7% | 1.6% |
| ALL_BUCKETS_COMBINED | `RUN_B_ONLINE_RL` | 188 | 43.1% | 0.97 | **0.77** | -0.032R | $-17.76 | $-386.08 | $2952.62 | **$-3338.70** | $4862.92 (4.85%) | 5.9% | 87.8% | 3.7% |

## Horizon Tier Rollups (`RUN_B_ONLINE_RL`)

| Tier | Run Mode | Trades | Win% | Gross PF | Net Post-Cost PF | Net Exp (R) | Net Exp ($) | Gross PnL ($) | Cost Drag ($) | Net PnL ($) | Max DD ($ / %) | Backstop% | Neural-Exit% | EOD Flatten% |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TIER: SCALP_0DTE | `RUN_B_ONLINE_RL` | 51 | 49.0% | 1.41 | **0.97** | -0.005R | $-1.18 | $+649.83 | $709.93 | **$-60.11** | $1043.51 (1.04%) | 2.0% | 96.1% | 2.0% |
| TIER: DAY (BRAIN-01 + BRAIN-02) | `RUN_B_ONLINE_RL` | 47 | 29.8% | 0.86 | **0.63** | -0.061R | $-25.79 | $-366.84 | $845.24 | **$-1212.08** | $1756.40 (1.75%) | 6.4% | 80.9% | 12.8% |
| TIER: SWING (BRAIN-03 + BRAIN-04) | `RUN_B_ONLINE_RL` | 90 | 46.7% | 0.93 | **0.79** | -0.032R | $-22.96 | $-669.07 | $1397.45 | **$-2066.51** | $4862.92 (4.79%) | 7.8% | 86.7% | 0.0% |

## Factored Vehicle Performance Split (`V_SHARES` vs `V_OPTION`)

| Run Mode | Vehicle | Trades | Win% | Gross PF | Net Post-Cost PF | Net Exp (R) | Avg Cost/Trade ($) | Total Gross PnL ($) | Total Cost Drag ($) | Total Net PnL ($) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `RUN_A_FROZEN_WARMUP_ML` | **V_SHARES** | 124 | 34.7% | 0.53 | **0.42** | -0.136R | $15.94 | $-5334.95 | $1976.38 | **$-7311.33** |
| `RUN_A_FROZEN_WARMUP_ML` | **V_OPTION** | 0 | 0.0% | 0.00 | **0.00** | +0.000R | $0.00 | $+0.00 | $0.00 | **$+0.00** |
| `RUN_B_ONLINE_RL` | **V_SHARES** | 188 | 43.1% | 0.97 | **0.77** | -0.032R | $15.71 | $-386.08 | $2952.62 | **$-3338.70** |
| `RUN_B_ONLINE_RL` | **V_OPTION** | 0 | 0.0% | 0.00 | **0.00** | +0.000R | $0.00 | $+0.00 | $0.00 | **$+0.00** |

## A/B Online Learning Delta (`RUN_B_ONLINE_RL` minus `RUN_A_FROZEN_WARMUP_ML`)

| Bucket | Δ Trades | Δ Win% | Δ Net PF | Δ Net Exp (R) | Δ Net PnL ($) | Δ Max DD ($) |
|---|---:|---:|---:|---:|---:|---:|
| SCALP_0DTE (1m) | +18 | +2.9% | +0.17 | +0.031R | **$+82.28** | $+509.32 |
| SCALP_0DTE (5m) | +12 | +21.2% | +0.17 | +0.026R | **$+79.85** | $+192.05 |
| BRAIN-01 (Penny Day) | +0 | +0.0% | +0.00 | -0.000R | **$-8.76** | $+18.62 |
| BRAIN-02 (Large-Cap Day) | -20 | -1.2% | +0.33 | +0.074R | **$+2360.18** | $-2208.75 |
| BRAIN-03 (Short Swing) | +36 | +12.3% | +0.55 | +0.146R | **$+672.17** | $+420.95 |
| BRAIN-04 (Core Swing) | +18 | +8.5% | +0.24 | +0.242R | **$+786.91** | $-519.99 |
| **DATASET_1_1M_TOTAL** | — | — | — | — | **$+82.28** | — |
| **DATASET_2_5M_1H_TOTAL** | — | — | — | — | **$+3890.35** | — |
| **ALL_BUCKETS_COMBINED** | **+64** | **+8.4%** | **+0.35** | **+0.104R** | **$+3972.63** | **$-3552.95** |

## Loss-Autopsy -> Re-Test -> Generation Promotion Ledger

| Gen ID | Phase | Brain | Red Trades Audited | Entry vs Exit Fault Split | Saliency (`|dL/dx|` / `|dL/dh|`) | Re-Test Net PF (Before -> After) | Red Loss Reduced ($) | Winners Preserved | Outcome |
|---:|---|---|---:|---|---|---|---:|---|---|
| 1 | `WARMUP_CAL` | SCALP_0DTE (1m) | 3 | `3 Entry / 0 Exit` | `0.3562 / 3.1422` | `5.76 -> 5.76` | $+7.98 | `5->5 wins (+3.33R -> +3.27R)` | `ROLLED_BACK` |
| 2 | `WARMUP_CAL` | SCALP_0DTE (5m) | 1 | `0 Entry / 1 Exit` | `0.0277 / 0.5895` | `45.85 -> 41.68` | $-3.61 | `9->9 wins (Net +3.76R -> +3.72R)` | **PROMOTED** |
| 1 | `WARMUP_CAL` | BRAIN-01 (Penny Day) | 0 | `0 Entry / 0 Exit` | `0.0827 / 0.9088` | `0.00 -> 3.82` | $-1808.99 | `0->23 wins (+0.00R -> +12.25R)` | `ROLLED_BACK` |
| 2 | `WARMUP_CAL` | BRAIN-02 (Large-Cap Day) | 2 | `2 Entry / 0 Exit` | `0.1035 / 0.6319` | `1.99 -> 6.18` | $-281.61 | `2->22 wins (Net +0.13R -> +4.34R)` | **PROMOTED** |
| 2 | `WARMUP_CAL` | BRAIN-03 (Short Swing) | 7 | `6 Entry / 1 Exit` | `0.1406 / 1.0433` | `6.98 -> 302.21` | $+939.95 | `21->29 wins (Net +12.25R -> +24.13R)` | **PROMOTED** |
| 1 | `OOS_D1` | SCALP_0DTE (1m) | 3 | `3 Entry / 0 Exit` | `0.3562 / 3.1422` | `5.76 -> 5.76` | $+7.70 | `5->5 wins (+3.33R -> +3.27R)` | `ROLLED_BACK` |
| 1 | `OOS_D2` | SCALP_0DTE (1m) | 3 | `3 Entry / 0 Exit` | `0.3562 / 3.1422` | `2.45 -> 2.40` | $+3.33 | `2->2 wins (+1.30R -> +1.30R)` | `ROLLED_BACK` |
| 2 | `OOS_D3` | SCALP_0DTE (1m) | 3 | `3 Entry / 0 Exit` | `0.3562 / 3.1422` | `2.45 -> 5.78` | $-4.92 | `2->10 wins (Net +0.67R -> +3.12R)` | **PROMOTED** |
| 3 | `OOS_D4` | SCALP_0DTE (1m) | 11 | `6 Entry / 5 Exit` | `0.0933 / 1.0722` | `1.70 -> 2.59` | $+546.96 | `13->5 wins (Net +1.86R -> +1.44R)` | **PROMOTED** |
| 3 | `OOS_D5` | SCALP_0DTE (1m) | 10 | `8 Entry / 2 Exit` | `0.0616 / 0.6816` | `1.29 -> 4.67` | $+546.24 | `6->6 wins (+2.57R -> +2.40R)` | `ROLLED_BACK` |
| 4 | `OOS_D6` | SCALP_0DTE (1m) | 11 | `9 Entry / 2 Exit` | `0.0573 / 0.6278` | `1.91 -> 10.22` | $+661.20 | `13->12 wins (Net +1.98R -> +3.38R)` | **PROMOTED** |
| 5 | `OOS_D7` | SCALP_0DTE (1m) | 8 | `7 Entry / 1 Exit` | `0.0270 / 0.2778` | `4.98 -> 17.58` | $+206.35 | `16->15 wins (Net +2.69R -> +3.27R)` | **PROMOTED** |
| 5 | `OOS_D8` | SCALP_0DTE (1m) | 9 | `5 Entry / 4 Exit` | `0.0315 / 0.3686` | `5.92 -> 13.03` | $+167.88 | `16->10 wins (+3.58R -> +2.58R)` | `ROLLED_BACK` |
| 5 | `OOS_D9` | SCALP_0DTE (1m) | 8 | `4 Entry / 4 Exit` | `0.0199 / 0.2281` | `6.38 -> 5.45` | $+22.26 | `16->12 wins (+3.58R -> +2.81R)` | `ROLLED_BACK` |
| 5 | `OOS_D10` | SCALP_0DTE (1m) | 8 | `4 Entry / 4 Exit` | `0.0199 / 0.2281` | `4.97 -> 4.98` | $+6.71 | `14->11 wins (+2.66R -> +2.73R)` | `ROLLED_BACK` |
| 2 | `OOS_D1` | SCALP_0DTE (5m) | 2 | `0 Entry / 2 Exit` | `0.0220 / 0.5165` | `8.10 -> 5.94` | $-34.79 | `5->5 wins (+1.41R -> +1.50R)` | `ROLLED_BACK` |
| 1 | `OOS_D1` | BRAIN-01 (Penny Day) | 0 | `0 Entry / 0 Exit` | `0.0250 / 0.3814` | `0.00 -> 0.29` | $-315.87 | `0->3 wins (+0.00R -> +0.24R)` | `ROLLED_BACK` |
| 2 | `OOS_D1` | BRAIN-02 (Large-Cap Day) | 6 | `5 Entry / 1 Exit` | `0.0472 / 0.3771` | `10.46 -> 2.35` | $-110.94 | `18->9 wins (+3.97R -> +1.68R)` | `ROLLED_BACK` |
| 2 | `OOS_D3` | SCALP_0DTE (5m) | 5 | `1 Entry / 4 Exit` | `0.0227 / 0.7181` | `2.08 -> 1.92` | $+40.18 | `4->2 wins (+0.95R -> +0.66R)` | `ROLLED_BACK` |
| 1 | `OOS_D3` | BRAIN-01 (Penny Day) | 0 | `0 Entry / 0 Exit` | `0.0735 / 0.6328` | `0.00 -> 12.00` | $+0.00 | `0->1 wins (+0.00R -> +0.08R)` | `ROLLED_BACK` |
| 2 | `OOS_D4` | BRAIN-02 (Large-Cap Day) | 5 | `5 Entry / 0 Exit` | `0.0546 / 0.4210` | `8.40 -> 4.03` | $+81.82 | `14->6 wins (+3.01R -> +0.63R)` | `ROLLED_BACK` |
| 1 | `OOS_D5` | BRAIN-01 (Penny Day) | 0 | `0 Entry / 0 Exit` | `0.0428 / 0.6494` | `0.00 -> 0.00` | $+0.00 | `0->0 wins (+0.00R -> +0.00R)` | `ROLLED_BACK` |
| 2 | `OOS_D6` | SCALP_0DTE (5m) | 5 | `1 Entry / 4 Exit` | `0.0227 / 0.7181` | `2.08 -> 1.23` | $+4.39 | `4->3 wins (+0.95R -> +0.62R)` | `ROLLED_BACK` |
| 2 | `OOS_D6` | BRAIN-02 (Large-Cap Day) | 5 | `5 Entry / 0 Exit` | `0.0546 / 0.4210` | `7.62 -> 12.00` | $+138.93 | `13->6 wins (+2.76R -> +0.51R)` | `ROLLED_BACK` |
| 1 | `OOS_D7` | BRAIN-01 (Penny Day) | 0 | `0 Entry / 0 Exit` | `0.0606 / 0.6155` | `0.00 -> 0.00` | $+0.00 | `0->0 wins (+0.00R -> +0.00R)` | `ROLLED_BACK` |
| 2 | `OOS_D7` | BRAIN-02 (Large-Cap Day) | 8 | `7 Entry / 1 Exit` | `0.0468 / 0.4346` | `2.82 -> 12.00` | $+433.57 | `14->5 wins (+3.22R -> +0.48R)` | `ROLLED_BACK` |
| 2 | `OOS_D9` | SCALP_0DTE (5m) | 4 | `1 Entry / 3 Exit` | `0.0214 / 0.7502` | `1.98 -> 12.00` | $+143.22 | `3->1 wins (+0.71R -> +0.04R)` | `ROLLED_BACK` |
| 1 | `OOS_D9` | BRAIN-01 (Penny Day) | 0 | `0 Entry / 0 Exit` | `0.1032 / 0.8285` | `0.00 -> 0.68` | $-251.71 | `0->4 wins (+0.00R -> +0.32R)` | `ROLLED_BACK` |
| 2 | `OOS_D10` | BRAIN-02 (Large-Cap Day) | 6 | `5 Entry / 1 Exit` | `0.0578 / 0.5127` | `1.79 -> 12.00` | $+367.64 | `11->2 wins (+1.75R -> +0.29R)` | `ROLLED_BACK` |
| 1 | `OOS_D11` | BRAIN-01 (Penny Day) | 0 | `0 Entry / 0 Exit` | `0.0958 / 0.9328` | `0.00 -> 0.00` | $+0.00 | `0->0 wins (+0.00R -> +0.00R)` | `ROLLED_BACK` |
| 2 | `OOS_D12` | SCALP_0DTE (5m) | 4 | `1 Entry / 3 Exit` | `0.0214 / 0.7502` | `0.99 -> 0.90` | $+1.01 | `1->2 wins (+0.30R -> +0.26R)` | `ROLLED_BACK` |
| 1 | `OOS_D13` | BRAIN-01 (Penny Day) | 0 | `0 Entry / 0 Exit` | `0.1351 / 1.2575` | `0.00 -> 12.00` | $+0.00 | `0->1 wins (+0.00R -> +0.08R)` | `ROLLED_BACK` |
| 2 | `OOS_D13` | BRAIN-02 (Large-Cap Day) | 6 | `5 Entry / 1 Exit` | `0.0578 / 0.5127` | `0.89 -> 1.32` | $+171.61 | `6->4 wins (+0.91R -> +0.68R)` | `ROLLED_BACK` |
| 2 | `OOS_D14` | BRAIN-02 (Large-Cap Day) | 6 | `5 Entry / 1 Exit` | `0.0269 / 0.3400` | `0.49 -> 0.00` | $+371.21 | `2->0 wins (+0.59R -> +0.00R)` | `ROLLED_BACK` |
| 2 | `OOS_D15` | SCALP_0DTE (5m) | 4 | `1 Entry / 3 Exit` | `0.0214 / 0.7502` | `0.99 -> 0.92` | $+23.74 | `1->1 wins (+0.30R -> +0.34R)` | `ROLLED_BACK` |
| 1 | `OOS_D15` | BRAIN-01 (Penny Day) | 0 | `0 Entry / 0 Exit` | `0.0740 / 0.9321` | `0.00 -> 0.17` | $-398.58 | `0->3 wins (+0.00R -> +0.16R)` | `ROLLED_BACK` |
| 2 | `OOS_D17` | SCALP_0DTE (5m) | 3 | `1 Entry / 2 Exit` | `0.0232 / 0.8525` | `1.24 -> 1.15` | $+12.77 | `1->1 wins (+0.30R -> +0.29R)` | `ROLLED_BACK` |
| 1 | `OOS_D17` | BRAIN-01 (Penny Day) | 0 | `0 Entry / 0 Exit` | `0.0680 / 0.5392` | `0.00 -> 0.00` | $+0.00 | `0->0 wins (+0.00R -> +0.00R)` | `ROLLED_BACK` |
| 2 | `OOS_D17` | BRAIN-02 (Large-Cap Day) | 5 | `4 Entry / 1 Exit` | `0.0254 / 0.3254` | `0.43 -> 12.00` | $+380.99 | `1->1 wins (+0.46R -> +0.18R)` | `ROLLED_BACK` |
| 2 | `OOS_D19` | SCALP_0DTE (5m) | 0 | `0 Entry / 0 Exit` | `0.0878 / 2.6356` | `0.00 -> 0.07` | $-772.99 | `0->2 wins (+0.00R -> +0.14R)` | `ROLLED_BACK` |
| 1 | `OOS_D19` | BRAIN-01 (Penny Day) | 0 | `0 Entry / 0 Exit` | `0.0788 / 0.7500` | `0.00 -> 0.00` | $+0.00 | `0->0 wins (+0.00R -> +0.00R)` | `ROLLED_BACK` |
| 2 | `OOS_D20` | BRAIN-02 (Large-Cap Day) | 6 | `4 Entry / 2 Exit` | `0.0218 / 0.2792` | `1.28 -> 3.10` | $+371.48 | `3->1 wins (+1.37R -> +0.33R)` | `ROLLED_BACK` |
| 2 | `OOS_D21` | SCALP_0DTE (5m) | 0 | `0 Entry / 0 Exit` | `0.1032 / 2.9514` | `0.00 -> 0.00` | $-98.75 | `0->0 wins (+0.00R -> +0.00R)` | `ROLLED_BACK` |
| 1 | `OOS_D21` | BRAIN-01 (Penny Day) | 0 | `0 Entry / 0 Exit` | `0.0974 / 1.3271` | `0.00 -> 0.00` | $+0.00 | `0->0 wins (+0.00R -> +0.00R)` | `ROLLED_BACK` |
| 2 | `OOS_D22` | BRAIN-02 (Large-Cap Day) | 4 | `3 Entry / 1 Exit` | `0.0164 / 0.2101` | `1.43 -> 1.01` | $+139.48 | `3->1 wins (+1.01R -> +0.33R)` | `ROLLED_BACK` |
| 2 | `OOS_D23` | SCALP_0DTE (5m) | 0 | `0 Entry / 0 Exit` | `0.0606 / 2.2768` | `0.00 -> 1.86` | $-17.75 | `0->1 wins (+0.00R -> +0.08R)` | `ROLLED_BACK` |
| 2 | `OOS_D23` | BRAIN-02 (Large-Cap Day) | 5 | `4 Entry / 1 Exit` | `0.0151 / 0.1904` | `2.54 -> 5.14` | $+291.24 | `5->2 wins (+2.02R -> +0.32R)` | `ROLLED_BACK` |
| 3 | `OOS_D24` | BRAIN-02 (Large-Cap Day) | 8 | `4 Entry / 4 Exit` | `0.0114 / 0.1394` | `2.07 -> 8.46` | $+232.03 | `5->5 wins (Net +1.02R -> +3.19R)` | **PROMOTED** |
| 2 | `OOS_D25` | SCALP_0DTE (5m) | 0 | `0 Entry / 0 Exit` | `0.0608 / 1.6053` | `0.00 -> 0.16` | $-181.78 | `0->1 wins (+0.00R -> +0.08R)` | `ROLLED_BACK` |
| 3 | `OOS_D25` | BRAIN-02 (Large-Cap Day) | 9 | `4 Entry / 5 Exit` | `0.0175 / 0.1648` | `1.87 -> 0.57` | $+238.29 | `5->3 wins (+3.61R -> +0.71R)` | `ROLLED_BACK` |
| 2 | `OOS_D27` | SCALP_0DTE (5m) | 0 | `0 Entry / 0 Exit` | `0.0590 / 2.4712` | `0.00 -> 12.00` | $+0.00 | `0->1 wins (+0.00R -> +0.01R)` | `ROLLED_BACK` |
| 4 | `OOS_D31` | BRAIN-02 (Large-Cap Day) | 16 | `5 Entry / 11 Exit` | `0.0174 / 0.2155` | `1.20 -> 53.38` | $+1311.40 | `9->1 wins (Net +0.66R -> +0.04R)` | **PROMOTED** |
| 3 | `OOS_D33` | SCALP_0DTE (5m) | 0 | `0 Entry / 0 Exit` | `0.1186 / 3.3060` | `0.00 -> 9.59` | $-22.94 | `0->3 wins (Net +0.00R -> +0.56R)` | **PROMOTED** |
| 5 | `OOS_D35` | BRAIN-02 (Large-Cap Day) | 5 | `2 Entry / 3 Exit` | `0.0223 / 0.2929` | `0.99 -> 1.34` | $+59.84 | `2->1 wins (Net +0.03R -> +0.14R)` | **PROMOTED** |
| 6 | `OOS_D38` | BRAIN-02 (Large-Cap Day) | 4 | `2 Entry / 2 Exit` | `0.0220 / 0.2685` | `1.34 -> 2.41` | $+69.98 | `1->2 wins (Net +0.14R -> +0.36R)` | **PROMOTED** |
| 3 | `OOS_D1` | BRAIN-03 (Short Swing) | 2 | `1 Entry / 1 Exit` | `0.1629 / 1.4695` | `2.51 -> 1.87` | $-1130.20 | `1->19 wins (Net +0.38R -> +1.34R)` | **PROMOTED** |
| 3 | `OOS_D4` | BRAIN-03 (Short Swing) | 18 | `10 Entry / 8 Exit` | `0.0799 / 1.0072` | `1.29 -> 1.21` | $+97.99 | `17->16 wins (+3.25R -> +2.88R)` | `ROLLED_BACK` |
| 3 | `OOS_D5` | BRAIN-03 (Short Swing) | 15 | `9 Entry / 6 Exit` | `0.1456 / 1.4745` | `0.95 -> 0.00` | $+1852.29 | `13->0 wins (+3.76R -> +0.00R)` | `ROLLED_BACK` |
| 3 | `OOS_D8` | BRAIN-03 (Short Swing) | 15 | `8 Entry / 7 Exit` | `0.1938 / 1.8161` | `1.77 -> 1.08` | $+626.02 | `13->6 wins (+4.94R -> +1.90R)` | `ROLLED_BACK` |
| 4 | `OOS_D9` | BRAIN-03 (Short Swing) | 13 | `11 Entry / 2 Exit` | `0.0815 / 0.7458` | `1.29 -> 25.16` | $+1035.08 | `10->10 wins (Net +0.92R -> +14.97R)` | **PROMOTED** |
| 4 | `OOS_D10` | BRAIN-03 (Short Swing) | 8 | `1 Entry / 7 Exit` | `0.8530 / 8.9221` | `0.79 -> 4.24` | $+254.28 | `6->7 wins (+4.74R -> +11.06R)` | `ROLLED_BACK` |
| 4 | `OOS_D13` | BRAIN-03 (Short Swing) | 7 | `5 Entry / 2 Exit` | `0.2866 / 2.8672` | `2.00 -> 2.16` | $+424.92 | `7->7 wins (+6.26R -> +4.90R)` | `ROLLED_BACK` |
| 4 | `OOS_D15` | BRAIN-03 (Short Swing) | 5 | `3 Entry / 2 Exit` | `0.2224 / 2.5940` | `2.11 -> 5.22` | $+1538.55 | `10->6 wins (+9.79R -> +5.26R)` | `ROLLED_BACK` |
| 4 | `OOS_D17` | BRAIN-03 (Short Swing) | 7 | `6 Entry / 1 Exit` | `0.1816 / 2.1861` | `0.92 -> 12.00` | $+2540.47 | `7->1 wins (+4.10R -> +0.08R)` | `ROLLED_BACK` |
| 4 | `OOS_D19` | BRAIN-03 (Short Swing) | 9 | `7 Entry / 2 Exit` | `0.2416 / 2.9874` | `0.61 -> 12.00` | $+3511.28 | `6->1 wins (+3.76R -> +0.08R)` | `ROLLED_BACK` |
| 4 | `OOS_D22` | BRAIN-03 (Short Swing) | 9 | `8 Entry / 1 Exit` | `0.3890 / 4.4980` | `0.18 -> 0.21` | $-412.98 | `4->4 wins (+1.17R -> +1.31R)` | `ROLLED_BACK` |
| 4 | `OOS_D24` | BRAIN-03 (Short Swing) | 11 | `9 Entry / 2 Exit` | `0.3240 / 3.5996` | `0.49 -> 12.00` | $+2787.36 | `5->1 wins (+2.19R -> +0.95R)` | `ROLLED_BACK` |
| 5 | `OOS_D27` | BRAIN-03 (Short Swing) | 8 | `6 Entry / 2 Exit` | `0.2345 / 3.1707` | `0.38 -> 1.61` | $+605.49 | `6->18 wins (Net -1.75R -> +1.50R)` | **PROMOTED** |
| 5 | `OOS_D30` | BRAIN-03 (Short Swing) | 19 | `14 Entry / 5 Exit` | `0.0733 / 0.5539` | `1.56 -> 1.54` | $-11.20 | `17->17 wins (+6.86R -> +6.85R)` | `ROLLED_BACK` |
| 5 | `OOS_D31` | BRAIN-03 (Short Swing) | 16 | `11 Entry / 5 Exit` | `0.0790 / 0.7789` | `1.25 -> 0.39` | $+1118.58 | `19->7 wins (+5.34R -> +0.54R)` | `ROLLED_BACK` |
| 6 | `OOS_D35` | BRAIN-03 (Short Swing) | 18 | `13 Entry / 5 Exit` | `0.1049 / 0.9788` | `1.17 -> 1.49` | $+22.72 | `22->22 wins (Net +0.99R -> +2.58R)` | **PROMOTED** |
| 7 | `OOS_D39` | BRAIN-03 (Short Swing) | 23 | `11 Entry / 12 Exit` | `0.0591 / 0.7294` | `1.08 -> 2.03` | $-1051.96 | `18->17 wins (Net +1.85R -> +9.25R)` | **PROMOTED** |
| 8 | `OOS_D40` | BRAIN-03 (Short Swing) | 14 | `6 Entry / 8 Exit` | `0.2368 / 3.3350` | `2.33 -> 7.47` | $+2677.97 | `14->18 wins (Net +12.16R -> +12.41R)` | **PROMOTED** |
| 1 | `OOS_D1` | BRAIN-04 (Core Swing) | 8 | `2 Entry / 6 Exit` | `0.2576 / 3.9017` | `2.10 -> 3.06` | $+878.68 | `8->15 wins (+7.45R -> +3.80R)` | `ROLLED_BACK` |
| 1 | `OOS_D3` | BRAIN-04 (Core Swing) | 11 | `2 Entry / 9 Exit` | `0.3346 / 4.3058` | `2.87 -> 4.92` | $+558.08 | `7->9 wins (+7.22R -> +13.05R)` | `ROLLED_BACK` |
| 1 | `OOS_D5` | BRAIN-04 (Core Swing) | 11 | `3 Entry / 8 Exit` | `0.3043 / 3.5389` | `3.53 -> 3.52` | $+401.44 | `7->8 wins (+7.33R -> +8.15R)` | `ROLLED_BACK` |
| 2 | `OOS_D7` | BRAIN-04 (Core Swing) | 6 | `4 Entry / 2 Exit` | `0.1313 / 1.7927` | `1.74 -> 4.98` | $+241.29 | `8->8 wins (Net +2.84R -> +7.42R)` | **PROMOTED** |
| 3 | `OOS_D8` | BRAIN-04 (Core Swing) | 8 | `1 Entry / 7 Exit` | `0.5273 / 5.5205` | `4.82 -> 4.98` | $-88.40 | `9->9 wins (Net +7.38R -> +7.28R)` | **PROMOTED** |
| 4 | `OOS_D11` | BRAIN-04 (Core Swing) | 9 | `2 Entry / 7 Exit` | `0.5282 / 5.0416` | `1.75 -> 1.77` | $+24.19 | `12->12 wins (Net +3.63R -> +3.70R)` | **PROMOTED** |
| 4 | `OOS_D14` | BRAIN-04 (Core Swing) | 9 | `3 Entry / 6 Exit` | `0.3899 / 3.2836` | `2.84 -> 1.40` | $-2580.52 | `13->9 wins (+5.98R -> +5.92R)` | `ROLLED_BACK` |
| 4 | `OOS_D16` | BRAIN-04 (Core Swing) | 8 | `3 Entry / 5 Exit` | `0.3730 / 2.9521` | `2.07 -> 3.40` | $+1053.28 | `10->7 wins (+4.58R -> +1.27R)` | `ROLLED_BACK` |
| 4 | `OOS_D17` | BRAIN-04 (Core Swing) | 7 | `3 Entry / 4 Exit` | `0.4037 / 3.0676` | `2.28 -> 1.95` | $-335.77 | `12->10 wins (+5.51R -> +3.97R)` | `ROLLED_BACK` |
| 4 | `OOS_D20` | BRAIN-04 (Core Swing) | 10 | `3 Entry / 7 Exit` | `0.4176 / 3.5482` | `1.10 -> 4.45` | $+2081.70 | `10->3 wins (+4.49R -> +0.48R)` | `ROLLED_BACK` |
| 4 | `OOS_D23` | BRAIN-04 (Core Swing) | 8 | `2 Entry / 6 Exit` | `0.3282 / 3.0518` | `2.01 -> 2.41` | $+803.85 | `12->16 wins (+8.14R -> +5.14R)` | `ROLLED_BACK` |
| 4 | `OOS_D24` | BRAIN-04 (Core Swing) | 9 | `3 Entry / 6 Exit` | `0.2364 / 2.4143` | `3.20 -> 3.04` | $-127.78 | `15->15 wins (+6.65R -> +6.71R)` | `ROLLED_BACK` |
| 4 | `OOS_D27` | BRAIN-04 (Core Swing) | 12 | `5 Entry / 7 Exit` | `0.1265 / 1.3217` | `0.64 -> 0.82` | $+331.18 | `15->14 wins (+1.88R -> +2.09R)` | `ROLLED_BACK` |
| 4 | `OOS_D30` | BRAIN-04 (Core Swing) | 15 | `6 Entry / 9 Exit` | `0.2024 / 2.0183` | `0.72 -> 46.77` | $+2043.97 | `14->4 wins (+3.35R -> +0.74R)` | `ROLLED_BACK` |
| 5 | `OOS_D33` | BRAIN-04 (Core Swing) | 14 | `8 Entry / 6 Exit` | `0.1610 / 1.7681` | `1.22 -> 3.69` | $-337.30 | `15->15 wins (Net +0.69R -> +7.89R)` | **PROMOTED** |
| 5 | `OOS_D35` | BRAIN-04 (Core Swing) | 14 | `2 Entry / 12 Exit` | `0.2380 / 3.2461` | `3.44 -> 2.62` | $-735.34 | `13->21 wins (+12.84R -> +11.15R)` | `ROLLED_BACK` |
| 6 | `OOS_D38` | BRAIN-04 (Core Swing) | 11 | `6 Entry / 5 Exit` | `0.0749 / 0.8275` | `3.49 -> 3.71` | $-393.81 | `12->12 wins (Net +7.38R -> +7.19R)` | **PROMOTED** |

## Autonomous 4-Vector Loss-Autopsy Attribution (`RUN_B_ONLINE_RL`)

| Bucket | Friction Inversion | Holding / EOD Decay | Vehicle Regret | Paralysis Regret | Flat Windows Replayed | Cost Sens (Before -> After) | Promoted Gens |
|---|---:|---:|---:|---:|---:|---|---:|
| SCALP_0DTE (1m) | 27 | 10 | 0 | 32 | 803 | `1.140 -> 1.443` | 5 |
| SCALP_0DTE (5m) | 7 | 5 | 0 | 288 | 2445 | `1.125 -> 1.261` | 3 |
| BRAIN-01 (Penny Day) | 6 | 2 | 0 | 672 | 2415 | `1.724 -> 1.724` | 1 |
| BRAIN-02 (Large-Cap Day) | 20 | 20 | 1 | 0 | 640 | `1.269 -> 2.070` | 6 |
| BRAIN-03 (Short Swing) | 68 | 59 | 19 | 32 | 955 | `0.864 -> 0.895` | 8 |
| BRAIN-04 (Core Swing) | 29 | 60 | 5 | 0 | 576 | `0.844 -> 0.938` | 6 |
