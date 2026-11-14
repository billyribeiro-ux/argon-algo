# ZERO-CHEAT END-TO-END DIFFERENTIABLE BPTT + CLOSED-LOOP LOSS-AUTOPSY RL PROOF REPORT

- **Generated At (UTC):** `2026-11-14T04:33:53.844768+00:00`
- **Dataset 1 (`1m` Intraday + Causal `5m` Session Pre-Training):** `21` total `1m` trading days (`11` OOS days), `167411` real `1m` OHLCV bars loaded
- **Dataset 2 (`5m` & `1h` 60-Ticker Universe):** `59` real US symbols across `65` total trading days (`24` Warmup Calibration -> `41` Strictly Out-of-Sample)
- **Zero-Rule Neural Architecture (`src/pure_rl_brain.rs`):** Stage A Multi-Horizon Physics World Model (`1, 3, 8, 24`-bar Pinball Quantile Loss + `ln(Range/RT_Cost)`) + Stage B `K=5` Orthogonal Specialist Policy Ensemble (`D=64`, `[h_ln, wm_fused, self_state, pos_interact]` 92-d skip-head) with Localized Boundary Surrogate Deadzone (`sech^2((|z| - tau) / temp)`), Unconstrained Differentiable Log-Space Cost Sensitivity (`k_theta = exp(log_k_theta)` via analytical `dL_net / d log_k_theta`), Turnover-Amortized Opportunity Critic (`Q_hold(x_t, s_t) - Q_flat(x_t, s_t)`), Purged `T_holdout` Cross-Sectional LCB Promotion Gate, and `M=24` `rayon` PBT + Multiplicative-Weights Routing.
- **Anti-Cheat Execution Physics:** Zero `> t` lookahead, next-bar `t+1` open fills with `5/12/35 bps` adverse equity slippage, `09:35–15:15 ET` flatten `15:50 ET` (`SCALP_0DTE`), `09:35–15:30 ET` flatten `15:55 ET` (`DAY`), skew-adjusted BSM + `^VIX9D`/`^VIX` IV + `$0.03–$0.08`/sh/leg spread + `$0.65`/contract/leg RT commission.
- **BPTT & Closed-Loop Loss-Autopsy Activity:** `193157` Causal Warmup Episode Bars | `270308` OOS Causal Episode Bars | `64` Online EOD RL & Routing Cycles (`RUN_B_ONLINE_RL`) | `371169` Autonomous `FLAT` Abstentions.

## Causal Walk-Forward Inner PBT Calibration (`t <= T_oos_start`)

| Brain / Horizon | Population | Session Episodes | PBT Gens | Best Inner-Val Trades | Best Inner-Val Net PF | Best Inner-Val Net PnL ($) |
|---|---:|---:|---:|---:|---:|---:|
| SCALP_0DTE (1m) | 24 | 150 | 3 | 5 | **2.84** | $+336.22 |
| SCALP_0DTE (5m) | 24 | 192 | 3 | 6 | **39.45** | $+142.27 |
| BRAIN-01 (Penny Day) | 24 | 379 | 3 | 28 | **0.67** | $-637.97 |
| BRAIN-02 (Large-Cap Day) | 24 | 818 | 3 | 10 | **3.68** | $+1119.69 |
| BRAIN-03 (Short Swing) | 24 | 59 | 4 | 30 | **6.39** | $+22950.97 |
| BRAIN-04 (Core Swing) | 24 | 59 | 4 | 26 | **5.01** | $+7942.24 |

## Executive Scorecard: `RUN_A_FROZEN_WARMUP_ML` vs `RUN_B_ONLINE_RL` (Gross vs Net Post-Cost)

| Bucket | Run Mode | Trades | Win% | Gross PF | Net Post-Cost PF | Net Exp (R) | Net Exp ($) | Gross PnL ($) | Cost Drag ($) | Net PnL ($) | Max DD ($ / %) | Backstop% | Neural-Exit% | EOD Flatten% |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SCALP_0DTE (1m) | `RUN_A_FROZEN_WARMUP_ML` | 35 | 34.3% | 0.62 | **0.45** | -0.124R | $-70.97 | $-1455.34 | $1028.54 | **$-2483.88** | $3000.04 (3.00%) | 0.0% | 100.0% | 0.0% |
| SCALP_0DTE (1m) | `RUN_B_ONLINE_RL` | 33 | 33.3% | 0.55 | **0.41** | -0.142R | $-81.55 | $-1732.25 | $958.74 | **$-2690.99** | $3000.04 (3.00%) | 0.0% | 100.0% | 0.0% |
| SCALP_0DTE (5m) | `RUN_A_FROZEN_WARMUP_ML` | 40 | 30.0% | 0.36 | **0.26** | -0.095R | $-47.93 | $-1467.77 | $449.47 | **$-1917.23** | $1956.83 (1.96%) | 0.0% | 100.0% | 0.0% |
| SCALP_0DTE (5m) | `RUN_B_ONLINE_RL` | 19 | 26.3% | 0.59 | **0.34** | -0.048R | $-24.93 | $-225.70 | $247.92 | **$-473.62** | $553.51 (0.55%) | 0.0% | 100.0% | 0.0% |
| BRAIN-01 (Penny Day) | `RUN_A_FROZEN_WARMUP_ML` | 189 | 31.7% | 1.14 | **0.44** | -0.078R | $-45.16 | $+1315.07 | $9850.00 | **$-8534.93** | $8891.18 (8.89%) | 0.5% | 94.2% | 5.3% |
| BRAIN-01 (Penny Day) | `RUN_B_ONLINE_RL` | 189 | 31.7% | 1.13 | **0.44** | -0.078R | $-45.78 | $+1300.60 | $9952.54 | **$-8651.94** | $9008.09 (9.01%) | 0.5% | 94.2% | 5.3% |
| BRAIN-02 (Large-Cap Day) | `RUN_A_FROZEN_WARMUP_ML` | 53 | 34.0% | 0.65 | **0.49** | -0.061R | $-31.36 | $-1000.27 | $661.74 | **$-1662.01** | $1751.23 (1.75%) | 0.0% | 100.0% | 0.0% |
| BRAIN-02 (Large-Cap Day) | `RUN_B_ONLINE_RL` | 11 | 54.5% | 2.19 | **1.61** | +0.039R | $+19.90 | $+352.90 | $134.00 | **$+218.90** | $212.83 (0.21%) | 0.0% | 100.0% | 0.0% |
| BRAIN-03 (Short Swing) | `RUN_A_FROZEN_WARMUP_ML` | 38 | 65.8% | 1.84 | **1.63** | +0.597R | $+209.51 | $+10056.72 | $2095.46 | **$+7961.26** | $4731.77 (4.39%) | 34.2% | 55.3% | 0.0% |
| BRAIN-03 (Short Swing) | `RUN_B_ONLINE_RL` | 38 | 65.8% | 1.84 | **1.63** | +0.597R | $+209.51 | $+10056.72 | $2095.46 | **$+7961.26** | $4731.77 (4.39%) | 34.2% | 55.3% | 0.0% |
| BRAIN-04 (Core Swing) | `RUN_A_FROZEN_WARMUP_ML` | 23 | 65.2% | 1.96 | **1.62** | +0.388R | $+135.82 | $+4411.76 | $1287.96 | **$+3123.80** | $2400.25 (2.33%) | 26.1% | 56.5% | 0.0% |
| BRAIN-04 (Core Swing) | `RUN_B_ONLINE_RL` | 23 | 65.2% | 1.96 | **1.62** | +0.388R | $+135.82 | $+4411.76 | $1287.96 | **$+3123.80** | $2400.25 (2.33%) | 26.1% | 56.5% | 0.0% |
| ALL_BUCKETS_COMBINED | `RUN_A_FROZEN_WARMUP_ML` | 378 | 37.6% | 1.34 | **0.92** | +0.015R | $-9.29 | $+11860.17 | $15373.18 | **$-3513.00** | $19467.90 (19.47%) | 5.3% | 89.9% | 2.6% |
| ALL_BUCKETS_COMBINED | `RUN_B_ONLINE_RL` | 313 | 39.0% | 1.46 | **0.99** | +0.037R | $-1.64 | $+14164.03 | $14676.62 | **$-512.59** | $16467.49 (16.47%) | 6.4% | 87.9% | 3.2% |

## Horizon Tier Rollups (`RUN_B_ONLINE_RL`)

| Tier | Run Mode | Trades | Win% | Gross PF | Net Post-Cost PF | Net Exp (R) | Net Exp ($) | Gross PnL ($) | Cost Drag ($) | Net PnL ($) | Max DD ($ / %) | Backstop% | Neural-Exit% | EOD Flatten% |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TIER: SCALP_0DTE | `RUN_B_ONLINE_RL` | 52 | 30.8% | 0.56 | **0.40** | -0.108R | $-60.86 | $-1957.95 | $1206.66 | **$-3164.60** | $3164.60 (3.16%) | 0.0% | 100.0% | 0.0% |
| TIER: DAY (BRAIN-01 + BRAIN-02) | `RUN_B_ONLINE_RL` | 200 | 33.0% | 1.16 | **0.47** | -0.072R | $-42.17 | $+1653.50 | $10086.54 | **$-8433.04** | $8789.19 (8.79%) | 0.5% | 94.5% | 5.0% |
| TIER: SWING (BRAIN-03 + BRAIN-04) | `RUN_B_ONLINE_RL` | 61 | 65.6% | 1.87 | **1.63** | +0.518R | $+181.72 | $+14468.48 | $3383.42 | **$+11085.06** | $7132.02 (6.43%) | 31.1% | 55.7% | 0.0% |

## Factored Vehicle Performance Split (`V_SHARES` vs `V_OPTION`)

| Run Mode | Vehicle | Trades | Win% | Gross PF | Net Post-Cost PF | Net Exp (R) | Avg Cost/Trade ($) | Total Gross PnL ($) | Total Cost Drag ($) | Total Net PnL ($) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `RUN_A_FROZEN_WARMUP_ML` | **V_SHARES** | 378 | 37.6% | 1.34 | **0.92** | +0.015R | $40.67 | $+11860.17 | $15373.18 | **$-3513.00** |
| `RUN_A_FROZEN_WARMUP_ML` | **V_OPTION** | 0 | 0.0% | 0.00 | **0.00** | +0.000R | $0.00 | $+0.00 | $0.00 | **$+0.00** |
| `RUN_B_ONLINE_RL` | **V_SHARES** | 313 | 39.0% | 1.46 | **0.99** | +0.037R | $46.89 | $+14164.03 | $14676.62 | **$-512.59** |
| `RUN_B_ONLINE_RL` | **V_OPTION** | 0 | 0.0% | 0.00 | **0.00** | +0.000R | $0.00 | $+0.00 | $0.00 | **$+0.00** |

## A/B Online Learning Delta (`RUN_B_ONLINE_RL` minus `RUN_A_FROZEN_WARMUP_ML`)

| Bucket | Δ Trades | Δ Win% | Δ Net PF | Δ Net Exp (R) | Δ Net PnL ($) | Δ Max DD ($) |
|---|---:|---:|---:|---:|---:|---:|
| SCALP_0DTE (1m) | -2 | -1.0% | -0.04 | -0.019R | **$-207.10** | $+0.00 |
| SCALP_0DTE (5m) | -21 | -3.7% | +0.08 | +0.047R | **$+1443.62** | $-1403.32 |
| BRAIN-01 (Penny Day) | +0 | +0.0% | -0.00 | +0.000R | **$-117.01** | $+116.91 |
| BRAIN-02 (Large-Cap Day) | -42 | +20.6% | +1.11 | +0.100R | **$+1880.91** | $-1538.40 |
| BRAIN-03 (Short Swing) | +0 | +0.0% | +0.00 | +0.000R | **$+0.00** | $+0.00 |
| BRAIN-04 (Core Swing) | +0 | +0.0% | +0.00 | +0.000R | **$+0.00** | $+0.00 |
| **DATASET_1_1M_TOTAL** | — | — | — | — | **$-207.10** | — |
| **DATASET_2_5M_1H_TOTAL** | — | — | — | — | **$+3207.52** | — |
| **ALL_BUCKETS_COMBINED** | **-65** | **+1.4%** | **+0.07** | **+0.023R** | **$+3000.41** | **$-3000.41** |

## Loss-Autopsy -> Re-Test -> Generation Promotion Ledger

| Gen ID | Phase | Brain | Red Trades Audited | Entry vs Exit Fault Split | Saliency (`|dL/dx|` / `|dL/dh|`) | Re-Test Net PF (Before -> After) | Red Loss Reduced ($) | Winners Preserved | Outcome |
|---:|---|---|---:|---|---|---|---:|---|---|
| 1 | `WARMUP_CAL` | BRAIN-01 (Penny Day) | 18 | `11 Entry / 7 Exit` | `0.1042 / 1.1126` | `0.67 -> 0.69` | $+49.97 | `11->13 wins (+2.33R -> +2.29R)` | `ROLLED_BACK` |
| 1 | `OOS_D1` | SCALP_0DTE (1m) | 2 | `1 Entry / 1 Exit` | `0.1428 / 1.1973` | `6.33 -> 3.70` | $-1369.33 | `9->32 wins (+2.64R -> +9.39R)` | `ROLLED_BACK` |
| 1 | `OOS_D4` | SCALP_0DTE (1m) | 7 | `5 Entry / 2 Exit` | `0.1956 / 3.0630` | `1.15 -> 1.94` | $-2034.20 | `10->37 wins (+2.92R -> +10.81R)` | `ROLLED_BACK` |
| 1 | `OOS_D5` | SCALP_0DTE (1m) | 15 | `10 Entry / 5 Exit` | `0.1375 / 2.6254` | `0.67 -> 1.32` | $-3077.23 | `12->45 wins (+3.96R -> +13.48R)` | `ROLLED_BACK` |
| 1 | `OOS_D6` | SCALP_0DTE (1m) | 17 | `12 Entry / 5 Exit` | `0.1310 / 2.5899` | `0.66 -> 2.32` | $+1907.05 | `14->20 wins (+4.33R -> +6.77R)` | `ROLLED_BACK` |
| 2 | `OOS_D8` | SCALP_0DTE (1m) | 21 | `16 Entry / 5 Exit` | `0.1251 / 2.4556` | `0.79 -> 2.19` | $+2231.98 | `19->21 wins (Net -1.52R -> +3.16R)` | **PROMOTED** |
| 2 | `OOS_D10` | SCALP_0DTE (1m) | 22 | `16 Entry / 6 Exit` | `0.2514 / 1.5972` | `1.97 -> 1.97` | $+25.13 | `21->21 wins (+6.59R -> +6.54R)` | `ROLLED_BACK` |
| 1 | `OOS_D1` | SCALP_0DTE (5m) | 2 | `2 Entry / 0 Exit` | `0.0579 / 0.5741` | `54.13 -> 2.96` | $-1377.68 | `8->29 wins (+1.46R -> +7.43R)` | `ROLLED_BACK` |
| 1 | `OOS_D1` | BRAIN-01 (Penny Day) | 23 | `12 Entry / 11 Exit` | `0.0918 / 0.9940` | `1.76 -> 1.66` | $-51.88 | `26->25 wins (+6.50R -> +6.26R)` | `ROLLED_BACK` |
| 1 | `OOS_D1` | BRAIN-02 (Large-Cap Day) | 10 | `5 Entry / 5 Exit` | `0.0191 / 0.2939` | `3.73 -> 7.08` | $+438.41 | `12->9 wins (+5.10R -> +4.14R)` | `ROLLED_BACK` |
| 1 | `OOS_D2` | BRAIN-01 (Penny Day) | 27 | `15 Entry / 12 Exit` | `0.1003 / 1.1024` | `1.25 -> 0.97` | $-480.88 | `26->24 wins (+6.19R -> +5.55R)` | `ROLLED_BACK` |
| 1 | `OOS_D3` | BRAIN-01 (Penny Day) | 28 | `16 Entry / 12 Exit` | `0.1039 / 1.1254` | `1.15 -> 1.00` | $-219.47 | `25->24 wins (+5.70R -> +5.34R)` | `ROLLED_BACK` |
| 1 | `OOS_D4` | SCALP_0DTE (5m) | 2 | `2 Entry / 0 Exit` | `0.0579 / 0.5741` | `54.13 -> 12.22` | $-50.84 | `8->9 wins (+1.46R -> +1.44R)` | `ROLLED_BACK` |
| 1 | `OOS_D4` | BRAIN-01 (Penny Day) | 29 | `17 Entry / 12 Exit` | `0.1021 / 1.1100` | `1.02 -> 0.98` | $-462.17 | `23->23 wins (+5.14R -> +5.33R)` | `ROLLED_BACK` |
| 1 | `OOS_D4` | BRAIN-02 (Large-Cap Day) | 9 | `5 Entry / 4 Exit` | `0.0207 / 0.3178` | `3.49 -> 6.39` | $+433.79 | `11->6 wins (+4.51R -> +3.30R)` | `ROLLED_BACK` |
| 1 | `OOS_D5` | BRAIN-01 (Penny Day) | 30 | `18 Entry / 12 Exit` | `0.1041 / 1.1339` | `0.96 -> 1.04` | $+427.62 | `22->21 wins (+4.87R -> +4.56R)` | `ROLLED_BACK` |
| 1 | `OOS_D5` | BRAIN-02 (Large-Cap Day) | 9 | `5 Entry / 4 Exit` | `0.0223 / 0.3478` | `2.71 -> 2.53` | $-116.56 | `10->23 wins (+3.87R -> +3.98R)` | `ROLLED_BACK` |
| 1 | `OOS_D6` | BRAIN-01 (Penny Day) | 31 | `19 Entry / 12 Exit` | `0.1024 / 1.1151` | `0.85 -> 0.92` | $+337.24 | `21->20 wins (+4.45R -> +4.31R)` | `ROLLED_BACK` |
| 2 | `OOS_D7` | SCALP_0DTE (5m) | 3 | `2 Entry / 1 Exit` | `0.0473 / 0.5201` | `5.40 -> 56.94` | $+96.26 | `7->8 wins (Net +0.85R -> +1.10R)` | **PROMOTED** |
| 1 | `OOS_D7` | BRAIN-01 (Penny Day) | 31 | `20 Entry / 11 Exit` | `0.1062 / 1.1498` | `0.65 -> 0.70` | $+337.15 | `19->18 wins (+3.53R -> +3.39R)` | `ROLLED_BACK` |
| 2 | `OOS_D7` | BRAIN-02 (Large-Cap Day) | 10 | `5 Entry / 5 Exit` | `0.0225 / 0.3552` | `2.44 -> 6.81` | $+618.59 | `9->5 wins (Net +2.15R -> +2.26R)` | **PROMOTED** |
| 1 | `OOS_D9` | BRAIN-01 (Penny Day) | 33 | `22 Entry / 11 Exit` | `0.1026 / 1.1048` | `0.62 -> 0.66` | $+337.28 | `20->19 wins (+3.53R -> +3.39R)` | `ROLLED_BACK` |
| 2 | `OOS_D10` | SCALP_0DTE (5m) | 0 | `0 Entry / 0 Exit` | `0.1539 / 2.5502` | `12.00 -> 12.00` | $+0.00 | `6->5 wins (+0.37R -> +0.36R)` | `ROLLED_BACK` |
| 1 | `OOS_D10` | BRAIN-01 (Penny Day) | 35 | `24 Entry / 11 Exit` | `0.1043 / 1.1231` | `0.56 -> 0.53` | $-2246.30 | `20->22 wins (+3.53R -> +5.10R)` | `ROLLED_BACK` |
| 3 | `OOS_D10` | BRAIN-02 (Large-Cap Day) | 4 | `2 Entry / 2 Exit` | `0.0230 / 0.3599` | `6.24 -> 8.22` | $+52.82 | `4->4 wins (Net +2.03R -> +2.10R)` | **PROMOTED** |
| 1 | `OOS_D12` | BRAIN-01 (Penny Day) | 37 | `25 Entry / 12 Exit` | `0.1057 / 1.1266` | `0.54 -> 0.45` | $-1944.05 | `20->26 wins (+3.53R -> +4.08R)` | `ROLLED_BACK` |
| 2 | `OOS_D13` | SCALP_0DTE (5m) | 0 | `0 Entry / 0 Exit` | `0.1274 / 2.1521` | `12.00 -> 4.74` | $-191.52 | `7->5 wins (+0.46R -> +1.66R)` | `ROLLED_BACK` |
| 1 | `OOS_D13` | BRAIN-01 (Penny Day) | 38 | `26 Entry / 12 Exit` | `0.1066 / 1.1302` | `0.54 -> 0.41` | $-2351.40 | `20->24 wins (+3.53R -> +4.12R)` | `ROLLED_BACK` |
| 3 | `OOS_D13` | BRAIN-02 (Large-Cap Day) | 3 | `1 Entry / 2 Exit` | `0.0182 / 0.2265` | `11.32 -> 1941.23` | $+115.22 | `4->3 wins (+2.41R -> +1.93R)` | `ROLLED_BACK` |
| 2 | `OOS_D16` | SCALP_0DTE (5m) | 1 | `1 Entry / 0 Exit` | `0.0103 / 0.1123` | `4.99 -> 4.74` | $-123.20 | `7->5 wins (+0.60R -> +1.66R)` | `ROLLED_BACK` |
| 3 | `OOS_D16` | BRAIN-02 (Large-Cap Day) | 2 | `0 Entry / 2 Exit` | `0.0064 / 0.0277` | `2037.63 -> 1941.23` | $+0.06 | `3->3 wins (+2.23R -> +1.94R)` | `ROLLED_BACK` |
| 4 | `OOS_D19` | BRAIN-02 (Large-Cap Day) | 2 | `0 Entry / 2 Exit` | `0.0064 / 0.0277` | `0.00 -> 12.00` | $+0.60 | `0->2 wins (Net -0.00R -> +0.08R)` | **PROMOTED** |
| 2 | `OOS_D21` | SCALP_0DTE (5m) | 2 | `2 Entry / 0 Exit` | `0.0193 / 0.2692` | `0.80 -> 0.87` | $-2319.95 | `3->30 wins (+0.31R -> +3.90R)` | `ROLLED_BACK` |
| 2 | `OOS_D24` | SCALP_0DTE (5m) | 4 | `3 Entry / 1 Exit` | `0.0356 / 0.3574` | `0.63 -> 0.80` | $-241.70 | `3->4 wins (+0.31R -> +0.77R)` | `ROLLED_BACK` |
| 4 | `OOS_D24` | BRAIN-02 (Large-Cap Day) | 2 | `1 Entry / 1 Exit` | `0.0057 / 0.0896` | `3.58 -> 1.07` | $-3415.76 | `1->22 wins (+0.35R -> +6.78R)` | `ROLLED_BACK` |
| 2 | `OOS_D25` | SCALP_0DTE (5m) | 7 | `6 Entry / 1 Exit` | `0.0429 / 0.5054` | `0.41 -> 0.88` | $-95.58 | `3->4 wins (+0.31R -> +0.85R)` | `ROLLED_BACK` |
| 3 | `OOS_D30` | SCALP_0DTE (5m) | 9 | `7 Entry / 2 Exit` | `0.0347 / 0.4086` | `0.37 -> 2.08` | $+204.89 | `3->10 wins (Net -0.54R -> +0.55R)` | **PROMOTED** |
| 4 | `OOS_D30` | BRAIN-02 (Large-Cap Day) | 2 | `1 Entry / 1 Exit` | `0.0057 / 0.0896` | `3.58 -> 0.76` | $-4139.88 | `1->18 wins (+0.35R -> +5.52R)` | `ROLLED_BACK` |
| 4 | `OOS_D31` | BRAIN-02 (Large-Cap Day) | 3 | `2 Entry / 1 Exit` | `0.0096 / 0.1888` | `1.60 -> 0.74` | $+51.52 | `2->1 wins (+0.66R -> +0.23R)` | `ROLLED_BACK` |
| 3 | `OOS_D36` | SCALP_0DTE (5m) | 4 | `4 Entry / 0 Exit` | `0.0603 / 0.4436` | `3.48 -> 1.05` | $-81.55 | `10->7 wins (+1.12R -> +0.48R)` | `ROLLED_BACK` |
| 4 | `OOS_D37` | BRAIN-02 (Large-Cap Day) | 3 | `2 Entry / 1 Exit` | `0.0096 / 0.1888` | `1.90 -> 0.69` | $-722.90 | `4->8 wins (+0.79R -> +1.19R)` | `ROLLED_BACK` |
| 3 | `OOS_D38` | SCALP_0DTE (5m) | 7 | `7 Entry / 0 Exit` | `0.0627 / 0.4503` | `1.65 -> 1.56` | $-13.02 | `9->8 wins (+1.01R -> +0.99R)` | `ROLLED_BACK` |
| 3 | `OOS_D40` | SCALP_0DTE (5m) | 7 | `7 Entry / 0 Exit` | `0.0399 / 0.2829` | `1.05 -> 0.52` | $+96.49 | `7->4 wins (+0.59R -> +0.20R)` | `ROLLED_BACK` |

## Autonomous 4-Vector Loss-Autopsy Attribution (`RUN_B_ONLINE_RL`)

| Bucket | Friction Inversion | Holding / EOD Decay | Vehicle Regret | Paralysis Regret | Flat Windows Replayed | Cost Sens (Before -> After) | Promoted Gens |
|---|---:|---:|---:|---:|---:|---|---:|
| SCALP_0DTE (1m) | 28 | 11 | 4 | 0 | 192 | `0.736 -> 0.707` | 2 |
| SCALP_0DTE (5m) | 16 | 3 | 0 | 64 | 703 | `1.113 -> 1.124` | 3 |
| BRAIN-01 (Penny Day) | 113 | 70 | 0 | 0 | 1056 | `1.226 -> 1.226` | 1 |
| BRAIN-02 (Large-Cap Day) | 8 | 9 | 0 | 32 | 601 | `1.048 -> 1.006` | 4 |
