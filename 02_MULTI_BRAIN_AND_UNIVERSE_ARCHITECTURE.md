# 02 — Multi-Brain Orchestrator, Custom Universe & Continuous Evolution
**Version:** 1.0 (Living Blueprint)  
**Core Mandate:** Selectable Goal-Horizon AI Brains + 100% Customizable Market Universe + Non-Stop Adaptation.

---

## 1. The 5 Selectable Specialized AI Brains
Each Brain shares the same "Zero-Indicator / Open-Minded" philosophy (`01_CORE_PHILOSOPHY`), meaning **none of them have hardcoded indicator rules**. They differ strictly by their **Target Time Horizon (Reward Window)** and the **Temporal Resolution** of the raw data they study, keeping their memory banks 100% isolated.

| Brain ID | Selectable Brain Name | Raw Data Temporal Lens | Target Payoff Horizon | Dual Asset Output |
| :--- | :--- | :--- | :--- | :--- |
| **BRAIN-01** | **Penny & Sub-Penny Anomaly Brain** | Tick, 1m, 3m, 5m, Daily + Float/Turnover | Minutes to Multi-Day Parabolic Runs | Shares (Micro-Cap / OTC / Sub-Penny) |
| **BRAIN-02** | **Intraday Day-Trading Brain** | Tick, 1m, 5m, 15m + Options 0DTE/Weekly + `$TICK/$VOLD` | `5 Minutes` to `6.5 Hours` (Flat by Close) | Shares + Single-Leg Calls/Puts & Scalps |
| **BRAIN-03** | **Short-Term Swing Brain** | 15m, 1H, 4H, Daily + Options Flow + Short Float | `1 to 5 Trading Days` | Shares + Short-Dated Calls/Puts & Debit Spreads |
| **BRAIN-04** | **Core Swing Trading Brain** | 4H, Daily, Weekly + Cross-Sector Relative Flow | `1 to 8 Weeks` | Shares + Multi-Week Options / Diagonals / Spreads |
| **BRAIN-05** | **Long-Term Portfolio & Wealth Brain** | Daily, Weekly, Monthly + Raw Revenue/FCF/Macro Data | `6 Months to 3+ Years` | Core Equity Shares + LEAPS |

*Extensibility Note:* The architecture supports creating unlimited additional custom Brains via configuration without altering core code.

---

## 2. The "Everything Customizable" Universe & Filter Engine
Inside any selected Brain, the user can configure what slice of the market the Brain scans in real time or backtests against:

### A. Market & Index Universe Selectors (Mix & Match)
* **`[Full Market Auto-Scan]`**: Scans all ~11,000+ US Listed Equities & OTC tickers via a Two-Stage Wide-Net Snapshot $\rightarrow$ Deep WebSocket Promotion funnel.
* **`[Preset Index Baskets]`**: One-click toggles for **S&P 500**, **Nasdaq 100 (NDX)**, **Russell 2000 (IWM)**, **Dow 30**, or **Sector Baskets** (Semis, Biotech, Energy, Financials, Mag-7).
* **`[Custom User Watchlists]`**: User-defined ticker lists.

### B. Precision Numeric Range Controls
* **Price Filter:** Supports 5-decimal precision from **`$0.00001`** (sub-penny / trips) up to **`$10,000.00+`** (e.g., set `$0.00001` to `$5.00` for Penny Brain, or `$20.00` to `No Max` for Nasdaq 100 Swing Brain).
* **Liquidity & Volume Floor:** Customizable Min Daily Share Volume and Min Daily Dollar Volume so illiquid traps are filtered out automatically.
* **Asset Capability Toggles:** `All Equities` | `Optionable Stocks Only` | `High Short Interest Only` | `Low Float Only`.

---

## 3. The 3-Speed Continuous Self-Learning Loop
The system never stays static. Every Brain continuously evolves based on live market evidence:

1. **Speed 1 — Intra-Day Real-Time Adaptation (Tick-by-Tick):**
   - Watches every open signal (`IN-TRADE`) bar-by-bar. If opposing absorption or tape exhaustion prints before the hard target, it fires an instant **Live Dynamic Exit**.
   - Adjusts confidence thresholds dynamically if live market internals shift into high-whipsaw chop.
2. **Speed 2 — Nightly Post-Market Autopsy (Daily at 4:15 PM EST):**
   - Grades every live signal fired that day (Actual MFE, MAE, Hold Time, and P&L).
   - **Hard-Negative Mining:** Re-analyzes any losing trade (bull/bear trap) to identify the subtle raw fingerprint that warned of failure.
   - **Missed-Move Discovery:** Scans the day's top market gainers/losers that the AI *missed*, rewinds their charts, and incorporates their pre-move traces into the memory bank.
3. **Speed 3 — Weekend Evolutionary Tournament (Champion vs. Challenger):**
   - Spawns mutated Challenger models tested against recent out-of-sample walk-forward market regimes; promotes a Challenger to live production only if it beats the Champion on Net Profit, Win %, and Drawdown control.

---

## 4. Future Additions / Brainstorm Notes
* [ ] *Add additional Brain types or custom universe rules here...*
