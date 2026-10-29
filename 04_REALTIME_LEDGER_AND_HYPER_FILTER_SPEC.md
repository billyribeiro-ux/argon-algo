# 04 — Real-Time Execution Ledger, Positions Opened & Export Engine Spec
**Version:** 1.1 (Updated with Dedicated Positions Opened Monitor & CSV/Excel `.xlsx` Exporter)  
**Core Mandate:** Radical Transparency. Dedicated Live "Positions Opened" Tracking, Complete Historical Ledger, Sub-10ms Hyper-Filtering, and 1-Click CSV / Multi-Sheet Excel Exports.

---

## 1. Master Executive Scoreboard (Top-Level Rollup)
Recalculates instantaneously (`<10ms`) whenever any filter, date window, or Brain is toggled. Tracks both closed history and live ticking open positions:

* **Net Profit ($ & %):** Total Realized Net Dollar Profit (`Gross Profit - Gross Loss - Estimated Slippage/Commissions`), Total Unrealized Open P&L ($ & %), and Combined Total Net Equity.
* **Win / Loss Breakdown:**
  - `Win Rate (%)` (`Total Wins / Total Closed Trades`)
  - `Total Closed Trades`, `Total Wins (# & $)`, `Total Losses (# & $)`, `Breakevens`
  - **`Positions Opened Counter`**: `Active Open Positions (#)` + `Live Ticking Open Net P&L ($ & %)`
* **Expectancy & Efficiency Metrics:**
  - `Profit Factor` (`Gross Profit $ / Gross Loss $`)
  - `Average Winner ($ / % / R-Multiple)` vs. `Average Loser ($ / % / R-Multiple)`
  - `Average Hold Time` (Displayed overall + split by *Winning Trades* vs. *Losing Trades*, formatted dynamically in `Minutes/Hours` for Day Trades and `Days/Weeks/Months` for Swings/Long-Term)
  - `Max Drawdown ($ & %)` & `Win/Loss Streak Counter`
* **Interactive Sizing Simulator:** Live slider to set *Position Sizing Mode* (`Fixed $ Per Trade`, `Fixed Risk $ Per Trade`, or `% of Account Equity`) and toggle between **`[Stock Shares View]`**, **`[Matched Options View]`**, or **`[Combined View]`**.

---

## 2. Dedicated "Positions Opened" Live Monitor (Active Trades Deck)
Every signal generated in real time immediately populates the **`POSITIONS OPENED`** table (can be docked above the Historical Ledger, viewed as a primary tab, or popped out to its own monitor/side-by-side split).

### Columns Tracked Live in `Positions Opened`:
1. **`Position Opened Date & Time`**: Exact timestamp down to the millisecond (`YYYY-MM-DD HH:MM:SS.ms EST`).
2. **`Live Hold Time (Ticking)`**: Real-time duration counter since entry (`01h 14m 08s` or `4d 06h 22m`).
3. **`Brain & Strategy`**: `Penny` | `Day Trade` | `Short Swing` | `Core Swing` | `Long-Term`.
4. **`Ticker & Direction`**: Symbol (`NVDA`), Index/Exchange tag, `LONG` or `SHORT`.
5. **`Stock Execution & Live Progress`**:
   - `Entry Price ($)`
   - `Live Current Price ($)` (ticking real-time)
   - `AI Structural Stop ($ & % distance from current)`
   - `Target 1 / Target 2 ($ & visual progress bar from Stop -> Entry -> Target)`
6. **`Matched Options Live Tracking`**:
   - `Option Contract` (e.g., `NVDA 2026-11-06 $145C`)
   - `Contract Entry Premium ($)` vs. `Live Bid/Ask Mid Premium ($)`
   - `Contract Stop ($)` & `Contract Target ($)`
7. **`Live Unrealized P&L`**:
   - `Open Stock P&L ($ & %)`
   - `Open Options P&L ($ & %)`
   - `Current R-Multiple` (e.g., `+1.9R`)
   - `Max Heat Taken So Far (MAE %)` & `Peak Open Gain So Far (MFE %)`
8. **`Live AI Trace Health Badge`**:
   - Real-time neural telemetry evaluating incoming candles: `🟢 TRACE ACCELERATING (HOLD)` | `🟡 T1 HIT - TRAILING STOP RAISED` | `🟠 BUYING EXHAUSTION DETECTED - PREPARING EXIT`.

*Automatic Transition Rule:* The exact millisecond an Open Position hits its Target, Stop-Loss, or an AI Dynamic Exit, it transitions from `Positions Opened` into the `Closed Historical Ledger` with all final metrics permanently locked.

---

## 3. Historical Closed Trade Ledger Schema (Every Completed Row)
Every completed trade in the ledger logs the full immutable record:

1. **Final Outcome Badge:** `CLOSED - TARGET HIT` | `CLOSED - AI DYNAMIC EXIT` | `CLOSED - TRAILING STOP` | `CLOSED - STOPPED OUT`
2. **Brain & Horizon Tag:** `Penny` | `Day Trade` | `Short Swing` | `Core Swing` | `Long-Term`
3. **Ticker & Direction:** `Symbol`, `Exchange/Index`, `LONG` or `SHORT`
4. **Full Timestamps & Hold Duration:**
   - `Position Opened Date & Time` (`YYYY-MM-DD HH:MM:SS.ms EST`)
   - `Position Closed Date & Time` (`YYYY-MM-DD HH:MM:SS.ms EST`)
   - `Total Hold Time` (`48m 12s` | `5d 3h` | `184d`)
5. **Stock Execution Levels:** `Entry Price ($)`, `Initial Structural Stop ($)`, `Target 1 ($)`, `Target 2 ($)`, `Actual Final Exit Price ($)`
6. **Matched Options Execution:** `Structure`, `Contract Spec`, `Option Entry ($)`, `Option Stop ($)`, `Option Final Exit ($)`
7. **Individual Realized Performance:**
   - `Individual Gross P&L ($)` & `Individual Net P&L ($)`
   - `Individual Stock Return (%)` & `Individual Option Return (%)`
   - `Realized R-Multiple` (`+3.4R` or `-1.0R`)
   - `Max Heat Taken (MAE %)` vs. `Peak Open Gain (MFE %)`
   - `Cumulative Running Net Profit ($)`
8. **Expandable "Fingerprint Trace" Snapshot:**
   - Replays the exact chart with Entry, Stop, Exit, and Hold Box plotted on the candles + the AI's Historical Precedent Match and 4-Stream Fingerprint breakdown.

---

## 4. The "Slice-Anything" Hyper-Filter Matrix
Every filter below combines freely and updates **Positions Opened**, **Closed Ledger**, **Executive Scoreboard**, and **Exports** in `<10ms`:

* **By View Mode:** `All Trades (Open + Closed)` | `Positions Opened Only` | `Closed History Only`
* **By AI Brain:** `All Brains` | `Penny` | `Day Trade` | `Short Swing` | `Core Swing` | `Long-Term`
* **By Outcome:** `All` | `Winners Only` | `Losers Only` | `Stopped Out` | `AI Dynamic Exit` | `>2R Winners`
* **By Index / Universe:** `All` | `S&P 500` | `Nasdaq 100` | `Russell 2000` | `OTC / Penny` | `Custom Basket`
* **By Ticker Search:** Multi-ticker filter (`TSLA, NVDA, AAPL`)
* **By Price Tier:** Custom min/max entry price (`$0.00001` to `$10,000+`)
* **By Hold Time Range:** `<15 mins` | `15m–1h` | `1h–6.5h` | `1–5 Days` | `1–8 Weeks` | `6+ Months`
* **By Date & Session Window:** Date Range Picker + Intraday Time-of-Day (`9:30–10:30`, `Midday`, `2:30–4:00`)
* **By AI Evidence Confidence:** Slider (`50%` to `99%+` historical trace similarity)
* **By Primary Fingerprint Driver:** `Gamma Squeeze` | `Short Squeeze` | `Volume Absorption` | `Internals Alignment`

---

## 5. Enterprise CSV & Multi-Sheet Excel (`.XLSX`) Export Engine
Accessible via 1-click buttons in the header of both `Positions Opened` and the `Master Ledger`:

### A. Export Scope Options
* **`Export Active Filtered Slice`**: Exports only the rows and scoreboard statistics matching the user's current active filters.
* **`Export Positions Opened Only`**: Instant snapshot of all currently open trades and live unrealized P&L.
* **`Export Complete Master Database`**: Exports every open and closed trade across all Brains.

### B. Supported Formats
1. **Standard Flat `.CSV` Export:**
   - RFC-4180 compliant UTF-8 CSV containing all 28+ columns (timestamps in ISO-8601, exact decimals down to `$0.00001`, stock + options metrics, and AI fingerprint tags) for immediate ingestion into Python, R, tax software, or external spreadsheets.
2. **Formatted Multi-Sheet Excel Workbook (`.XLSX` via `ExcelJS` / `rust_xlsxwriter`):**
   - **Tab 1 (`Executive_Summary`):** Formatted scoreboard with native Excel formulas (`SUM`, `AVERAGE`, `COUNTIF`), Win/Loss %, Net Profit, Profit Factor, and a comparative summary table across all 5 Brains.
   - **Tab 2 (`Positions_Opened`):** All currently open positions with live unrealized P&L, hold duration, stops, and targets.
   - **Tab 3 (`Closed_Trade_Ledger`):** Complete historical trade log with frozen top headers, auto-filters enabled on every column, currency/percentage cell formatting, and conditional green/red P&L styling.
   - **Tab 4 (`By_Brain_Breakdowns`):** Separate pre-organized tables for Penny, Day Trade, Short Swing, Core Swing, and Long-Term performance auditing.
